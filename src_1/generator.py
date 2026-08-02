"""
generator.py
============

検索(Retrieval)で得られたソース群をもとに、ローカルLLM
（デフォルト: Qwen/Qwen3-0.6B）を使ってグラウンディングされた
（＝検索結果に根拠を持つ）自然言語の回答を生成するモジュール。

処理の流れ:
  1. MinimalSearchResults（質問＋検索結果のソース位置）を受け取る
  2. 各ソースについて、実ファイルから該当する文字範囲だけを切り出す
  3. 「検索結果の外の知識を使わない」よう強く指示するシステムプロンプトと、
     切り出したコンテキスト＋質問を組み立てたユーザープロンプトを作る
  4. チャット形式のプロンプトをトークナイズし、LLMに生成させる
  5. 生成されたテキストだけ（プロンプト部分を除く）をデコードして返す
"""

import itertools
from pathlib import Path
from typing import List, Dict, Any, cast

import torch
from more_itertools import batched
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
)

from src.models import (
    MinimalSource,
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)
from src.retriever import Retriever


class Generator:
    """
    ローカルLLMを使ってグラウンディングされた自然言語の回答を生成するクラス。

    インスタンス生成時にトークナイザーとモデルを一度だけロードし、
    以降の generate_answers() 呼び出しではロード済みのモデルを使い回す
    （モデルのロードは数秒〜数十秒かかるため、質問のたびに読み込み直すのは避ける）。
    """

    def __init__(
        self,
        # 仕様上の必須モデル。他のモデルを使う場合も、
        # 最終的にQwen/Qwen3-0.6Bで動作することが要件になっているため、
        # デフォルト値としてこれを指定している。
        model_name: str = "Qwen/Qwen3-0.6B",
        batch_size: int = 1,
        max_new_tokens: int = 256,
        k: int = 5,
    ) -> None:
        """
        LLMマネージャーの初期化。

        Args:
            model_name: 使用するHugging Face上のモデル名
                （デフォルトは仕様で指定されている Qwen/Qwen3-0.6B）
            batch_size: 一度に推論するプロンプトの数。
                大きくすると1問あたりの処理は速くなるが、
                メモリ使用量とレイテンシのトレードオフがある。
            max_new_tokens: 生成する回答の最大トークン数
            k: デフォルトで各回答に使うソース数
                （generate_answers() 呼び出し時にkを指定しない場合に使われる）
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.k = k
        # GPUが使える環境ならCUDAを、なければCPUを使う
        # （campus機はCPUのみのため、CPUフォールバックが必須）
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading LLM model: {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left"
        )
        # Qwen系のモデルなど、pad_tokenが定義されていない場合があるため、
        # その場合はeos_tokenで代用する（バッチ推論時のパディングに必要）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            # CUDAが使えるならHugging Face側に device_map の自動割り当てを
            # 任せ、CPU環境では明示的に"cpu"を指定する
            device_map="auto" if self.device == "cuda" else "cpu",
        )

    def _load_chunk(self, source: MinimalSource) -> str:
        """
        ファイルのパスと文字インデックスの範囲を指定して、
        実ファイルから該当部分の文字列を切り出す。

        検索結果（MinimalSource）自体はテキストの中身を持っておらず、
        「どのファイルの何文字目から何文字目まで」という位置情報しか
        持たないため、実際に回答を生成する段階でファイルを開いて
        中身を読み出す必要がある。

        Args:
            source: MinimalSourceオブジェクト（file_path, indices を含む）

        Returns:
            切り出されたテキスト(str)。
            ファイルが見つからない、または読み込みに失敗した場合は
            例外を投げずにエラーメッセージ文字列を返す
            （1件のソースの読み込み失敗で回答生成全体を止めないため）。
        """
        file_path = Path(source.file_path)
        if not file_path.exists():
            return f"[Error: File not found {source.file_path}]"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # last_character_index は inclusive（その文字を含む）なので、
            # Pythonのスライスに渡す際は +1 する必要がある
            return content[
                source.first_character_index:source.last_character_index + 1
            ]
        except Exception as e:
            return f"[Error loading chunk: {e}]"

    def _generate_prompt(
        self, result: MinimalSearchResults, k: int | None = None
    ) -> List[Dict[str, str]]:
        """
        LLMに与えるためのシステムプロンプトとユーザープロンプト
        （検索コンテキスト＋質問）を組み立てる。

        システムプロンプトでは「検索結果に書かれていないことは
        答えない」「知らなければ知らないと言う」ことを明示的に
        指示しており、これが回答のグラウンディング
        （検索結果に根拠を持たせること）を担保する仕組みになっている。

        Args:
            result: MinimalSearchResultsオブジェクト（質問と検索結果）
            k: このプロンプトで実際に使用するソース数。
                Noneの場合はインスタンス生成時に指定した self.k を使う。
                単発の generate_answers() 呼び出しごとに異なる k を
                使いたい場合（例: APIで毎リクエストごとに k が変わる場合）に、
                Generatorインスタンスを作り直さずに済むようにするための引数。

        Returns:
            チャット形式メッセージのリスト
            (形: [{"role": "system", "content": "..."}, ...])
        """
        effective_k = self.k if k is None else k
        system_prompt = {
            "role": "system",
            "content": (
                "You are a precise and helpful assistant. Answer the user's "
                "question using ONLY the retrieved context provided below. "
                "Follow these rules strictly:\n"
                "- If the answer is not in the context, say: \"I don't have "
                'enough information to answer that."\n'
                "- Do not use outside knowledge or make up information.\n"
                "- Keep answers concise and grounded in the provided text.\n"
                "- When possible, cite which document/source supports your "
                "answer."
            ),
        }

        # 検索結果のうち先頭 effective_k 件だけをコンテキストとして使う。
        # islice を使うことで、retrieved_sources がそれ以上の件数を
        # 持っていても不要なファイル読み込みをしないようにしている。
        chunks = [
            (source.file_path, self._load_chunk(source))
            for source in itertools.islice(
                result.retrieved_sources, effective_k
            )
        ]
        formatted_sources = []
        for i, (file_path, content) in enumerate(chunks, start=1):
            formatted_sources.append(
                f"[Source {i}] File: {file_path}\nContent: {content}\n"
            )
        context_str = "\n".join(formatted_sources)

        user_prompt = {
            "role": "user",
            "content": (
                "Retrieved Context:\n"
                "---\n"
                f"{context_str}"
                "---\n\n"
                f"Question: {result.question}\n\n"
                "Answer based only on the retrieved context above."
            ),
        }
        return [system_prompt, user_prompt]

    @torch.inference_mode()
    def generate_answers(
        self,
        search_results: List[MinimalSearchResults],
        k: int | None = None,
    ) -> List[str]:
        """
        検索結果のリストを受け取り、バッチ単位でLLMに推論を行って
        回答文字列のリストを生成する。

        Args:
            search_results: MinimalSearchResultsのリスト
                （1件が1つの質問に対応する）
            k: 各質問のプロンプト作成時に使用するソース数。
                Noneの場合はインスタンスの self.k を使う。
                （例: APIサーバーでは1つのGeneratorインスタンスを
                使い回しつつ、リクエストごとに異なる k を指定できる）

        Returns:
            生成された回答文字列のリスト (List[str])。
            search_results と同じ順序・同じ件数で対応する。
        """
        # 各質問についてチャット形式のプロンプトをあらかじめ全部組み立てておく
        prompt_messages = [
            self._generate_prompt(res, k=k) for res in search_results
        ]
        outputs: List[str] = []

        with torch.no_grad():
            # self.batch_size 件ずつまとめて推論することで、
            # 1件ずつ推論するよりGPU/CPUの利用効率を上げる
            for batch in batched(prompt_messages, self.batch_size):
                # チャットテンプレートを適用し、トークンID列に変換する。
                # padding=True でバッチ内の長さを揃え、
                # add_generation_prompt=True でモデルに
                # 「ここから回答を書き始めてください」という
                # 合図となるトークン列を付加する。
                prompt_tokens = self.tokenizer.apply_chat_template(
                    list(batch),
                    tokenize=True,
                    padding=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                assert isinstance(prompt_tokens, BatchEncoding)
                prompt_tokens = prompt_tokens.to(self.device)
                # 生成されたトークン列の先頭には入力プロンプトがそのまま
                # 含まれるため、入力の長さを記録しておき、後で切り落とす
                input_length = prompt_tokens["input_ids"].shape[1]
                generated_ids = cast(Any, self.model).generate(
                    **prompt_tokens, max_new_tokens=self.max_new_tokens
                )
                # 入力プロンプト部分を除いた「新しく生成された部分」だけを取り出す
                new_tokens = generated_ids[:, input_length:]
                decoded = self.tokenizer.batch_decode(
                    new_tokens, skip_special_tokens=True
                )
                outputs.extend(decoded)

        return outputs

    def answer_dataset(
        self, dataset_path: str, k: int = 5
    ) -> StudentSearchResultsAndAnswer:
        """
        データセット全体（質問のJSONファイル）を読み込み、
        検索(Retriever)から回答生成(Generator)までを一括で実行する。

        Retrieverの生成・検索・Generatorでの回答生成という
        パイプライン全体をこのメソッド1つで完結させたい場合に使う
        （CLIの answer_dataset コマンドの一部で利用される）。

        Args:
            dataset_path: 質問データセットのJSONファイルパス
            k: 検索・回答生成の両方で使用する上位件数

        Returns:
            StudentSearchResultsAndAnswer: search_dataset と
                answer_dataset を合わせた最終的な出力形式
        """
        retriever = Retriever()
        search_results_obj = retriever.search_dataset(dataset_path, k=k)

        print("Generating answers using LLM...")
        # ここで k を明示的に渡すことで、検索時に使った k と
        # 回答生成時にプロンプトへ詰め込むソース数を必ず一致させる。
        # （k を渡し忘れると、self.k のデフォルト値である5件だけが
        #  常に使われてしまい、k=10などを指定しても反映されないバグになる）
        answers = self.generate_answers(
            search_results_obj.search_results, k=k
        )

        minimal_answers: List[MinimalAnswer] = []
        for search_res, ans in zip(search_results_obj.search_results, answers):
            minimal_answers.append(
                MinimalAnswer(
                    question_id=search_res.question_id,
                    question=search_res.question,
                    retrieved_sources=search_res.retrieved_sources,
                    answer=ans,
                )
            )

        return StudentSearchResultsAndAnswer(
            search_results=minimal_answers, k=k
        )
