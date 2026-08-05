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
    ローカルLLMの回答を生成
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        batch_size: int = 1,
        max_new_tokens: int = 256,
        k: int = 5,
    ) -> None:
        """
        Args:
            model_name: 使用するHugging Face上のモデル名
                （デフォルトは仕様で指定されている Qwen/Qwen3-0.6B）
            batch_size: 一度に推論するプロンプトの数。
                大きくすると1問あたりの処理は速くなるが、
                メモリ使用量とレイテンシのトレードオフがある。
            max_new_tokens: 生成する回答の最大トークン数
            k: デフォルトで各回答に使うソース数
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.k = k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading LLM model: {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto" if self.device == "cuda" else "cpu",
        )

    def _load_chunk(self, source: MinimalSource) -> str:
        """
        ファイルのパスと文字インデックスの範囲を指定して、
        実ファイルから該当部分の文字列を切り出すmethod
        Args:
            source: MinimalSource

        Returns:
            text
            ファイルが見つからない、または読み込みに失敗した場合は
            例外を投げずにエラーメッセージ文字列を返す。
        """
        file_path = Path(source.file_path)
        if not file_path.exists():
            return f"[Error: File not found {source.file_path}]"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            return content[
                source.first_character_index:source.last_character_index + 1
            ]
        except Exception as e:
            return f"[Error loading chunk: {e}]"

    def _generate_prompt(
        self, result: MinimalSearchResults, k: int | None = None
    ) -> List[Dict[str, str]]:
        """
        Args:
            result: MinimalSearchResults
            k: このプロンプトで実際に使用するソース数
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

        prompt_messages = [
            self._generate_prompt(res, k=k) for res in search_results
        ]
        outputs: List[str] = []

        with torch.no_grad():
            for batch in batched(prompt_messages, self.batch_size):
                prompt_tokens = self.tokenizer.apply_chat_template(
                    list(batch),
                    tokenize=True,
                    padding=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                assert isinstance(prompt_tokens, BatchEncoding)
                prompt_tokens = prompt_tokens.to(self.device)
                input_length = prompt_tokens["input_ids"].shape[1]
                generated_ids = cast(Any, self.model).generate(
                    **prompt_tokens, max_new_tokens=self.max_new_tokens
                )
                new_tokens = generated_ids[:, input_length:]
                decoded = self.tokenizer.batch_decode(
                    new_tokens, skip_special_tokens=True
                )
                outputs.extend(decoded)

        return outputs

    def answer_dataset(
        self, dataset_path: str, k: int = 5
    ) -> StudentSearchResultsAndAnswer:

        retriever = Retriever()
        search_results_obj = retriever.search_dataset(dataset_path, k=k)

        print("Generating answers using LLM...")
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
