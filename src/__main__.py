"""
__main__.py
===========

RAGパイプラインのコマンドラインエントリポイント。
Python Fire を使い、RAGCLI クラスの各メソッドをそのままサブコマンドとして
公開している（例: `uv run python -m src index --max_chunk_size 2000`）。

コマンド:
    index          コーパスをインデックス化する
    search         単一クエリで検索する
    search_dataset データセット全体で検索する（結果をJSONに保存）
    answer         単一クエリで検索＋回答生成する
    answer_dataset データセット全体で検索＋回答生成する（結果をJSONに保存）
    evaluate       自分の検索結果をground truthと比較してrecall@kを計算する
                   （※ 本番の採点は moulinette が行うため、これは
                     あくまで開発中の自己検証用）
"""

import sys
import json
from pathlib import Path
import fire
from src.indexer import Indexer
from src.retriever import Retriever
from src.generator import Generator
from src.models import (
    MinimalSource,
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)


class RAGCLI:
    """
    Python Fireで公開するCLIのエントリクラス。
    各public methodがそのままサブコマンドになる。
    """

    def index(
            self,
            max_chunk_size: int = 2000,
            skip_vector: bool = False,
            debug: bool = False
            ) -> None:
        """
        コーパス（data/raw/ 配下）を読み込み、チャンクに分割して
        インデックスを構築・保存する。

        Args:
            max_chunk_size: 1チャンクあたりの最大文字数
            debug: Trueの場合、インデックス化されたチャンク数を表示する

        Note:
            失敗しても例外を外に投げず、標準エラー出力にメッセージを
            出すだけに留める（CLIが未処理の例外でクラッシュしないため）。
        """
        try:
            indexer = Indexer(
                max_chunk_size=max_chunk_size,
                skip_vector=skip_vector
                )
            indexer.index_corpus()
            if debug:
                print(
                    "📊 [DEBUG] Indexed chunks count: "
                    f"{len(indexer.chunks)}"
                )
        except Exception as e:
            print(f"Indexing failed: {e}", file=sys.stderr)

    def search(self, query: str = "", k: int = 5, debug: bool = False) -> None:
        """
        単一の質問文字列に対して、ハイブリッド検索（BM25＋セマンティック）
        を用いて関連度の高い上位 k 件を検索し、結果を標準出力に表示する。

        Args:
            query: 検索したい質問文字列
            k: 取得する上位件数
            debug: Trueの場合、各結果について実際に切り出されたテキストの
                    プレビューも表示する
        """
        if not query:
            print("Error: query cannot be empty.", file=sys.stderr)
            return

        print(f"Searching for: '{query}' (top {k})")
        try:
            retriever = Retriever()
            sources = retriever.search(query, k)

            if not sources:
                print("No results found.")
                return

            for i, src in enumerate(sources):
                print(
                    f"[{i+1}] {src.file_path} "
                    f"[{src.first_character_index}:{src.last_character_index}]"
                )

                if debug:
                    preview = self._load_preview_text(src)
                    indented = self._indent_text(preview, 4)
                    print(f"    └─ 📄 [TEXT SLICE]:\n{indented}")

        except Exception as e:
            print(f"Search failed: {e}", file=sys.stderr)

    def search_dataset(
        self,
        dataset_path: str = "",
        k: int = 5,
        save_directory: str = "",
        debug: bool = False
    ) -> None:
        """
        データセット（複数質問を含むJSON）を読み込み、
        一括で検索を実行して StudentSearchResults 形式のJSONとして保存する。

        Args:
            dataset_path: 質問データセットのJSONファイルパス
            k: 各質問ごとに取得する上位件数
            save_directory: 出力先ディレクトリ
                （データセットのファイル名がそのまま出力ファイル名になる）
            debug: このコマンドでは未使用（インターフェース統一のために保持）
        """
        if not dataset_path or not save_directory:
            print("Error: path required.", file=sys.stderr)
            return

        try:
            retriever = Retriever()
            results = retriever.search_dataset(dataset_path, k)

            file_name = Path(dataset_path).name
            save_dir_path = Path(save_directory)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            output_path = save_dir_path / file_name

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(results.model_dump_json(indent=2))

            print(f"Saved student_search_results to {output_path}")

        except Exception as e:
            print(f"Dataset search failed: {e}", file=sys.stderr)

    def answer(self, query: str = "", k: int = 5, debug: bool = False) -> None:
        """
        単一の質問に対して検索を行い、検索結果からプロンプト用の
        コンテキスト文字列を組み立て、LLMに渡して回答を生成する

        Args:
            query: 質問文字列（空文字はエラー扱い）
            k: 検索・回答生成の両方で使用する上位件数
            debug: Trueの場合、切り出された各ソースのテキストや、
                実際にLLMへ渡される完全なプロンプトを表示する
                （プロンプトの中身をデバッグしたいときに有用）
        """
        if not query:
            print("Error: query cannot be empty.", file=sys.stderr)
            return

        print(f"Answering query: '{query}' (using top {k} sources)")
        try:
            retriever = Retriever()
            sources = retriever.search(query, k)
            search_result = MinimalSearchResults(
                question_id="single-query",
                question=query,
                retrieved_sources=sources
            )
            # ここでは k をそのまま Generator のインスタンス生成時に
            # 渡している（＝self.k として保持される）ため、
            # 後続の generate_answers() でも同じ k がデフォルトとして使われる。
            generator = Generator(k=k)

            if debug:
                print("\n" + "=" * 60)
                print("🔍 [DEBUG] 文字列・コンテキスト構築の過程 (String Building)")
                print("=" * 60)
                for i, src in enumerate(sources, start=1):
                    preview = self._load_preview_text(src)
                    print(
                        f"\n[Source {i}] Path: {src.file_path}\n"
                        f"  Range: [{src.first_character_index}:"
                        f"{src.last_character_index}]"
                    )
                    print(
                        "  └ ✂️ 切り出されたテキスト文字列 "
                        f"({len(preview)} chars):\n"
                        f"{self._indent_text(preview, 4)}"
                    )

                print("\n" + "-" * 60)
                print("🤖 [DEBUG] 最終的なLLMプロンプト:")
                print("-" * 60)
                prompt_msgs = generator._generate_prompt(search_result)
                for msg in prompt_msgs:
                    print(f"[{msg['role'].upper()}]")
                    print(msg['content'])
                    print("-" * 40)

            answers = generator.generate_answers([search_result])
            print(f"\n💡 Answer:\n{answers[0]}")
        except Exception as e:
            print(f"Answering failed: {e}", file=sys.stderr)

    def answer_dataset(
        self,
        student_search_results_path: str = "",
        save_directory: str = "",
        debug: bool = False
    ) -> None:
        """
        既に検索済みの StudentSearchResults 形式のJSON
        （student_search_results_path）を読み込み、
        データセットに対して一括で回答生成を行い、
        StudentSearchResultsAndAnswer 形式のJSONとして保存する。

        Args:
            student_search_results_path: search_dataset の出力JSONのパス
            save_directory: 出力先ディレクトリ
            debug: このコマンドでは未使用（インターフェース統一のために保持）

        Note:
            入力JSONに "search_results" キーが含まれる場合はそれを
            StudentSearchResults として解釈して回答生成する。
            含まれない場合は、後方互換のため
            Generator.answer_dataset()（検索から一括実行するバージョン）
            にフォールバックする。
        """
        if not student_search_results_path or not save_directory:
            print("Error: paths are required.", file=sys.stderr)
            return

        print(f"Generating answers for: {student_search_results_path}")
        try:
            generator = Generator()
            input_path = Path(student_search_results_path)
            if not input_path.exists():
                raise FileNotFoundError(
                    f"Path not found: {student_search_results_path}"
                )

            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if "search_results" in data:
                # すでに検索済みの結果が渡された正規のケース：
                # JSONをパースしてpydanticモデルに変換してから回答生成する
                search_results_list = []
                for sr in data["search_results"]:
                    sources = [
                        MinimalSource(**s) for s in sr["retrieved_sources"]
                    ]
                    search_results_list.append(
                        MinimalSearchResults(
                            question_id=sr["question_id"],
                            question=sr["question"],
                            retrieved_sources=sources
                        )
                    )
                answers = generator.generate_answers(search_results_list)
                minimal_answers = []
                for sr, ans in zip(search_results_list, answers):
                    minimal_answers.append(
                        MinimalAnswer(
                            question_id=sr.question_id,
                            question=sr.question,
                            retrieved_sources=sr.retrieved_sources,
                            answer=ans
                        )
                    )
                result_obj = StudentSearchResultsAndAnswer(
                    search_results=minimal_answers,
                    k=data.get("k", 5)
                )
            else:
                # "search_results" キーがない＝検索前の生データセットが
                # 渡されたケース。Generator側で検索からやり直す。
                result_obj = generator.answer_dataset(str(input_path))

            file_name = input_path.name
            save_dir_path = Path(save_directory)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            output_path = save_dir_path / file_name

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result_obj.model_dump_json(indent=2))

            print(
                f"Saved student_search_results_and_answer to {output_path}"
            )

        except Exception as e:
            print(f"Dataset answering failed: {e}", file=sys.stderr)

    def evaluate(
        self,
        student_search_results_path: str = "",
        dataset_path: str = "",
        debug: bool = False
    ) -> None:
        """
        学生（自分）の検索結果と、正解データ（ground truth）を比較して
        ヒット率（Hit Rate、recall@kの簡易版）を計算する。

        あくまで開発中に自分の実装の良し悪しを素早く確認するための
        コマンドであり、本番の評価は仕様書にある通り
        moulinette 実行ファイルが行う（このコマンドのロジックとは
        別実装であり、細部が完全に一致するとは限らない）。

        Args:
            student_search_results_path: 自分の search_dataset の出力JSON
            dataset_path: 正解ソースを含む ground-truth データセットJSON
            debug: このコマンドでは未使用（インターフェース統一のために保持）

        判定ロジック:
            各質問について、自分の検索結果 (retrieved) の中に、
            正解ソース (gt_sources) と「同じファイルパスかつ
            文字範囲が重なる」ものが1つでもあればヒットとみなす。
        """
        if not student_search_results_path or not dataset_path:
            print("Error: both paths are required.", file=sys.stderr)
            return

        try:
            student_path = Path(student_search_results_path)
            gt_path = Path(dataset_path)

            if not student_path.exists():
                raise FileNotFoundError(
                    f"Student results not found: {student_path}"
                )
            if not gt_path.exists():
                raise FileNotFoundError(f"Dataset not found: {gt_path}")

            with open(student_path, "r", encoding="utf-8") as f:
                student_data = json.load(f)

            with open(gt_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)

            student_results = student_data.get("search_results", [])
            # 正解ソースを持つ質問（AnsweredQuestion）だけを
            # question_id をキーにした辞書に変換しておく
            gt_questions = {
                q["question_id"]: q
                for q in gt_data.get("rag_questions", [])
                if "sources" in q
            }

            total = 0
            hits = 0

            for sr in student_results:
                q_id = sr.get("question_id")
                if q_id not in gt_questions:
                    # 正解データが存在しない質問は評価対象から除外する
                    continue

                total += 1
                retrieved = sr.get("retrieved_sources", [])
                gt_sources = gt_questions[q_id]["sources"]

                matched = False
                for r_src in retrieved:
                    for gt_src in gt_sources:
                        if r_src["file_path"] == gt_src["file_path"]:
                            r_start = r_src["first_character_index"]
                            r_end = r_src["last_character_index"]
                            gt_start = gt_src["first_character_index"]
                            gt_end = gt_src["last_character_index"]

                            # 2つの区間 [r_start, r_end] と
                            # [gt_start, gt_end] が重なっているかどうかの判定。
                            # 「重ならない」条件（r_end < gt_start または
                            # r_start > gt_end）の否定＝「重なる」となる。
                            if not (r_end < gt_start or r_start > gt_end):
                                matched = True
                                break
                    if matched:
                        break

                if matched:
                    hits += 1

            hit_rate = (hits / total) * 100 if total > 0 else 0.0
            print("📊 Evaluation Results:")
            print(f"  - Total Questions Evaluated: {total}")
            print(f"  - Hits (Correct Source Found): {hits}")
            print(f"  - Hit Rate: {hit_rate:.2f}%")

        except Exception as e:
            print(f"Evaluation failed: {e}", file=sys.stderr)

    def _load_preview_text(self, source: MinimalSource) -> str:
        """
        デバッグ表示用に、ソースが指す文字範囲のテキストを
        安全に読み込むヘルパーメソッド。

        Args:
            source: プレビューしたい MinimalSource

        Returns:
            切り出されたテキスト。読み込みに失敗した場合は
            例外を投げずにエラーメッセージ文字列を返す。
        """
        try:
            with open(source.file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content[
                source.first_character_index:source.last_character_index + 1
            ]
        except Exception:
            return "[Could not load preview]"

    def _indent_text(self, text: str, spaces: int) -> str:
        """
        デバッグ表示の見た目を整えるためインデントを付与するヘルパー。
        表示が長くなりすぎないよう、先頭10行までに制限している。

        Args:
            text: インデントを付けたい元のテキスト
            spaces: インデントとして挿入する半角スペースの数

        Returns:
            各行の先頭にインデントを付与したテキスト（最大10行）
        """
        indent = " " * spaces
        lines = text.strip().splitlines()
        return "\n".join(f"{indent}{line}" for line in lines[:10])


def main() -> None:
    """
    CLIのエントリポイント。

    Python Fire に RAGCLI クラスを渡すことで、クラスの各メソッドを
    そのままサブコマンドとして扱えるようにする。
    Ctrl+C による中断や、Fire自体が捕捉できなかった予期しない例外も
    ここで最終的にキャッチし、非0の終了コードで終了させることで、
    未処理のトレースバックがそのまま表示されてしまう事態を避ける。
    """
    try:
        fire.Fire(RAGCLI)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
