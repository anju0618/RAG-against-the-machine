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
    エントリクラス
    """

    def index(self, max_chunk_size: int = 2000, debug: bool = False) -> None:
        """
        vLLMコーパスを読み込み、チャンクに分割してインデックスを構築・保存します。
        """
        try:
            indexer = Indexer(max_chunk_size=max_chunk_size)
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
        単一の質問文字列に対して、BM25を用いて関連度の高い上位 k 件を検索
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
        データセット（複数質問のJSON）を読み込み、一括で検索を実行し保存します。
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
        単一の質問に対して検索を行い、文字列（コンテキスト・プロンプト）を
        組み立ててLLMに渡し、回答を生成
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
                print("🤖 [DEBUG] 最終的なLLMプロンプト（入力文字列）の全貌:")
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
        データセットに対して一括で回答生成を行い、JSONに保存します。
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
        評価プレースホルダー
        """
        if not student_search_results_path or not dataset_path:
            print("Error: both paths are required.", file=sys.stderr)
            return
        print("Evaluating results...")
        print(f"Student Data: {student_search_results_path}")
        print(f"Ground Truth: {dataset_path}")

    def _load_preview_text(self, source: MinimalSource) -> str:
        """
        デバッグ表示用にソースのテキストを安全に読み込むヘルパー
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
        デバッグ表示の見た目を整えるためインデントを付与するヘルパー
        """
        indent = " " * spaces
        lines = text.strip().splitlines()
        return "\n".join(f"{indent}{line}" for line in lines[:10])


def main() -> None:
    """
    エントリポイント
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
