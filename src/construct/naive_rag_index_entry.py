import argparse
import os

from src.navieRAG import NaiveRAGAgent

LOG_NAME_ENV_KEY = "RAKG_LOGGER_NAME"
DEFAULT_LOGGER_NAME = "NaiveRAGIndex"

LOG_FILE_ENV_KEY = "RAKG_LOGGER_FILE"
DEFAULT_LOGGER_FILE = "NaiveRAGIndex.log"


# Let the entry point control the shared logger name used by dependent modules.
os.environ.setdefault(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME)
os.environ.setdefault(LOG_FILE_ENV_KEY, DEFAULT_LOGGER_FILE)


def build_naive_rag_index(json_path: str, output_dir: str, done_offset: int = 0) -> None:
    """Build naive RAG retrieval indices for all topics."""
    agent = NaiveRAGAgent()
    summary = agent.process_all_topics(
        json_path=json_path,
        output_dir=output_dir,
        done_offset=done_offset,
    )

    print("[INFO] Naive RAG index build completed.")
    print(f"[INFO] total_topics={summary.get('total_topics', 0)}")
    print(f"[INFO] processed_topics={summary.get('processed_topics', 0)}")
    print(f"[INFO] failed_topics_count={summary.get('failed_topics_count', 0)}")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build naive RAG retrieval index from topic JSON."
    )
    parser.add_argument(
        "--json-path",
        default="./data/raw/MINE_short10.json",
        help="Path to source topic JSON (list of {'topic', 'content'}).",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/short/naive/processed",
        help="Output directory to save naive rag index files.",
    )
    parser.add_argument(
        "--done-offset",
        type=int,
        default=0,
        help="Skip topics with index <= done_offset.",
    )

    args = parser.parse_args()
    build_naive_rag_index(
        json_path=args.json_path,
        output_dir=args.output_dir,
        done_offset=args.done_offset,
    )


if __name__ == "__main__":
    main()
