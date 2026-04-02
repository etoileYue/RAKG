import argparse
import json
import os
from typing import List

from src.navieRAG import NaiveRAGAgent

PRESET_QUESTIONS = [
    "蝴蝶的生命周期包括哪四个主要阶段？",
    "雌性蝴蝶通常把卵产在哪里？为什么？",
    "毛毛虫在幼虫阶段的主要活动是什么？",
    "在蛹阶段，毛毛虫会发生什么变化？",
    "成虫蝴蝶主要吃什么食物？它们在生态系统中有什么作用？",
]


def build_index_if_needed(agent: NaiveRAGAgent, data_path: str, output_dir: str, topic_idx: int) -> str:
    index_path = os.path.join(output_dir, "naive_rag_index", f"{topic_idx}.json")
    if os.path.exists(index_path):
        return index_path

    agent.process_all_topics(json_path=data_path, output_dir=output_dir)
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found after processing: {index_path}")
    return index_path


def run_qa(index_path: str, questions: List[str], top_k: int = 5) -> None:
    agent = NaiveRAGAgent()

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    for idx, question in enumerate(questions, start=1):
        qa_result = agent.answer_question(
            question=question,
            index_input=index_data,
            top_k=top_k,
        )

        print("\n" + "=" * 80)
        print(f"[Q{idx}] {question}")
        print(qa_result.get("formatted_answer", ""))


def main():
    parser = argparse.ArgumentParser(description="Naive RAG baseline entry")
    parser.add_argument(
        "--data-path",
        default="./data/raw/MINE_test.json",
        help="Path to source topic JSON (list of {'topic','content'}).",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/test/processed",
        help="Directory to save/read naive rag index files.",
    )
    parser.add_argument(
        "--index-path",
        default="",
        help="Optional direct index path. If set, skip index build.",
    )
    parser.add_argument(
        "--topic-idx",
        type=int,
        default=1,
        help="Topic index used for QA when auto-building index.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval size.")

    args = parser.parse_args()

    agent = NaiveRAGAgent()
    index_path = args.index_path.strip()
    if not index_path:
        index_path = build_index_if_needed(
            agent=agent,
            data_path=args.data_path,
            output_dir=args.output_dir,
            topic_idx=args.topic_idx,
        )

    run_qa(index_path=index_path, questions=PRESET_QUESTIONS, top_k=args.top_k)


if __name__ == "__main__":
    main()
