import argparse
import os
import sys
from typing import List
from src.kgAgent import NER_Agent

PRESET_QUESTIONS = [
    "蝴蝶的生命周期包括哪四个主要阶段？", # 蝴蝶的生命周期包括四个主要阶段：卵（egg）、幼虫（larva，也叫毛毛虫）、蛹（pupa，也叫蛹壳或蝶蛹）、成虫（adult butterfly）
    "雌性蝴蝶通常把卵产在哪里？为什么？", # 雌性蝴蝶通常把卵产在特定寄主植物叶子的背面，这样既能为孵化后的幼虫提供食物，又可以帮助保护卵免受捕食者的攻击
    "毛毛虫在幼虫阶段的主要活动是什么？", # 毛毛虫在幼虫阶段主要不停地吃叶子并快速生长，同时会多次蜕皮（molting），每次蜕皮后都会长出更大的新皮肤
    "在蛹阶段，毛毛虫会发生什么变化？", # 在蛹阶段，毛毛虫会形成一个叫做蛹壳（chrysalis）的保护结构，并在里面经历变态（metamorphosis）。它的身体组织会发生巨大变化，形成新的结构，如翅膀和触角，最终变成蝴蝶。
    "成虫蝴蝶主要吃什么食物？它们在生态系统中有什么作用？", # 成虫蝴蝶主要用长长的口器（proboscis）吸食花蜜。它们在生态系统中帮助植物授粉，并且也是鸟类、爬行动物和其他昆虫的食物来源，对生态系统的多样性有重要作用。
]


def run_qa(
    kg_path: str,
    questions: List[str],
    max_hop: int = 1,
    seed_top_k: int = 5,
    max_context_items: int = 30,
) -> None:
    ner_agent = NER_Agent()

    # 初始化一次图谱索引，后续问题复用同一个 cache_key。
    cache_key = ner_agent.initialize_qa_graph_index(kg_path, cache_key="default_qa_graph")
    print(f"[INFO] Graph index initialized. cache_key={cache_key}")

    for idx, question in enumerate(questions, start=1):
        qa_result = ner_agent.answer_question_with_kg(
            question=question,
            cache_key=cache_key,
            max_hop=max_hop,
            seed_top_k=seed_top_k,
            max_context_items=max_context_items,
        )

        print("\n" + "=" * 80)
        print(f"[Q{idx}] {question}")
        print(qa_result.get("formatted_answer", ""))


def main():
    parser = argparse.ArgumentParser(description="QA retrieval entry for a built knowledge graph")
    parser.add_argument(
        "--kg-path",
        default="./data/short/processed/RAKG_graph_re/1.json",
        help="Path to KG JSON file (must contain 'entities' and 'relations').",
    )
    parser.add_argument("--max-hop", type=int, default=1, help="Neighbor expansion hop (1 or 2).")
    parser.add_argument("--seed-top-k", type=int, default=5, help="Top-k seed entities matched from question.")
    parser.add_argument("--max-context-items", type=int, default=30, help="Max retrieved evidence items per question.")
    args = parser.parse_args()

    run_qa(
        kg_path=args.kg_path,
        questions=PRESET_QUESTIONS,
        max_hop=args.max_hop,
        seed_top_k=args.seed_top_k,
        max_context_items=args.max_context_items,
    )


if __name__ == "__main__":
    main()
