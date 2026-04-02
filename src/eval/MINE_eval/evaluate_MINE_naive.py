import ast
import json
import os
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.llm_provider import LLMProvider


llm_provider = LLMProvider()
embedding_model = llm_provider.get_embedding_model()
client = llm_provider.get_llm()


def load_questions_answers_from_rakg_eval(rakg_eval_file: str) -> List[List[Dict]]:
    """复用 evaluate_MINE_RAKG.py 中定义的 all_questions_answers，避免重复维护。"""
    with open(rakg_eval_file, "r", encoding="utf-8") as f:
        source = f.read()

    module = ast.parse(source, filename=rakg_eval_file)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "all_questions_answers":
                        value = ast.literal_eval(stmt.value)
                        if not isinstance(value, list):
                            raise TypeError("all_questions_answers is not a list")
                        return value

    raise ValueError(f"Cannot find all_questions_answers in {rakg_eval_file}")


def load_naive_index(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_fields = ["sentences", "vectors", "sentence_to_id"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"index file missing required field: {field}, file={file_path}")
    return data


def retrieve_naive_items(query: str, index_data: Dict, top_k: int = 8) -> List[Dict]:
    sentences = index_data.get("sentences", [])
    vectors = index_data.get("vectors", [])
    sentence_to_id = index_data.get("sentence_to_id", {})

    if not sentences or not vectors:
        return []

    query_vector = embedding_model.embed_query(query)
    sentence_vectors = np.array(vectors, dtype=float)
    similarities = cosine_similarity([query_vector], sentence_vectors)[0]

    top_k = max(1, min(top_k, len(sentences)))
    top_indices = np.argsort(similarities)[::-1][:top_k]

    retrieved_items = []
    for rank, sent_idx in enumerate(top_indices, start=1):
        sentence = sentences[sent_idx]
        retrieved_items.append(
            {
                "rank": rank,
                "sentence": sentence,
                "sentence_id": sentence_to_id.get(sentence, ""),
                "score": float(similarities[sent_idx]),
            }
        )
    return retrieved_items


def build_context_text(retrieved_items: List[Dict]) -> str:
    """与 NaiveRAGAgent._build_context 对齐。"""
    lines = []
    for idx, item in enumerate(retrieved_items, start=1):
        lines.append(
            f"{idx}. [sentence_id={item.get('sentence_id', '')}] "
            f"{item.get('sentence', '')} (score={item.get('score', 0):.4f})"
        )
    return "\n".join(lines)


def gpt_evaluate_response(correct_answer: str, context: str) -> int:
    prompt = f"""
    Context:
    {context}

    Correct Answer:
    {correct_answer}

    Task:
    Determine whether the context contains the information stated in the correct answer. \\
    Return JSON only, with this schema: {{"result": 1}} if yes, or {{"result": 0}} if no.
    """
    system_instruction = (
        "You are an evaluator who derives the correct answer based on contextual information."
        "Only output a JSON object, and the 'result' field must be 0 or 1."
    )

    response = client.invoke(f"{system_instruction}\n\n{prompt}")
    raw = response.content if hasattr(response, "content") else str(response)
    raw_text = str(raw).strip()

    try:
        parsed = json.loads(raw_text)
    except Exception:
        parsed = None

    value = None
    if isinstance(parsed, dict):
        value = parsed.get("result")
    elif isinstance(parsed, (int, float, bool)):
        value = parsed
    elif isinstance(parsed, str):
        value = parsed.strip()
    else:
        value = raw_text

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if int(value) == 1 else 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes"}:
            return 1
        if lowered in {"0", "false", "no"}:
            return 0
        if "1" in lowered and "0" not in lowered:
            return 1
        if "0" in lowered and "1" not in lowered:
            return 0

    raise ValueError(f"Unable to parse evaluation result from response: {raw_text}")


def evaluate_accuracy(
    questions_answers: List[Dict],
    index_data: Dict,
    output_file: str,
    top_k: int = 5,
) -> Dict:
    correct = 0
    results = []

    for qa in questions_answers:
        correct_answer = qa.get("answer", "")
        try:
            retrieved_items = retrieve_naive_items(correct_answer, index_data=index_data, top_k=top_k)
            context_text = build_context_text(retrieved_items)

            evaluation = gpt_evaluate_response(correct_answer, context_text)
            results.append(
                {
                    "correct_answer": correct_answer,
                    "retrieved_context": context_text,
                    "retrieved_items": retrieved_items,
                    "evaluation": evaluation,
                }
            )
            correct += evaluation
        except Exception as e:
            print(f"An error occurred while processing QA: {qa}")
            print(f"Error message: {str(e)}")
            traceback.print_exc()
            results.append(
                {
                    "correct_answer": correct_answer,
                    "retrieved_context": None,
                    "retrieved_items": [],
                    "evaluation": 0,
                    "error": str(e),
                }
            )

    total = len(questions_answers)
    accuracy = (correct / total) if total else 0.0
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": f"{accuracy * 100:.2f}%",
    }
    results.append(summary)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_file}")

    return summary


def _sorted_index_files(index_dir: str) -> List[Path]:
    paths = [p for p in Path(index_dir).glob("*.json") if p.is_file()]

    def sort_key(p: Path) -> Tuple[int, str]:
        stem = p.stem
        try:
            return int(stem), stem
        except ValueError:
            return 10**9, stem

    return sorted(paths, key=sort_key)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    rakg_eval_file = str(base_dir / "evaluate_MINE_RAKG.py")

    index_dir = "data/short/naive/processed/naive_rag_index"
    output_dir = "data/short/naive/processed/result"
    top_k = 5

    all_questions_answers = load_questions_answers_from_rakg_eval(rakg_eval_file)
    index_files = _sorted_index_files(index_dir)

    if not index_files:
        raise FileNotFoundError(f"No index files found in: {index_dir}")

    run_summaries = []
    for idx, index_file in enumerate(index_files, start=1):
        if idx > len(all_questions_answers):
            print(f"[WARN] Skip {index_file} because questions_answers is missing for idx={idx}")
            continue

        print(f"Processing file: {index_file}")
        questions_answers = all_questions_answers[idx - 1]
        output_file = os.path.join(output_dir, f"{index_file.stem}_results.json")

        index_data = load_naive_index(str(index_file))
        summary = evaluate_accuracy(
            questions_answers=questions_answers,
            index_data=index_data,
            output_file=output_file,
            top_k=top_k,
        )
        run_summaries.append(
            {
                "index": index_file.stem,
                "source_file": str(index_file),
                "output_file": output_file,
                **summary,
            }
        )

    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(run_summaries, f, ensure_ascii=False, indent=2)
    print(f"Run summary saved to {summary_file}")


if __name__ == "__main__":
    main()
