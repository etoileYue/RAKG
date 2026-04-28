"""Generate extra QA pairs for MultiFieldQA-ZH contexts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_provider import LLMProvider
from src.utils import parse_json_like_response, run_with_rate_limit_retry


DEFAULT_INPUT_PATH = "data/multifieldqa_zh/test.jsonl"
DEFAULT_OUTPUT_PATH = "data/multifieldqa_zh/expanded_qa.jsonl"
DEFAULT_SUMMARY_PATH = "data/multifieldqa_zh/qa_generation_summary.json"
DEFAULT_QA_COUNT = 9

GENERATION_PROMPT = """
你是一个中文问答对生成助手，请基于“原文”生成用于RAG测试的高质量问答对。

要求：
1. 所有问题和答案必须严格来源于原文，禁止引入原文之外的信息。
2. 问题之间不能重复或仅改写措辞，应覆盖不同信息点。
3. 问题应能够被原文明确回答。
4. 答案需准确、简洁。
5. evidence 必须是能够直接支持答案的原文片段。
6. 如果原文不足以生成指定数量，只输出高质量结果。
7. 仅输出合法 JSON，不要输出其他内容。

需要生成的问答对数量：
{qa_count}

已有问题（避免重复）：
{original_question}

原文：
{context}

输出格式：
{{
  "qa_pairs": [
    {{
      "question": "",
      "answer": "",
      "evidence": ""
    }}
  ]
}}
""".strip()


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def coerce_answers(raw_answers) -> List[str]:
    if raw_answers is None:
        return []
    if isinstance(raw_answers, list):
        return [str(item).strip() for item in raw_answers if str(item).strip()]
    text = str(raw_answers).strip()
    return [text] if text else []


def normalize_text(value) -> str:
    return "".join(str(value or "").strip().split())


def load_input_samples(path: Path) -> List[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for sample_index, line in enumerate(handle):
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)
            sample_id = str(raw.get("_id") or raw.get("sample_id") or sample_index)
            samples.append(
                {
                    "source_sample_id": sample_id,
                    "source_sample_index": sample_index,
                    "question": str(raw.get("input") or raw.get("question") or ""),
                    "answers": coerce_answers(raw.get("answers") or raw.get("answer")),
                    "context": str(raw.get("context") or ""),
                    "length": raw.get("length"),
                    "dataset": raw.get("dataset", "multifieldqa_zh"),
                    "language": raw.get("language", "zh"),
                    "all_classes": raw.get("all_classes"),
                }
            )
    return samples


def select_samples(samples: List[dict], start: int, end: Optional[int], limit: Optional[int]) -> List[dict]:
    if start < 0:
        raise ValueError("--start must be >= 0")
    if end is not None and end < start:
        raise ValueError("--end must be >= --start")
    if limit is not None and limit < 0:
        raise ValueError("--limit must be >= 0")

    selected = samples[start:end]
    if limit is not None:
        selected = selected[:limit]
    return selected


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_original_qa_pair(sample: dict) -> dict:
    return {
        "qa_id": "0",
        "question": sample["question"],
        "answers": sample["answers"],
        "qa_source": "original",
        "evidence": "",
    }


def build_group_record(sample: dict, qa_pairs: Iterable[dict]) -> dict:
    return {
        "source_sample_index": sample["source_sample_index"],
        "length": sample.get("length"),
        "dataset": sample.get("dataset"),
        "language": sample.get("language"),
        "all_classes": sample.get("all_classes"),
        "qa_pairs": assign_qa_ids(qa_pairs),
    }


def qa_pair_from_record(record: dict) -> dict:
    return {
        "qa_id": str(record.get("qa_id", "")).strip(),
        "question": str(record.get("question") or record.get("input") or "").strip(),
        "answers": coerce_answers(record.get("answers") or record.get("answer")),
        "qa_source": str(record.get("qa_source") or ("original" if "qa_id" not in record else "generated")),
        "evidence": str(record.get("evidence", "")).strip(),
    }


def assign_qa_ids(qa_pairs: Iterable[dict]) -> List[dict]:
    normalized_pairs = []
    for pair in qa_pairs:
        normalized_pairs.append(
            {
                "qa_id": str(pair.get("qa_id", "")).strip(),
                "question": str(pair.get("question", "")).strip(),
                "answers": coerce_answers(pair.get("answers") or pair.get("answer")),
                "qa_source": str(pair.get("qa_source") or "generated"),
                "evidence": str(pair.get("evidence", "")).strip(),
            }
        )

    def sort_key(item: dict) -> Tuple[int, str]:
        return (
            0 if item.get("qa_source") == "original" else 1,
            str(item.get("qa_id", "")),
        )

    ordered = sorted(normalized_pairs, key=sort_key)
    for index, pair in enumerate(ordered):
        pair["qa_id"] = str(index)
    return ordered


def ensure_original_qa_pair(sample: dict, group: dict) -> dict:
    qa_pairs = list(group.get("qa_pairs") or [])
    original = build_original_qa_pair(sample)
    existing_original_index = next(
        (idx for idx, pair in enumerate(qa_pairs) if pair.get("qa_source") == "original"),
        None,
    )
    if existing_original_index is None:
        qa_pairs.insert(0, original)
    else:
        qa_pairs[existing_original_index] = original

    updated = build_group_record(sample, qa_pairs)
    return updated


def build_generation_prompt(sample: dict, qa_count: int) -> str:
    return GENERATION_PROMPT.format(
        qa_count=qa_count,
        original_question=sample.get("question", ""),
        context=sample.get("context", ""),
    )


def parse_qa_generation_response(response) -> List[dict]:
    parsed = parse_json_like_response(response)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response is not a JSON object.")
    raw_pairs = parsed.get("qa_pairs")
    if not isinstance(raw_pairs, list):
        raise ValueError("LLM response must contain a qa_pairs list.")

    pairs = []
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, dict):
            continue
        answers = coerce_answers(raw_pair.get("answers") or raw_pair.get("answer"))
        answer = answers[0] if answers else ""
        pairs.append(
            {
                "question": str(raw_pair.get("question", "")).strip(),
                "answer": answer.strip(),
                "evidence": str(raw_pair.get("evidence", "")).strip(),
            }
        )
    return pairs


def validate_generated_pair(pair: dict, context: str) -> Optional[dict]:
    question = str(pair.get("question", "")).strip()
    answer = str(pair.get("answer", "")).strip()
    evidence = str(pair.get("evidence", "")).strip()
    if not question or not answer:
        return None
    return {"question": question, "answer": answer, "evidence": evidence}


def build_generated_qa_pair(pair: dict) -> dict:
    return {
        "qa_id": "",
        "question": pair["question"],
        "answers": [pair["answer"]],
        "qa_source": "generated",
        "evidence": pair["evidence"],
    }


def qa_duplicate_keys(qa_pairs: Iterable[dict]) -> set[Tuple[str, str]]:
    keys = set()
    for pair in qa_pairs:
        question_key = normalize_text(pair.get("question", ""))
        for answer in coerce_answers(pair.get("answers") or pair.get("answer")):
            answer_key = normalize_text(answer)
            if question_key and answer_key:
                keys.add((question_key, answer_key))
    return keys


def generate_qa_pairs_for_sample(
    model,
    sample: dict,
    qa_count: int,
    existing_pairs: Iterable[dict],
) -> tuple[List[dict], dict]:
    prompt = build_generation_prompt(sample, qa_count)
    response = model.invoke(prompt)
    raw_pairs = parse_qa_generation_response(response)

    qa_pairs = []
    invalid_count = 0
    unmatched_answer_count = 0
    unmatched_evidence_count = 0
    seen_pairs = qa_duplicate_keys(existing_pairs)
    for pair in raw_pairs:
        valid_pair = validate_generated_pair(pair, sample["context"])
        if valid_pair is None:
            invalid_count += 1
            continue
        duplicate_key = (normalize_text(valid_pair["question"]), normalize_text(valid_pair["answer"]))
        if duplicate_key in seen_pairs:
            invalid_count += 1
            continue
        seen_pairs.add(duplicate_key)
        if valid_pair["answer"] not in sample["context"]:
            unmatched_answer_count += 1
        if valid_pair["evidence"] and valid_pair["evidence"] not in sample["context"]:
            unmatched_evidence_count += 1
        qa_pairs.append(build_generated_qa_pair(valid_pair))
        if len(qa_pairs) >= qa_count:
            break

    stats = {
        "source_sample_index": sample["source_sample_index"],
        "requested_count": qa_count,
        "raw_count": len(raw_pairs),
        "valid_count": len(qa_pairs),
        "invalid_count": invalid_count,
        "unmatched_answer_count": unmatched_answer_count,
        "unmatched_evidence_count": unmatched_evidence_count,
        "shortfall": max(0, qa_count - len(qa_pairs)),
    }
    return qa_pairs, stats


def sort_records(records: Iterable[dict]) -> List[dict]:
    return sorted(
        records,
        key=lambda record: (
            int(record.get("source_sample_index", record.get("sample_index", 10**12))),
        ),
    )


def normalize_existing_groups(records: Iterable[dict], samples: Iterable[dict]) -> Dict[int, dict]:
    sample_by_index = {int(sample["source_sample_index"]): sample for sample in samples}
    grouped: Dict[int, dict] = {}

    for fallback_index, record in enumerate(records):
        source_sample_index = int(record.get("source_sample_index", record.get("sample_index", fallback_index)))
        sample = sample_by_index.get(
            source_sample_index,
            {
                "source_sample_index": source_sample_index,
                "length": record.get("length"),
                "dataset": record.get("dataset", "multifieldqa_zh"),
                "language": record.get("language", "zh"),
                "all_classes": record.get("all_classes"),
                "question": "",
                "answers": [],
            },
        )
        current = grouped.get(source_sample_index)
        if current is None:
            current = build_group_record(sample, [])
            grouped[source_sample_index] = current

        if isinstance(record.get("qa_pairs"), list):
            current["qa_pairs"].extend(record.get("qa_pairs") or [])
        else:
            current["qa_pairs"].append(qa_pair_from_record(record))

    for source_sample_index, group in list(grouped.items()):
        sample = sample_by_index.get(
            source_sample_index,
            {
                "source_sample_index": source_sample_index,
                "length": group.get("length"),
                "dataset": group.get("dataset", "multifieldqa_zh"),
                "language": group.get("language", "zh"),
                "all_classes": group.get("all_classes"),
                "question": "",
                "answers": [],
            },
        )
        grouped[source_sample_index] = build_group_record(sample, group.get("qa_pairs") or [])
    return grouped


def count_generated_pairs(group: Optional[dict]) -> int:
    if not group:
        return 0
    return sum(1 for pair in group.get("qa_pairs", []) if pair.get("qa_source") == "generated")


def count_total_qa_pairs(groups: Iterable[dict]) -> int:
    return sum(len(group.get("qa_pairs") or []) for group in groups)


def rate_limit_retry_kwargs(args, label: str) -> dict:
    return {
        "label": label,
        "max_retries": args.rate_limit_max_retries,
        "initial_wait_seconds": args.rate_limit_initial_wait,
        "max_wait_seconds": args.rate_limit_max_wait,
    }


def run_generation(args) -> dict:
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path)

    samples = load_input_samples(input_path)
    selected_samples = select_samples(samples, start=args.start, end=args.end, limit=args.limit)
    existing_records = read_jsonl(output_path)
    grouped_records = normalize_existing_groups(existing_records, samples)
    selected_indices = {int(sample["source_sample_index"]) for sample in selected_samples}
    if args.force:
        for source_sample_index in selected_indices:
            grouped_records.pop(source_sample_index, None)

    model = None
    generated_count = 0
    per_sample_stats = []

    print(
        "MultiFieldQA-ZH QA generation started: "
        f"input={input_path} output={output_path} summary={summary_path} "
        f"total_samples={len(samples)} selected={len(selected_samples)} "
        f"qa_count={args.qa_count} force={bool(args.force)}"
    )

    for offset, sample in enumerate(selected_samples, start=1):
        source_sample_index = int(sample["source_sample_index"])
        group = ensure_original_qa_pair(
            sample,
            grouped_records.get(source_sample_index) or build_group_record(sample, []),
        )
        grouped_records[source_sample_index] = group
        existing_generated_count = count_generated_pairs(group)
        missing_count = max(0, args.qa_count - existing_generated_count)
        print(
            f"[{offset}/{len(selected_samples)}] source_sample_index={source_sample_index} "
            f"existing_generated={existing_generated_count} missing={missing_count}"
        )

        if not args.force and missing_count == 0:
            print(
                f"[{offset}/{len(selected_samples)}] skip "
                f"source_sample_index={source_sample_index} generated={existing_generated_count}"
            )
            per_sample_stats.append(
                {
                    "source_sample_index": sample["source_sample_index"],
                    "requested_count": args.qa_count,
                    "raw_count": 0,
                    "valid_count": 0,
                    "invalid_count": 0,
                    "unmatched_answer_count": 0,
                    "unmatched_evidence_count": 0,
                    "shortfall": 0,
                    "skipped_existing": True,
                }
            )
            continue

        existing_pairs = group.get("qa_pairs") or []
        if model is None:
            model = LLMProvider().get_llm()
        qa_pairs, stats = run_with_rate_limit_retry(
            lambda sample=sample, existing_pairs=existing_pairs, missing_count=missing_count: generate_qa_pairs_for_sample(
                model,
                sample,
                missing_count,
                existing_pairs,
            ),
            **rate_limit_retry_kwargs(args, f"generate QA for sample index {source_sample_index}"),
        )
        merged_pairs = list(existing_pairs) + qa_pairs
        grouped_records[source_sample_index] = build_group_record(sample, merged_pairs)
        generated_count += len(qa_pairs)
        per_sample_stats.append(stats)
        print(
            f"[{offset}/{len(selected_samples)}] generate "
            f"source_sample_index={source_sample_index} raw={stats['raw_count']} "
            f"valid={stats['valid_count']} invalid={stats['invalid_count']} "
            f"unmatched_answer={stats['unmatched_answer_count']} "
            f"unmatched_evidence={stats['unmatched_evidence_count']} "
            f"shortfall={stats['shortfall']}"
        )

    merged_records = sort_records(grouped_records.values())
    write_jsonl(output_path, merged_records)
    total_qa_count = count_total_qa_pairs(merged_records)

    summary = {
        "phase": "generate_multifieldqa_zh_qa",
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "selected_count": len(selected_samples),
        "total_sample_count": len(samples),
        "qa_count": args.qa_count,
        "force": bool(args.force),
        "existing_count": len(existing_records),
        "generated_count": generated_count,
        "original_count": sum(
            1
            for record in merged_records
            for pair in record.get("qa_pairs", [])
            if pair.get("qa_source") == "original"
        ),
        "total_count": total_qa_count,
        "group_count": len(merged_records),
        "shortfall_count": sum(int(item.get("shortfall", 0)) for item in per_sample_stats),
        "invalid_count": sum(int(item.get("invalid_count", 0)) for item in per_sample_stats),
        "unmatched_answer_count": sum(int(item.get("unmatched_answer_count", 0)) for item in per_sample_stats),
        "unmatched_evidence_count": sum(int(item.get("unmatched_evidence_count", 0)) for item in per_sample_stats),
        "range": {"start": args.start, "end": args.end, "limit": args.limit},
        "samples": per_sample_stats,
        "updated_at": utc_timestamp(),
    }
    write_json(summary_path, summary)
    print(
        "MultiFieldQA-ZH QA generation finished: "
        f"generated={generated_count} total_qa={total_qa_count} "
        f"output={output_path} summary={summary_path}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate extra QA pairs for MultiFieldQA-ZH.")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH, help="Path to original MultiFieldQA-ZH JSONL.")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Path to expanded QA JSONL.")
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH, help="Path to generation summary JSON.")
    parser.add_argument("--qa-count", type=int, default=DEFAULT_QA_COUNT, help="Generated QA count per source sample.")
    parser.add_argument("--start", type=int, default=0, help="Inclusive source sample start index.")
    parser.add_argument("--end", type=int, default=None, help="Exclusive source sample end index.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum source samples after slicing.")
    parser.add_argument("--force", action="store_true", help="Regenerate selected samples and rewrite output records.")
    parser.add_argument(
        "--rate-limit-max-retries",
        type=int,
        default=8,
        help="Maximum retries for rate-limit errors after the initial attempt.",
    )
    parser.add_argument(
        "--rate-limit-initial-wait",
        type=float,
        default=30.0,
        help="Initial wait in seconds before retrying rate-limit errors.",
    )
    parser.add_argument(
        "--rate-limit-max-wait",
        type=float,
        default=300.0,
        help="Maximum wait in seconds between rate-limit retries.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.qa_count < 0:
        parser.error("--qa-count must be >= 0")
    return run_generation(args)


def cli(argv: Optional[List[str]] = None) -> int:
    summary = main(argv)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
