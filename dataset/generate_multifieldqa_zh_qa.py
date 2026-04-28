"""Generate extra QA pairs for MultiFieldQA-ZH contexts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

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


def build_original_record(sample: dict) -> dict:
    return {
        "qa_id": f"{sample['source_sample_id']}#original",
        "source_sample_id": sample["source_sample_id"],
        "source_sample_index": sample["source_sample_index"],
        "question": sample["question"],
        "answers": sample["answers"],
        "qa_source": "original",
        "evidence": "",
        "length": sample.get("length"),
        "dataset": sample.get("dataset"),
        "language": sample.get("language"),
        "all_classes": sample.get("all_classes"),
    }


def add_original_qa_pairs(samples: Iterable[dict], records: Iterable[dict]) -> List[dict]:
    """Return records with one original QA record per source sample, without duplicates."""

    merged = {}
    for record in records:
        qa_id = str(record.get("qa_id", "")).strip()
        if qa_id:
            merged[qa_id] = dict(record)

    for sample in samples:
        original = build_original_record(sample)
        merged.setdefault(original["qa_id"], original)

    return sort_records(merged.values())


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
    if not question or not answer or not evidence:
        return None
    if evidence not in context:
        return None
    if answer not in context:
        return None
    return {"question": question, "answer": answer, "evidence": evidence}


def build_generated_record(sample: dict, pair: dict, sequence_number: int) -> dict:
    return {
        "qa_id": f"{sample['source_sample_id']}#gen{sequence_number:03d}",
        "source_sample_id": sample["source_sample_id"],
        "source_sample_index": sample["source_sample_index"],
        "question": pair["question"],
        "answers": [pair["answer"]],
        "qa_source": "generated",
        "evidence": pair["evidence"],
        "length": sample.get("length"),
        "dataset": sample.get("dataset"),
        "language": sample.get("language"),
        "all_classes": sample.get("all_classes"),
    }


def generate_records_for_sample(model, sample: dict, qa_count: int) -> tuple[List[dict], dict]:
    prompt = build_generation_prompt(sample, qa_count)
    response = model.invoke(prompt)
    raw_pairs = parse_qa_generation_response(response)

    records = []
    invalid_count = 0
    seen_questions = {sample.get("question", "").strip()}
    seen_answers = set()
    for pair in raw_pairs:
        valid_pair = validate_generated_pair(pair, sample["context"])
        if valid_pair is None:
            invalid_count += 1
            continue
        duplicate_key = (valid_pair["question"], valid_pair["answer"])
        if valid_pair["question"] in seen_questions or duplicate_key in seen_answers:
            invalid_count += 1
            continue
        seen_questions.add(valid_pair["question"])
        seen_answers.add(duplicate_key)
        records.append(build_generated_record(sample, valid_pair, len(records) + 1))
        if len(records) >= qa_count:
            break

    stats = {
        "source_sample_id": sample["source_sample_id"],
        "source_sample_index": sample["source_sample_index"],
        "requested_count": qa_count,
        "raw_count": len(raw_pairs),
        "valid_count": len(records),
        "invalid_count": invalid_count,
        "shortfall": max(0, qa_count - len(records)),
    }
    return records, stats


def sort_records(records: Iterable[dict]) -> List[dict]:
    return sorted(
        records,
        key=lambda record: (
            int(record.get("source_sample_index", record.get("sample_index", 10**12))),
            str(record.get("qa_id", "")),
        ),
    )


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
    existing_records = [] if args.force else read_jsonl(output_path)
    record_by_id = {
        str(record.get("qa_id")): record
        for record in existing_records
        if str(record.get("qa_id", "")).strip()
    }

    model = LLMProvider().get_llm() if selected_samples else None
    generated_records = []
    per_sample_stats = []

    for sample in selected_samples:
        missing_ids = [
            f"{sample['source_sample_id']}#gen{idx:03d}"
            for idx in range(1, args.qa_count + 1)
            if f"{sample['source_sample_id']}#gen{idx:03d}" not in record_by_id
        ]
        if not args.force and not missing_ids:
            per_sample_stats.append(
                {
                    "source_sample_id": sample["source_sample_id"],
                    "source_sample_index": sample["source_sample_index"],
                    "requested_count": args.qa_count,
                    "raw_count": 0,
                    "valid_count": args.qa_count,
                    "invalid_count": 0,
                    "shortfall": 0,
                    "skipped_existing": True,
                }
            )
            continue

        records, stats = run_with_rate_limit_retry(
            lambda sample=sample: generate_records_for_sample(model, sample, args.qa_count),
            **rate_limit_retry_kwargs(args, f"generate QA for {sample['source_sample_id']}"),
        )
        for record in records:
            record_by_id[record["qa_id"]] = record
        generated_records.extend(records)
        per_sample_stats.append(stats)

    merged_records = add_original_qa_pairs(selected_samples, record_by_id.values())
    write_jsonl(output_path, merged_records)

    summary = {
        "phase": "generate_multifieldqa_zh_qa",
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "selected_count": len(selected_samples),
        "qa_count": args.qa_count,
        "force": bool(args.force),
        "existing_count": len(existing_records),
        "generated_count": len(generated_records),
        "original_count": sum(1 for record in merged_records if record.get("qa_source") == "original"),
        "total_count": len(merged_records),
        "shortfall_count": sum(int(item.get("shortfall", 0)) for item in per_sample_stats),
        "range": {"start": args.start, "end": args.end, "limit": args.limit},
        "samples": per_sample_stats,
        "updated_at": utc_timestamp(),
    }
    write_json(summary_path, summary)
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
