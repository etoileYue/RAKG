#!/usr/bin/env python3
"""Convert Multi-Doc-QA-Chinese raw samples into retrieval training JSONL."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Sequence


DEFAULT_DATASET_NAME = "yuyijiong/Multi-Doc-QA-Chinese"
DEFAULT_DATA_DIR = "raw"
DEFAULT_OUTPUT = "data/processed/multidocqa_chinese/raw_retrieval_train.jsonl"
DEFAULT_MAX_NEGATIVES = 20

FIELD_ALIASES = {
    "id": [
        "id",
        "sample_id",
        "question_id",
        "qid",
        "uid",
    ],
    "query": [
        "qa",
        "question",
        "query",
        "prompt",
        "instruction",
        "input",
    ],
    "positive": [
        "positive_doc",
        "positive_document",
        "positive",
        "relevant_doc",
        "relevant_document",
        "relevant_passage",
        "supporting_doc",
        "supporting_document",
        "gold_doc",
        "gold_document",
        "document",
        "doc",
    ],
    "negatives": [
        "negative_doc",
        "negative_docs",
        "negative_documents",
        "negative_passages",
        "negatives",
        "irrelevant_docs",
        "irrelevant_documents",
        "distractor_docs",
        "distractor_documents",
        "distractors",
        "hard_negatives",
    ],
    "answer": [
        "answer",
        "answers",
        "response",
        "output",
        "target",
    ],
}

FIELD_HINTS = {
    "query": ("question", "query", "prompt", "instruction"),
    "positive": ("positive", "relevant", "support", "gold"),
    "negatives": ("negative", "irrelevant", "distractor"),
    "answer": ("answer", "response", "output"),
}


@dataclass
class ConversionStats:
    total_rows: int = 0
    written_rows: int = 0
    filtered_empty_positive: int = 0
    filtered_empty_negative: int = 0
    filtered_invalid_query: int = 0
    duplicate_negative_rows: int = 0
    positive_removed_from_negatives: int = 0
    truncated_negative_rows: int = 0
    multi_positive_rows: int = 0
    total_negative_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        average_negatives = (
            round(self.total_negative_count / self.written_rows, 2)
            if self.written_rows
            else 0.0
        )
        return {
            "total_rows": self.total_rows,
            "written_rows": self.written_rows,
            "filtered_empty_positive": self.filtered_empty_positive,
            "filtered_empty_negative": self.filtered_empty_negative,
            "filtered_invalid_query": self.filtered_invalid_query,
            "duplicate_negative_rows": self.duplicate_negative_rows,
            "positive_removed_from_negatives": self.positive_removed_from_negatives,
            "truncated_negative_rows": self.truncated_negative_rows,
            "multi_positive_rows": self.multi_positive_rows,
            "average_negative_count": average_negatives,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert yuyijiong/Multi-Doc-QA-Chinese raw samples into retrieval "
            "training JSONL with query / positives / negatives fields."
        )
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-negatives", type=int, default=DEFAULT_MAX_NEGATIVES)
    parser.add_argument("--query-field", default=None)
    parser.add_argument("--positive-field", default=None)
    parser.add_argument("--negative-field", default=None)
    parser.add_argument("--answer-field", default=None)
    parser.add_argument("--id-field", default=None)
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load, map, and validate records without writing output files.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def extract_qa_text(value: Any, key: str) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return clean_text(value.get(key))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            if isinstance(item, dict):
                text = clean_text(item.get(key))
                if text:
                    return text
            else:
                text = clean_text(item)
                if text:
                    return text
        return ""
    return clean_text(value)


def extract_query_text(value: Any) -> str:
    return extract_qa_text(value, "question")


def extract_answer_text(value: Any) -> str:
    return extract_qa_text(value, "answer")


def normalize_doc_item(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("content", "text", "title"):
            text = clean_text(value.get(key))
            if text:
                return text
        return ""
    return clean_text(value)


def normalize_doc_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        text = normalize_doc_item(value)
        return [text] if text else []
    if isinstance(value, str):
        text = clean_text(value)
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        docs = []
        for item in value:
            text = normalize_doc_item(item)
            if text:
                docs.append(text)
        return docs
    text = normalize_doc_item(value)
    return [text] if text else []


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def candidate_matches(column: str, field_name: str) -> int:
    lowered = column.lower()
    score = 0
    for index, alias in enumerate(FIELD_ALIASES[field_name]):
        if lowered == alias:
            score = max(score, 1000 - index)
    for hint in FIELD_HINTS.get(field_name, ()):
        if hint in lowered:
            score = max(score, 100 - len(lowered))
    return score


def pick_field(
    columns: Sequence[str],
    field_name: str,
    provided: str | None = None,
    used_fields: set[str] | None = None,
) -> str | None:
    if provided:
        if provided not in columns:
            raise ValueError(
                f"Provided {field_name!r} field {provided!r} is not in dataset columns: {list(columns)}"
            )
        return provided

    if used_fields is None:
        used_fields = set()

    scored = []
    for column in columns:
        if column in used_fields:
            continue
        score = candidate_matches(column, field_name)
        if score > 0:
            scored.append((score, column))

    if not scored:
        return None

    scored.sort(reverse=True)
    return scored[0][1]


def resolve_field_mapping(
    columns: Sequence[str],
    overrides: dict[str, str | None] | None = None,
) -> dict[str, str | None]:
    overrides = overrides or {}
    used_fields: set[str] = set()

    mapping: dict[str, str | None] = {}
    for field_name in ("id", "query", "positive", "negatives"):
        provided = overrides.get(field_name)
        selected = pick_field(columns, field_name, provided=provided, used_fields=used_fields)
        mapping[field_name] = selected
        if selected:
            used_fields.add(selected)

    mapping["answer"] = pick_field(
        columns,
        "answer",
        provided=overrides.get("answer"),
        used_fields=None,
    )
    if not mapping["answer"] and mapping["query"]:
        mapping["answer"] = mapping["query"]

    if not mapping["query"] or not mapping["positive"] or not mapping["negatives"]:
        raise ValueError(
            "Could not resolve required fields. "
            f"Resolved mapping={mapping}, columns={list(columns)}. "
            "Use --query-field / --positive-field / --negative-field to override."
        )

    return mapping


def build_sample_id(record: dict[str, Any], mapping: dict[str, str | None], split: str, row_index: int) -> str:
    id_field = mapping.get("id")
    if id_field:
        candidate = clean_text(record.get(id_field))
        if candidate:
            return candidate
    return f"{split}-{row_index:06d}"


def convert_record(
    record: dict[str, Any],
    mapping: dict[str, str | None],
    split: str,
    row_index: int,
    max_negatives: int,
    dataset_name: str,
    data_dir: str,
    stats: ConversionStats,
) -> dict[str, Any] | None:
    stats.total_rows += 1

    query = extract_query_text(record.get(mapping["query"])) if mapping["query"] else ""
    if not query:
        stats.filtered_invalid_query += 1
        return None

    positive_docs = normalize_doc_list(record.get(mapping["positive"])) if mapping["positive"] else []
    if len(positive_docs) > 1:
        stats.multi_positive_rows += 1
    if not positive_docs:
        stats.filtered_empty_positive += 1
        return None

    positive_doc = positive_docs[0]

    negative_docs = (
        normalize_doc_list(record.get(mapping["negatives"])) if mapping["negatives"] else []
    )
    deduped_negatives = dedupe_preserve_order(negative_docs)
    if len(deduped_negatives) != len(negative_docs):
        stats.duplicate_negative_rows += 1

    filtered_negatives = [doc for doc in deduped_negatives if doc != positive_doc]
    if len(filtered_negatives) != len(deduped_negatives):
        stats.positive_removed_from_negatives += 1

    if len(filtered_negatives) > max_negatives:
        stats.truncated_negative_rows += 1
        filtered_negatives = filtered_negatives[:max_negatives]

    if not filtered_negatives:
        stats.filtered_empty_negative += 1
        return None

    answer = extract_answer_text(record.get(mapping["answer"])) if mapping.get("answer") else ""

    stats.written_rows += 1
    stats.total_negative_count += len(filtered_negatives)

    return {
        "id": build_sample_id(record, mapping, split, row_index),
        "query": query,
        "positives": [positive_doc],
        "negatives": filtered_negatives,
        "answer": answer,
        "meta": {
            "source_dataset": dataset_name,
            "subset": data_dir,
            "split": split,
            "negative_count": len(filtered_negatives),
        },
    }


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent_dir(path)
    count = 0
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def preview_samples(rows: Sequence[dict[str, Any]], preview_count: int) -> None:
    if preview_count <= 0:
        return
    preview_rows = list(rows[:preview_count])
    if not preview_rows:
        print("No converted rows available for preview.")
        return
    print(f"Previewing {len(preview_rows)} converted rows:")
    for row in preview_rows:
        print(
            json.dumps(
                {
                    "id": row["id"],
                    "query": row["query"],
                    "positives": row["positives"],
                    "negative_count": len(row["negatives"]),
                },
                ensure_ascii=False,
            )
        )


def resolve_output_path(args: argparse.Namespace, split: str) -> Path:
    if args.split != "all":
        return Path(args.output)

    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        output_path = Path(args.output)
        if output_path.suffix == ".jsonl":
            base_dir = output_path.parent
        else:
            base_dir = output_path

    return base_dir / f"raw_retrieval_{split}.jsonl"


def load_hf_dataset(dataset_name: str, data_dir: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The 'datasets' package is required for this script. "
            "Install dependencies from requirements.txt first."
        ) from exc

    return load_dataset(dataset_name, data_dir=data_dir)


def convert_split(
    split_name: str,
    split_dataset: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], ConversionStats, dict[str, str | None], int]:
    columns = list(split_dataset.column_names)
    print(f"[{split_name}] columns: {columns}")

    mapping = resolve_field_mapping(
        columns,
        overrides={
            "id": args.id_field,
            "query": args.query_field,
            "positive": args.positive_field,
            "negatives": args.negative_field,
            "answer": args.answer_field,
        },
    )
    print(f"[{split_name}] resolved field mapping: {mapping}")

    stats = ConversionStats()
    preview_rows: list[dict[str, Any]] = []
    written_count = 0
    output_path = None if args.dry_run else resolve_output_path(args, split_name)
    if output_path is not None:
        ensure_parent_dir(output_path)
        output_path.write_text("", encoding="utf-8")

    for row_index, record in enumerate(split_dataset, start=1):
        sample = convert_record(
            record=record,
            mapping=mapping,
            split=split_name,
            row_index=row_index,
            max_negatives=args.max_negatives,
            dataset_name=args.dataset_name,
            data_dir=args.data_dir,
            stats=stats,
        )
        if sample is not None:
            if len(preview_rows) < args.preview_count:
                preview_rows.append(sample)
            if output_path is not None:
                append_jsonl(output_path, sample)
                written_count += 1

    return preview_rows, stats, mapping, written_count


def main() -> None:
    args = parse_args()
    if args.max_negatives < 1:
        raise SystemExit("--max-negatives must be >= 1")

    dataset = load_hf_dataset(args.dataset_name, args.data_dir)
    available_splits = list(dataset.keys())
    print(f"Available splits: {available_splits}")

    if args.split == "all":
        target_splits = available_splits
    else:
        if args.split not in dataset:
            raise SystemExit(
                f"Split {args.split!r} not found. Available splits: {available_splits}"
            )
        target_splits = [args.split]

    for split_name in target_splits:
        preview_rows, stats, _mapping, written_count = convert_split(split_name, dataset[split_name], args)
        preview_samples(preview_rows, args.preview_count)

        if args.dry_run:
            print(f"[{split_name}] dry-run complete. No files written.")
        else:
            output_path = resolve_output_path(args, split_name)
            print(f"[{split_name}] wrote {written_count} rows to {output_path}")

        print(
            f"[{split_name}] stats: "
            f"{json.dumps(stats.to_dict(), ensure_ascii=False, sort_keys=True)}"
        )


if __name__ == "__main__":
    main()
