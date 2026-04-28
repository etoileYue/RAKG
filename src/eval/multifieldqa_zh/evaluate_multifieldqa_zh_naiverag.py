"""Stage-wise NaiveRAG baseline evaluation for LongBench MultiFieldQA-ZH."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import List, Optional

from src.eval.multifieldqa_zh.evaluate_multifieldqa_zh import (
    ANSWER_JUDGE_PROMPT,
    DEFAULT_QA_DATASET_PATH,
    RETRIEVAL_JUDGE_PROMPT,
    build_base_record,
    build_error_record,
    ensure_dir,
    index_records_by_qa_or_sample_id,
    index_records_by_sample_id,
    invoke_binary_judge,
    load_dataset,
    load_jsonl,
    load_qa_dataset,
    max_qa_f1_zh_score,
    needs_score_work,
    resolve_qa_dataset_path,
    select_samples,
    sort_records,
    utc_timestamp,
    write_json,
    write_jsonl,
)
from src.llm_provider import LLMProvider
from src.naiveRAG import NaiveRAGAgent


DEFAULT_DATASET_PATH = "data/multifieldqa_zh/test.jsonl"
DEFAULT_OUTPUT_ROOT = "data/eval/naiveRAG"
DEFAULT_TOP_K = 5


def resolve_output_paths(output_root: Path) -> dict:
    summary_dir = output_root / "summary"
    result_dir = output_root / "result"
    return {
        "output_root": output_root,
        "index_dir": output_root / "index",
        "summary_dir": summary_dir,
        "result_dir": result_dir,
        "build_manifest_path": summary_dir / "build_manifest.jsonl",
        "build_summary_path": summary_dir / "build_summary.json",
        "predictions_path": result_dir / "predictions.jsonl",
        "answer_summary_path": summary_dir / "answer_summary.json",
        "scored_results_path": result_dir / "scored_results.jsonl",
        "score_summary_path": summary_dir / "score_summary.json",
    }


def build_index_path(output_paths: dict, sample: dict) -> Path:
    return output_paths["index_dir"] / f"{sample['sample_index']}.json"


def path_has_content(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def is_build_complete(record: Optional[dict]) -> bool:
    return bool(
        record
        and record.get("status") == "success"
        and str(record.get("index_path", "")).strip()
        and Path(record["index_path"]).exists()
    )


def is_answer_complete(record: Optional[dict]) -> bool:
    return bool(record and record.get("status") == "success")


def backfill_manifest_from_existing_indexes(
    samples: List[dict],
    existing_manifest: dict,
    output_paths: dict,
) -> int:
    backfilled_count = 0
    for sample in samples:
        current_record = existing_manifest.get(sample["sample_id"])
        if is_build_complete(current_record):
            continue

        index_path = build_index_path(output_paths, sample)
        if not path_has_content(index_path):
            continue

        record = build_base_record(sample)
        record.update(
            {
                "phase": "build",
                "status": "success",
                "topic": sample["sample_id"],
                "index_path": str(index_path.resolve()),
                "updated_at": utc_timestamp(),
                "recovered_from_existing_index": True,
            }
        )
        existing_manifest[sample["sample_id"]] = record
        backfilled_count += 1
    return backfilled_count


def build_stage(args) -> dict:
    dataset_path = Path(args.dataset_path)
    output_paths = resolve_output_paths(Path(args.output_root))
    ensure_dir(output_paths["index_dir"])

    samples = load_dataset(dataset_path)
    selected_samples = select_samples(samples, start=args.start, end=args.end, limit=args.limit)
    existing_manifest = index_records_by_sample_id(load_jsonl(output_paths["build_manifest_path"]))
    backfilled_manifest_count = backfill_manifest_from_existing_indexes(
        selected_samples,
        existing_manifest,
        output_paths,
    )

    run_stats = {"selected_count": len(selected_samples), "skipped_count": 0, "success_count": 0, "error_count": 0}
    pending_samples = []
    for sample in selected_samples:
        current_record = existing_manifest.get(sample["sample_id"])
        if not args.force and is_build_complete(current_record):
            run_stats["skipped_count"] += 1
        else:
            pending_samples.append(sample)

    agent = NaiveRAGAgent() if pending_samples else None
    for sample in pending_samples:
        index_path = build_index_path(output_paths, sample)
        try:
            result = agent.process(
                topic_data={"topic": sample["sample_id"], "content": sample["context"]},
                idx=sample["sample_index"],
                total_topics=len(samples),
                rag_output_dir=str(output_paths["index_dir"]),
            )

            record = build_base_record(sample)
            record.update(
                {
                    "phase": "build",
                    "status": "success",
                    "topic": sample["sample_id"],
                    "index_path": str(Path(result.get("output_path") or index_path).resolve()),
                    "updated_at": utc_timestamp(),
                }
            )
            existing_manifest[sample["sample_id"]] = record
            run_stats["success_count"] += 1
        except Exception as exc:
            record = build_error_record(sample, phase="build", error=exc)
            record["index_path"] = str(index_path.resolve())
            existing_manifest[sample["sample_id"]] = record
            run_stats["error_count"] += 1

    manifest_records = sort_records(existing_manifest.values())
    write_jsonl(output_paths["build_manifest_path"], manifest_records)

    summary = {
        "phase": "build",
        "dataset_path": str(dataset_path.resolve()),
        "output_root": str(output_paths["output_root"].resolve()),
        "index_dir": str(output_paths["index_dir"].resolve()),
        "manifest_path": str(output_paths["build_manifest_path"].resolve()),
        "backfilled_manifest_count": backfilled_manifest_count,
        "range": {"start": args.start, "end": args.end, "limit": args.limit},
        "force": bool(args.force),
        **run_stats,
        "completed_count": sum(1 for record in manifest_records if record.get("status") == "success"),
        "failed_count_total": sum(1 for record in manifest_records if record.get("status") == "error"),
        "updated_at": utc_timestamp(),
    }
    write_json(output_paths["build_summary_path"], summary)
    return summary


def answer_stage(args) -> dict:
    dataset_path = Path(args.dataset_path)
    qa_dataset_path = resolve_qa_dataset_path(args)
    output_paths = resolve_output_paths(Path(args.output_root))
    samples = load_qa_dataset(qa_dataset_path)
    selected_samples = select_samples(samples, start=args.start, end=args.end, limit=args.limit)

    existing_predictions = index_records_by_qa_or_sample_id(load_jsonl(output_paths["predictions_path"]))
    manifest_records = index_records_by_sample_id(load_jsonl(output_paths["build_manifest_path"]))

    run_stats = {"selected_count": len(selected_samples), "skipped_count": 0, "success_count": 0, "error_count": 0}
    agent = None

    for sample in selected_samples:
        record_id = str(sample.get("qa_id") or sample["sample_id"])
        current_record = existing_predictions.get(record_id)
        if not args.force and is_answer_complete(current_record):
            run_stats["skipped_count"] += 1
            continue

        source_sample_id = str(sample.get("source_sample_id", sample["sample_id"]))
        source_sample_index = int(sample.get("source_sample_index", sample["sample_index"]))
        manifest_record = manifest_records.get(source_sample_id, {})
        index_path = str(
            Path(
                manifest_record.get("index_path")
                or (output_paths["index_dir"] / f"{source_sample_index}.json")
            ).resolve()
        )

        base_record = build_base_record(sample)
        base_record.update(
            {
                "phase": "answer",
                "index_path": index_path,
                "pred_answer": "",
                "formatted_answer": "",
                "retrieval": {"context_text": "", "items": []},
                "evidence_sources": [],
                "llm_output_raw": "",
                "updated_at": utc_timestamp(),
            }
        )

        try:
            if not Path(index_path).exists():
                raise FileNotFoundError(f"NaiveRAG index file not found: {index_path}")

            if agent is None:
                agent = NaiveRAGAgent()

            result = agent.answer_question(
                question=sample["question"],
                index_input=index_path,
                top_k=args.top_k,
            )
            retrieval = result.get("retrieval", {})
            if not isinstance(retrieval, dict):
                retrieval = {"context_text": "", "items": []}

            base_record.update(
                {
                    "status": "success",
                    "pred_answer": result.get("answer", ""),
                    "formatted_answer": result.get("formatted_answer", ""),
                    "retrieval": {
                        "context_text": retrieval.get("context_text", ""),
                        "items": retrieval.get("items", []),
                        "top_k": retrieval.get("top_k", args.top_k),
                    },
                    "evidence_sources": result.get("evidence_sources", []),
                    "llm_output_raw": result.get("llm_output_raw", ""),
                    "updated_at": utc_timestamp(),
                }
            )
            existing_predictions[record_id] = base_record
            run_stats["success_count"] += 1
        except Exception as exc:
            base_record.update(
                {
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "updated_at": utc_timestamp(),
                }
            )
            existing_predictions[record_id] = base_record
            run_stats["error_count"] += 1

    prediction_records = sort_records(existing_predictions.values())
    write_jsonl(output_paths["predictions_path"], prediction_records)

    summary = {
        "phase": "answer",
        "dataset_path": str(dataset_path.resolve()),
        "qa_dataset_path": str(qa_dataset_path.resolve()),
        "predictions_path": str(output_paths["predictions_path"].resolve()),
        "manifest_path": str(output_paths["build_manifest_path"].resolve()),
        "range": {"start": args.start, "end": args.end, "limit": args.limit},
        "force": bool(args.force),
        "top_k": args.top_k,
        **run_stats,
        "completed_count": sum(1 for record in prediction_records if record.get("status") == "success"),
        "failed_count_total": sum(1 for record in prediction_records if record.get("status") == "error"),
        "updated_at": utc_timestamp(),
    }
    write_json(output_paths["answer_summary_path"], summary)
    return summary


def score_stage(args) -> dict:
    output_paths = resolve_output_paths(Path(args.output_root))
    prediction_records = load_jsonl(output_paths["predictions_path"])
    prediction_map = index_records_by_qa_or_sample_id(prediction_records)
    qa_dataset_path = resolve_qa_dataset_path(args)
    samples = load_qa_dataset(qa_dataset_path)
    selected_samples = select_samples(samples, start=args.start, end=args.end, limit=args.limit)
    existing_scored = index_records_by_qa_or_sample_id(load_jsonl(output_paths["scored_results_path"]))

    need_answer_judge = (not args.skip_llm_judge) and args.judge_model_mode in {"answer", "both"}
    need_retrieval_judge = (not args.skip_llm_judge) and args.judge_model_mode in {"retrieval", "both"}
    judge_model = None
    if need_answer_judge or need_retrieval_judge:
        judge_model = LLMProvider().get_llm()

    run_stats = {"selected_count": len(selected_samples), "skipped_count": 0, "success_count": 0, "error_count": 0}

    for sample in selected_samples:
        record_id = str(sample.get("qa_id") or sample["sample_id"])
        current_record = existing_scored.get(record_id)
        if not args.force and not needs_score_work(current_record, need_answer_judge, need_retrieval_judge):
            run_stats["skipped_count"] += 1
            continue

        prediction_record = prediction_map.get(record_id, {})
        retrieval = prediction_record.get("retrieval", {})
        if not isinstance(retrieval, dict):
            retrieval = {}
        pred_answer = str(prediction_record.get("pred_answer", ""))
        context_text = str(retrieval.get("context_text", ""))

        base_record = dict(current_record or {})
        base_record.update(build_base_record(sample))
        base_record.update(
            {
                "phase": "score",
                "index_path": prediction_record.get("index_path", ""),
                "pred_answer": pred_answer,
                "formatted_answer": prediction_record.get("formatted_answer", ""),
                "retrieval": retrieval,
                "evidence_sources": prediction_record.get("evidence_sources", []),
                "llm_output_raw": prediction_record.get("llm_output_raw", ""),
                "official_f1": max_qa_f1_zh_score(pred_answer, sample["answers"]),
                "updated_at": utc_timestamp(),
            }
        )

        try:
            if need_answer_judge:
                if pred_answer.strip():
                    prompt = ANSWER_JUDGE_PROMPT.format(
                        question=sample["question"],
                        answers=json.dumps(sample["answers"], ensure_ascii=False),
                        prediction=pred_answer,
                    )
                    base_record["answer_judge"] = invoke_binary_judge(judge_model, prompt)
                else:
                    base_record["answer_judge"] = 0

            if need_retrieval_judge:
                if context_text.strip():
                    prompt = RETRIEVAL_JUDGE_PROMPT.format(
                        question=sample["question"],
                        answers=json.dumps(sample["answers"], ensure_ascii=False),
                        context=context_text,
                    )
                    base_record["retrieval_judge"] = invoke_binary_judge(judge_model, prompt)
                else:
                    base_record["retrieval_judge"] = 0

            base_record["status"] = "success"
            existing_scored[record_id] = base_record
            run_stats["success_count"] += 1
        except Exception as exc:
            base_record["status"] = "error"
            base_record["error"] = str(exc)
            base_record["traceback"] = traceback.format_exc()
            existing_scored[record_id] = base_record
            run_stats["error_count"] += 1

    scored_records = sort_records(existing_scored.values())
    write_jsonl(output_paths["scored_results_path"], scored_records)

    f1_values = [float(record.get("official_f1", 0.0)) for record in scored_records]
    answer_judge_values = []
    retrieval_judge_values = []
    if need_answer_judge:
        answer_judge_values = [
            record["answer_judge"] for record in scored_records if record.get("answer_judge") in (0, 1)
        ]
    if need_retrieval_judge:
        retrieval_judge_values = [
            record["retrieval_judge"] for record in scored_records if record.get("retrieval_judge") in (0, 1)
        ]

    summary = {
        "phase": "score",
        "qa_dataset_path": str(qa_dataset_path.resolve()),
        "predictions_path": str(output_paths["predictions_path"].resolve()),
        "scored_results_path": str(output_paths["scored_results_path"].resolve()),
        "range": {"start": args.start, "end": args.end, "limit": args.limit},
        "force": bool(args.force),
        "skip_llm_judge": bool(args.skip_llm_judge),
        "judge_model_mode": args.judge_model_mode,
        "count": len(scored_records),
        "avg_f1": (sum(f1_values) / len(f1_values)) if f1_values else 0.0,
        "answer_judge_accuracy": (
            sum(answer_judge_values) / len(answer_judge_values) if answer_judge_values else None
        ),
        "retrieval_judge_accuracy": (
            sum(retrieval_judge_values) / len(retrieval_judge_values) if retrieval_judge_values else None
        ),
        "error_count": sum(1 for record in scored_records if record.get("status") == "error"),
        **run_stats,
        "updated_at": utc_timestamp(),
    }
    write_json(output_paths["score_summary_path"], summary)
    return summary


def all_stage(args) -> dict:
    build_summary = build_stage(args)
    answer_summary = answer_stage(args)
    score_summary = score_stage(args)
    return {
        "phase": "all",
        "build": build_summary,
        "answer": answer_summary,
        "score": score_summary,
        "updated_at": utc_timestamp(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NaiveRAG baseline evaluation for LongBench MultiFieldQA-ZH.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="Path to MultiFieldQA-ZH JSONL.")
        subparser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Evaluation output directory.")
        subparser.add_argument("--start", type=int, default=0, help="Inclusive dataset row start index.")
        subparser.add_argument("--end", type=int, default=None, help="Exclusive dataset row end index.")
        subparser.add_argument("--limit", type=int, default=None, help="Maximum number of samples after slicing.")
        subparser.add_argument("--force", action="store_true", help="Re-run selected samples even if already completed.")

    def add_top_k_argument(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-k retrieval size.")

    def add_qa_dataset_argument(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--qa-dataset-path",
            default=DEFAULT_QA_DATASET_PATH,
            help="Path to expanded QA JSONL for answer/score stages.",
        )

    def add_judge_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--skip-llm-judge",
            action="store_true",
            help="Only compute official LongBench F1 without LLM judges.",
        )
        subparser.add_argument(
            "--judge-model-mode",
            choices=["answer", "retrieval", "both"],
            default="both",
            help="Which LLM judge to run when LLM judging is enabled.",
        )

    build_parser_cmd = subparsers.add_parser("build", help="Build a NaiveRAG index for each sample context.")
    add_common_arguments(build_parser_cmd)
    add_top_k_argument(build_parser_cmd)
    build_parser_cmd.set_defaults(handler=build_stage)

    answer_parser_cmd = subparsers.add_parser("answer", help="Answer each question using NaiveRAG retrieval.")
    add_common_arguments(answer_parser_cmd)
    add_qa_dataset_argument(answer_parser_cmd)
    add_top_k_argument(answer_parser_cmd)
    answer_parser_cmd.set_defaults(handler=answer_stage)

    score_parser_cmd = subparsers.add_parser("score", help="Score predictions with official F1 and optional LLM judges.")
    add_common_arguments(score_parser_cmd)
    add_qa_dataset_argument(score_parser_cmd)
    add_top_k_argument(score_parser_cmd)
    add_judge_arguments(score_parser_cmd)
    score_parser_cmd.set_defaults(handler=score_stage)

    all_parser_cmd = subparsers.add_parser("all", help="Run build, answer, and score in order.")
    add_common_arguments(all_parser_cmd)
    add_qa_dataset_argument(all_parser_cmd)
    add_top_k_argument(all_parser_cmd)
    add_judge_arguments(all_parser_cmd)
    all_parser_cmd.set_defaults(handler=all_stage)

    return parser


def main(argv: Optional[List[str]] = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


def cli(argv: Optional[List[str]] = None) -> int:
    summary = main(argv)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
