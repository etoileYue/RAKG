"""Stage-wise evaluation entrypoint for LongBench MultiFieldQA-ZH."""

from __future__ import annotations

import argparse
import json
import string
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.kgAgent import NER_Agent
from src.llm_executor import LLMExecutor, LLMTask, LLMTaskError
from src.llm_provider import LLMProvider
from src.utils import parse_json_like_response, run_with_rate_limit_retry


DEFAULT_DATASET_PATH = "data/multifieldqa_zh/test.jsonl"
DEFAULT_QA_DATASET_PATH = "data/multifieldqa_zh/expanded_qa.jsonl"
DEFAULT_OUTPUT_ROOT = "data/eval/multifieldqa_zh"

DEFAULT_MAX_HOP = 1
DEFAULT_SEED_TOP_K = 5
DEFAULT_MAX_CONTEXT_ITEMS = 30
ANSWER_CHECKPOINT_FILE_NAME = "checkpoint_state.json"
SCORE_CHECKPOINT_FILE_NAME = "checkpoint_state.json"

ANSWER_JUDGE_PROMPT = """
你是中文问答自动评测器，需要判断模型答案是否与任一参考答案语义等价。

问题：
{question}

参考答案列表：
{answers}

模型答案：
{prediction}

评测标准：
1. 只要模型答案与任一参考答案表达的是同一核心事实，即判为 1。
2. 若模型答案缺失关键事实、事实错误、答非所问，判为 0。
3. 简洁表述、近义改写、同义替换、不同句式都算等价。
4. 只输出 JSON，不要输出解释或 markdown。

输出格式：
{{"result": 1}}
或
{{"result": 0}}
""".strip()

RETRIEVAL_JUDGE_PROMPT = """
你是中文检索自动评测器，需要判断检索上下文是否覆盖任一参考答案所需的关键信息。

问题：
{question}

参考答案列表：
{answers}

检索上下文：
{context}

评测标准：
1. 只要上下文中已经包含足以支持推出任一参考答案的关键信息，即判为 1。
2. 若上下文缺失关键事实，无法支持得出任一参考答案，判为 0。
3. 不要求逐字复现，但必须信息充分且不相互矛盾。
4. 只输出 JSON，不要输出解释或 markdown。

输出格式：
{{"result": 1}}
或
{{"result": 0}}
""".strip()


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[dict]:
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


def append_jsonl(path: Path, records: Iterable[dict]) -> int:
    records = list(records)
    if not records:
        return 0

    ensure_dir(path.parent)
    appended_count = 0
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            needs_leading_newline = handle.read(1) != b"\n"
    else:
        needs_leading_newline = False

    with path.open("a", encoding="utf-8") as handle:
        if needs_leading_newline:
            handle.write("\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            appended_count += 1
    return appended_count


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_json_atomic(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def coerce_answers(raw_answers) -> List[str]:
    if raw_answers is None:
        return []
    if isinstance(raw_answers, list):
        return [str(item) for item in raw_answers if str(item).strip()]
    text = str(raw_answers).strip()
    return [text] if text else []


def load_dataset(dataset_path: Path) -> List[dict]:
    samples = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for sample_index, line in enumerate(handle):
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)
            sample_id = str(raw.get("_id") or sample_index)
            samples.append(
                {
                    "sample_index": sample_index,
                    "sample_id": sample_id,
                    "question": str(raw.get("input", "")),
                    "context": str(raw.get("context", "")),
                    "answers": coerce_answers(raw.get("answers")),
                    "length": raw.get("length"),
                    "dataset": raw.get("dataset", "multifieldqa_zh"),
                    "language": raw.get("language", "zh"),
                    "all_classes": raw.get("all_classes"),
                }
            )
    return samples


def load_qa_dataset(qa_dataset_path: Path) -> List[dict]:
    qa_samples = []
    with qa_dataset_path.open("r", encoding="utf-8") as handle:
        for qa_index, line in enumerate(handle):
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)
            if isinstance(raw.get("qa_pairs"), list):
                source_sample_index = int(raw.get("source_sample_index", raw.get("sample_index", qa_index)))
                source_sample_id = str(
                    raw.get("source_sample_id")
                    or raw.get("_id")
                    or raw.get("sample_id")
                    or source_sample_index
                )
                for pair in raw.get("qa_pairs") or []:
                    short_qa_id = str(pair.get("qa_id", len(qa_samples))).strip()
                    qa_id = f"{source_sample_index}:{short_qa_id}"
                    qa_source = pair.get("qa_source")
                    if qa_source is None:
                        qa_source = "original" if short_qa_id == "0" else "generated"
                    qa_samples.append(
                        {
                            "qa_id": qa_id,
                            "qa_index": len(qa_samples),
                            "sample_id": source_sample_id,
                            "sample_index": source_sample_index,
                            "source_sample_id": source_sample_id,
                            "source_sample_index": source_sample_index,
                            "question": str(pair.get("question") or pair.get("input") or ""),
                            "answers": coerce_answers(pair.get("answers") or pair.get("answer")),
                            "qa_source": qa_source,
                            "evidence": str(pair.get("evidence", "")),
                            "length": raw.get("length"),
                            "dataset": raw.get("dataset", "multifieldqa_zh"),
                            "language": raw.get("language", "zh"),
                            "all_classes": raw.get("all_classes"),
                        }
                    )
                continue

            source_sample_index = int(raw.get("source_sample_index", raw.get("sample_index", qa_index)))
            source_sample_id = str(
                raw.get("source_sample_id")
                or raw.get("_id")
                or raw.get("sample_id")
                or source_sample_index
            )
            qa_id = str(raw.get("qa_id") or raw.get("sample_id") or source_sample_id)
            qa_source = raw.get("qa_source")
            if qa_source is None:
                qa_source = "original" if "qa_id" not in raw else "generated"
            qa_samples.append(
                {
                    "qa_id": qa_id,
                    "qa_index": qa_index,
                    "sample_id": source_sample_id,
                    "sample_index": source_sample_index,
                    "source_sample_id": source_sample_id,
                    "source_sample_index": source_sample_index,
                    "question": str(raw.get("question") or raw.get("input") or ""),
                    "answers": coerce_answers(raw.get("answers") or raw.get("answer")),
                    "qa_source": qa_source,
                    "evidence": str(raw.get("evidence", "")),
                    "length": raw.get("length"),
                    "dataset": raw.get("dataset", "multifieldqa_zh"),
                    "language": raw.get("language", "zh"),
                    "all_classes": raw.get("all_classes"),
                }
            )
    return qa_samples


def resolve_qa_dataset_path(args) -> Path:
    requested_path = getattr(args, "qa_dataset_path", DEFAULT_QA_DATASET_PATH)
    dataset_path = Path(args.dataset_path)
    if requested_path == DEFAULT_QA_DATASET_PATH and str(dataset_path) != DEFAULT_DATASET_PATH:
        return dataset_path
    qa_dataset_path = Path(requested_path)
    if qa_dataset_path.exists():
        return qa_dataset_path
    if requested_path == DEFAULT_QA_DATASET_PATH:
        if dataset_path.exists():
            return dataset_path
    return qa_dataset_path


def validate_range(start: int, end: Optional[int], limit: Optional[int]) -> None:
    if start < 0:
        raise ValueError("--start must be >= 0")
    if end is not None and end < start:
        raise ValueError("--end must be >= --start")
    if limit is not None and limit < 0:
        raise ValueError("--limit must be >= 0")


def select_samples(samples: List[dict], start: int, end: Optional[int], limit: Optional[int]) -> List[dict]:
    validate_range(start, end, limit)
    subset = samples[start:end]
    if limit is not None:
        subset = subset[:limit]
    return subset


def get_source_sample_index(sample: dict) -> int:
    raw_index = sample.get("source_sample_index", sample.get("sample_index"))
    try:
        return int(raw_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"QA sample missing numeric source_sample_index/sample_index: {sample!r}") from exc


def select_qa_samples(samples: List[dict], start: int, end: Optional[int], limit: Optional[int]) -> List[dict]:
    """Select expanded QA rows by their original source sample index."""
    validate_range(start, end, limit)
    selected = []
    selected_source_indices = set()

    for sample in samples:
        source_sample_index = get_source_sample_index(sample)
        if source_sample_index < start:
            continue
        if end is not None and source_sample_index >= end:
            continue
        if limit is not None and source_sample_index not in selected_source_indices:
            if len(selected_source_indices) >= limit:
                continue
            selected_source_indices.add(source_sample_index)
        selected.append(sample)

    return selected


def index_records_by_sample_id(records: Iterable[dict]) -> Dict[str, dict]:
    indexed = {}
    for record in records:
        sample_id = str(record.get("sample_id", "")).strip()
        if sample_id:
            indexed[sample_id] = record
    return indexed


def index_records_by_qa_or_sample_id(records: Iterable[dict]) -> Dict[str, dict]:
    indexed = {}
    for record in records:
        record_id = str(record.get("qa_id") or record.get("sample_id") or "").strip()
        if record_id:
            indexed[record_id] = record
    return indexed


def sort_records(records: Iterable[dict]) -> List[dict]:
    return sorted(
        records,
        key=lambda item: (
            int(item.get("source_sample_index", item.get("sample_index", 10**12)))
            if str(item.get("source_sample_index", item.get("sample_index", ""))).isdigit()
            else 10**12,
            int(item.get("qa_index", 10**12))
            if str(item.get("qa_index", "")).isdigit()
            else 10**12,
            str(item.get("qa_id", "")),
            str(item.get("sample_id", "")),
        ),
    )


def resolve_output_paths(output_root: Path) -> dict:
    summary_dir = output_root / "summary"
    result_dir = output_root / "result"
    return {
        "output_root": output_root,
        "graphs_dir": output_root / "graphs",
        "summary_dir": summary_dir,
        "result_dir": result_dir,
        "build_manifest_path": summary_dir / "build_manifest.jsonl",
        "build_summary_path": summary_dir / "build_summary.json",
        "predictions_path": result_dir / "predictions.jsonl",
        "answer_summary_path": summary_dir / "answer_summary.json",
        "scored_results_path": result_dir / "scored_results.jsonl",
        "score_summary_path": summary_dir / "score_summary.json",
        "build_cache_root": output_root / "build_cache",
        "answer_cache_root": output_root / "answer_cache",
        "answer_checkpoint_path": output_root / "answer_cache" / ANSWER_CHECKPOINT_FILE_NAME,
        "score_cache_root": output_root / "score_cache",
        "score_checkpoint_path": output_root / "score_cache" / SCORE_CHECKPOINT_FILE_NAME,
    }


def build_base_record(sample: dict) -> dict:
    record = {
        "sample_id": sample["sample_id"],
        "sample_index": sample["sample_index"],
        "question": sample["question"],
        "answers": sample["answers"],
        "length": sample.get("length"),
        "dataset": sample.get("dataset"),
        "language": sample.get("language"),
        "all_classes": sample.get("all_classes"),
    }
    if sample.get("qa_id"):
        record.update(
            {
                "qa_id": sample["qa_id"],
                "qa_index": sample.get("qa_index"),
                "source_sample_id": sample.get("source_sample_id", sample["sample_id"]),
                "source_sample_index": sample.get("source_sample_index", sample["sample_index"]),
                "qa_source": sample.get("qa_source"),
                "evidence": sample.get("evidence", ""),
            }
        )
    return record


def build_error_record(sample: dict, phase: str, error: Exception) -> dict:
    record = build_base_record(sample)
    record.update(
        {
            "phase": phase,
            "status": "error",
            "error": str(error),
            "traceback": traceback.format_exc(),
            "updated_at": utc_timestamp(),
        }
    )
    return record


def build_task_error_record(sample: dict, phase: str, task_error: LLMTaskError) -> dict:
    record = build_base_record(sample)
    record.update(
        {
            "phase": phase,
            "status": "error",
            "error": task_error.message,
            "traceback": task_error.traceback_text,
            "updated_at": utc_timestamp(),
        }
    )
    return record


def build_checkpoint_topics(samples: List[dict]) -> List[dict]:
    return [
        {"topic": sample["sample_id"], "content": sample["context"]}
        for sample in samples
    ]


def build_checkpoint_indices(samples: List[dict]) -> List[int]:
    return [int(sample["sample_index"]) for sample in samples]


def build_graph_path(output_paths: dict, sample: dict) -> Path:
    return output_paths["graphs_dir"] / f"{sample['sample_index']}.json"


def build_stage_cache_paths(output_paths: dict, sample: dict) -> Dict[str, Path]:
    cache_root = output_paths["build_cache_root"]
    idx = sample["sample_index"]
    return {
        "ner": cache_root / "ner_data" / f"output_text_ner_{idx}.jsonl",
        "sim": cache_root / "sim_data" / f"output_sim_{idx}.json",
        "rel": cache_root / "rel_data" / f"output_kg_{idx}.jsonl",
    }


def path_has_content(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def backfill_manifest_from_existing_graphs(
    samples: List[dict],
    existing_manifest: Dict[str, dict],
    output_paths: dict,
    checkpoint_path: str,
) -> int:
    backfilled_count = 0
    for sample in samples:
        current_record = existing_manifest.get(sample["sample_id"])
        if is_build_complete(current_record):
            continue

        graph_path = build_graph_path(output_paths, sample)
        if not path_has_content(graph_path):
            continue

        record = build_base_record(sample)
        record.update(
            {
                "phase": "build",
                "status": "success",
                "topic": sample["sample_id"],
                "graph_path": str(graph_path.resolve()),
                "checkpoint_path": str(Path(checkpoint_path).resolve()) if checkpoint_path else "",
                "updated_at": utc_timestamp(),
                "recovered_from_existing_graph": True,
            }
        )
        existing_manifest[sample["sample_id"]] = record
        backfilled_count += 1
    return backfilled_count


def discover_existing_stage_caches(samples: List[dict], output_paths: dict) -> List[dict]:
    cache_records = []
    for sample in samples:
        existing_stages = {
            stage: str(path.resolve())
            for stage, path in build_stage_cache_paths(output_paths, sample).items()
            if path_has_content(path)
        }
        if existing_stages:
            cache_records.append(
                {
                    "sample_id": sample["sample_id"],
                    "sample_index": sample["sample_index"],
                    "stages": sorted(existing_stages),
                    "paths": existing_stages,
                }
            )
    return cache_records


def is_build_complete(record: Optional[dict]) -> bool:
    return bool(
        record
        and record.get("status") == "success"
        and str(record.get("graph_path", "")).strip()
        and Path(record["graph_path"]).exists()
    )


def is_answer_complete(record: Optional[dict]) -> bool:
    return bool(record and record.get("status") == "success")


def load_record_checkpoint(path: Path, phase: str) -> dict:
    if not path.exists():
        return {"version": 1, "phase": phase, "records": {}}
    with path.open("r", encoding="utf-8") as handle:
        checkpoint = json.load(handle)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid {phase} checkpoint at {path}: root must be a JSON object")
    records = checkpoint.get("records")
    if not isinstance(records, dict):
        checkpoint["records"] = {}
    else:
        checkpoint["records"] = {
            str(record_id): record
            for record_id, record in records.items()
            if isinstance(record, dict)
        }
    checkpoint.setdefault("version", 1)
    checkpoint.setdefault("phase", phase)
    return checkpoint


def load_answer_checkpoint(path: Path) -> dict:
    return load_record_checkpoint(path, "answer")


def load_score_checkpoint(path: Path) -> dict:
    return load_record_checkpoint(path, "score")


def checkpoint_stage_record(
    checkpoint_state: dict,
    checkpoint_path: Path,
    phase: str,
    record_id: str,
    record: dict,
) -> None:
    checkpoint_state.setdefault("version", 1)
    checkpoint_state.setdefault("phase", phase)
    checkpoint_state.setdefault("records", {})[record_id] = record
    checkpoint_state["updated_at"] = utc_timestamp()
    write_json_atomic(checkpoint_path, checkpoint_state)


def checkpoint_answer_record(
    checkpoint_state: dict,
    checkpoint_path: Path,
    record_id: str,
    record: dict,
) -> None:
    checkpoint_stage_record(checkpoint_state, checkpoint_path, "answer", record_id, record)


def checkpoint_score_record(
    checkpoint_state: dict,
    checkpoint_path: Path,
    record_id: str,
    record: dict,
) -> None:
    checkpoint_stage_record(checkpoint_state, checkpoint_path, "score", record_id, record)


def needs_score_work(
    record: Optional[dict],
    need_answer_judge: bool,
    need_retrieval_judge: bool,
) -> bool:
    if not record or record.get("status") != "success":
        return True
    if "official_f1" not in record:
        return True
    if need_answer_judge and record.get("answer_judge") not in (0, 1):
        return True
    if need_retrieval_judge and record.get("retrieval_judge") not in (0, 1):
        return True
    return False


def _get_jieba():
    try:
        import jieba  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "jieba is required for MultiFieldQA-ZH scoring. Install it before running `score`."
        ) from exc
    return jieba


def normalize_zh_answer(text: str) -> str:
    cn_punctuation = (
        "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～"
        "｟｠｢｣､、〃》「」『』〖〗〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    )
    all_punctuation = set(string.punctuation + cn_punctuation)
    lowered = str(text or "").lower()
    without_punc = "".join(ch for ch in lowered if ch not in all_punctuation)
    return "".join(without_punc.split())


def f1_score(prediction_tokens: List[str], ground_truth_tokens: List[str]) -> float:
    from collections import Counter

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_zh_score(prediction: str, ground_truth: str) -> float:
    jieba = _get_jieba()
    prediction_tokens = list(jieba.cut(str(prediction or ""), cut_all=False))
    ground_truth_tokens = list(jieba.cut(str(ground_truth or ""), cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if token]
    ground_truth_tokens = [token for token in ground_truth_tokens if token]
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    return f1_score(prediction_tokens, ground_truth_tokens)


def max_qa_f1_zh_score(prediction: str, answers: List[str]) -> float:
    scores = [qa_f1_zh_score(prediction, answer) for answer in answers]
    return max(scores) if scores else 0.0


def parse_binary_result(payload) -> int:
    if isinstance(payload, dict):
        payload = payload.get("result")
    if isinstance(payload, bool):
        return int(payload)
    if isinstance(payload, (int, float)):
        return 1 if int(payload) == 1 else 0
    text = str(payload or "").strip().lower()
    if text in {"1", "true", "yes"}:
        return 1
    if text in {"0", "false", "no"}:
        return 0
    raise ValueError(f"Unable to parse judge result from payload: {payload!r}")


def invoke_binary_judge(model, prompt: str) -> int:
    response = model.invoke(prompt)
    parsed = parse_json_like_response(response)
    if parsed is None:
        raw = response.content if hasattr(response, "content") else str(response)
        return parse_binary_result(raw)
    return parse_binary_result(parsed)


def rate_limit_retry_kwargs(args, label: str) -> dict:
    return {
        "label": label,
        "max_retries": args.rate_limit_max_retries,
        "initial_wait_seconds": args.rate_limit_initial_wait,
        "max_wait_seconds": args.rate_limit_max_wait,
    }


def build_stage(args) -> dict:
    dataset_path = Path(args.dataset_path)
    output_paths = resolve_output_paths(Path(args.output_root))
    ensure_dir(output_paths["graphs_dir"])
    ensure_dir(output_paths["build_cache_root"])

    samples = load_dataset(dataset_path)
    selected_samples = select_samples(samples, start=args.start, end=args.end, limit=args.limit)
    existing_manifest = index_records_by_sample_id(load_jsonl(output_paths["build_manifest_path"]))
    checkpoint_path = str(output_paths["build_cache_root"] / NER_Agent.CHECKPOINT_FILE_NAME)
    backfilled_manifest_count = backfill_manifest_from_existing_graphs(
        selected_samples,
        existing_manifest,
        output_paths,
        checkpoint_path,
    )

    run_stats = {
        "selected_count": len(selected_samples),
        "skipped_count": 0,
        "success_count": 0,
        "error_count": 0,
    }
    pending_samples = []
    for sample in selected_samples:
        current_record = existing_manifest.get(sample["sample_id"])
        if not args.force and is_build_complete(current_record):
            run_stats["skipped_count"] += 1
        else:
            pending_samples.append(sample)

    agent = None
    checkpoint_state = None
    checkpoint_loaded = False
    auto_resume = False
    ner_output_dir = output_paths["build_cache_root"] / "ner_data"
    rel_output_dir = output_paths["build_cache_root"] / "rel_data"
    sim_output_dir = output_paths["build_cache_root"] / "sim_data"
    ensure_dir(ner_output_dir)
    ensure_dir(rel_output_dir)
    ensure_dir(sim_output_dir)
    existing_stage_cache_records = discover_existing_stage_caches(pending_samples, output_paths)

    if pending_samples:
        agent = NER_Agent()
        checkpoint_state, checkpoint_path, checkpoint_loaded = agent._prepare_checkpoint_state(
            output_dir=str(output_paths["build_cache_root"]),
            topics=build_checkpoint_topics(samples),
            topic_indices=build_checkpoint_indices(samples),
            ner_output_dir=str(ner_output_dir),
            rel_output_dir=str(rel_output_dir),
            sim_output_dir=str(sim_output_dir),
            graph_output_dir=str(output_paths["graphs_dir"]),
            force_rebuild=bool(args.force),
        )
        auto_resume = (not args.force) and (
            checkpoint_loaded or bool(existing_stage_cache_records)
        )

    for sample in pending_samples:
        try:
            result = run_with_rate_limit_retry(
                lambda: agent.process(
                    topic_data={"topic": sample["sample_id"], "content": sample["context"]},
                    idx=sample["sample_index"],
                    total_topics=len(samples),
                    ner_output_dir=str(ner_output_dir),
                    rel_output_dir=str(rel_output_dir),
                    sim_output_dir=str(sim_output_dir),
                    graph_output_dir=str(output_paths["graphs_dir"]),
                    checkpoint_state=checkpoint_state,
                    checkpoint_path=checkpoint_path,
                    auto_resume=auto_resume,
                ),
                **rate_limit_retry_kwargs(args, f"build sample {sample['sample_id']}"),
            )
            record = build_base_record(sample)
            record.update(
                {
                    "phase": "build",
                    "status": "success",
                    "topic": sample["sample_id"],
                    "graph_path": str(Path(result["output_path"]).resolve()),
                    "checkpoint_path": str(Path(checkpoint_path).resolve()) if checkpoint_path else "",
                    "updated_at": utc_timestamp(),
                }
            )
            existing_manifest[sample["sample_id"]] = record
            run_stats["success_count"] += 1
        except Exception as exc:
            record = build_error_record(sample, phase="build", error=exc)
            record.update(
                {
                    "graph_path": str(
                        (output_paths["graphs_dir"] / f"{sample['sample_index']}.json").resolve()
                    ),
                    "checkpoint_path": str(Path(checkpoint_path).resolve()) if checkpoint_path else "",
                }
            )
            existing_manifest[sample["sample_id"]] = record
            run_stats["error_count"] += 1

    manifest_records = sort_records(existing_manifest.values())
    write_jsonl(output_paths["build_manifest_path"], manifest_records)

    summary = {
        "phase": "build",
        "dataset_path": str(dataset_path.resolve()),
        "output_root": str(output_paths["output_root"].resolve()),
        "graphs_dir": str(output_paths["graphs_dir"].resolve()),
        "manifest_path": str(output_paths["build_manifest_path"].resolve()),
        "checkpoint_path": str(Path(checkpoint_path).resolve()) if checkpoint_path else "",
        "checkpoint_loaded": bool(checkpoint_loaded),
        "auto_resume": bool(auto_resume),
        "backfilled_manifest_count": backfilled_manifest_count,
        "existing_stage_cache_count": len(existing_stage_cache_records),
        "existing_stage_caches": existing_stage_cache_records,
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
    selected_samples = select_qa_samples(samples, start=args.start, end=args.end, limit=args.limit)

    existing_predictions = index_records_by_qa_or_sample_id(load_jsonl(output_paths["predictions_path"]))
    answer_checkpoint_path = output_paths["answer_checkpoint_path"]
    answer_checkpoint_loaded = answer_checkpoint_path.exists()
    answer_checkpoint_state = load_answer_checkpoint(answer_checkpoint_path)
    answer_checkpoint_records = answer_checkpoint_state.get("records", {})
    manifest_records = index_records_by_sample_id(load_jsonl(output_paths["build_manifest_path"]))
    graphs_dir = output_paths["graphs_dir"]

    run_stats = {"selected_count": len(selected_samples), "skipped_count": 0, "success_count": 0, "error_count": 0}
    appended_records = []
    pending_payloads = []
    restored_from_checkpoint_count = 0

    for sample in selected_samples:
        record_id = str(sample.get("qa_id") or sample["sample_id"])
        current_record = existing_predictions.get(record_id)
        checkpoint_record = answer_checkpoint_records.get(record_id)
        if (
            not args.force
            and not is_answer_complete(current_record)
            and is_answer_complete(checkpoint_record)
        ):
            restored_record = dict(checkpoint_record)
            restored_record["recovered_from_answer_checkpoint"] = True
            existing_predictions[record_id] = restored_record
            current_record = restored_record
            restored_from_checkpoint_count += 1

        if not args.force and is_answer_complete(current_record):
            run_stats["skipped_count"] += 1
            continue

        source_sample_id = str(sample.get("source_sample_id", sample["sample_id"]))
        source_sample_index = int(sample.get("source_sample_index", sample["sample_index"]))
        manifest_record = manifest_records.get(source_sample_id, {})
        graph_path = str(
            Path(
                manifest_record.get("graph_path")
                or (graphs_dir / f"{source_sample_index}.json")
            ).resolve()
        )

        base_record = build_base_record(sample)
        base_record.update(
            {
                "phase": "answer",
                "graph_path": graph_path,
                "checkpoint_path": str(answer_checkpoint_path.resolve()),
                "pred_answer": "",
                "formatted_answer": "",
                "graph_paths": [],
                "retrieval": {
                    "context_text": "",
                    "evidence_items": [],
                    "graph_paths": [],
                    "matched_nodes": [],
                    "seed_nodes": [],
                },
                "updated_at": utc_timestamp(),
            }
        )
        pending_payloads.append(
            {
                "sample": sample,
                "record_id": record_id,
                "graph_path": graph_path,
                "base_record": base_record,
            }
        )

    answer_checkpoint_state.update(
        {
            "version": 1,
            "phase": "answer",
            "dataset_path": str(dataset_path.resolve()),
            "qa_dataset_path": str(qa_dataset_path.resolve()),
            "output_root": str(output_paths["output_root"].resolve()),
            "predictions_path": str(output_paths["predictions_path"].resolve()),
            "updated_at": utc_timestamp(),
        }
    )
    answer_checkpoint_lock = threading.Lock()

    def answer_one(payload: dict) -> dict:
        sample = payload["sample"]
        record_id = payload["record_id"]
        graph_path = payload["graph_path"]
        base_record = dict(payload["base_record"])
        try:
            if not Path(graph_path).exists():
                raise FileNotFoundError(f"Graph file not found: {graph_path}")

            agent = NER_Agent()
            cache_key = f"multifieldqa_zh:{record_id}"

            def answer_attempt():
                try:
                    agent.initialize_qa_graph_index(graph_path, cache_key=cache_key, force_rebuild=True)
                    return agent.answer_question_with_kg(
                        question=sample["question"],
                        cache_key=cache_key,
                        max_hop=DEFAULT_MAX_HOP,
                        seed_top_k=DEFAULT_SEED_TOP_K,
                        max_context_items=DEFAULT_MAX_CONTEXT_ITEMS,
                    )
                finally:
                    agent.clear_qa_graph_index(cache_key)

            result = run_with_rate_limit_retry(
                answer_attempt,
                **rate_limit_retry_kwargs(args, f"answer sample {record_id}"),
            )

            base_record.update(
                {
                    "status": "success",
                    "pred_answer": result.get("answer", ""),
                    "formatted_answer": result.get("formatted_answer", ""),
                    "graph_paths": result.get("graph_paths", []),
                    "retrieval": result.get("retrieval", {}),
                    "llm_output_raw": result.get("llm_output_raw", ""),
                    "updated_at": utc_timestamp(),
                }
            )
        except Exception as exc:
            base_record.update(
                {
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "updated_at": utc_timestamp(),
                }
            )
        with answer_checkpoint_lock:
            checkpoint_answer_record(
                answer_checkpoint_state,
                answer_checkpoint_path,
                record_id,
                base_record,
            )
        return {"record_id": record_id, "record": base_record}

    tasks = [
        LLMTask(
            kind="multifieldqa_zh_answer",
            payload=payload,
            invoke_fn=answer_one,
            metadata={
                "record_id": payload["record_id"],
                "sample": payload["sample"],
                "graph_path": payload["graph_path"],
            },
        )
        for payload in pending_payloads
    ]
    task_results = LLMExecutor().invoke_batch(
        tasks,
        progress_enabled=True,
        progress_label="MFQA-ZH answer",
    )
    for result in task_results:
        if isinstance(result, LLMTaskError):
            sample = result.metadata.get("sample", {})
            record_id = str(result.metadata.get("record_id") or sample.get("qa_id") or sample.get("sample_id") or "")
            record = build_task_error_record(sample, phase="answer", task_error=result)
            graph_path = str(result.metadata.get("graph_path", ""))
            record.update(
                {
                    "graph_path": graph_path,
                    "checkpoint_path": str(answer_checkpoint_path.resolve()),
                    "pred_answer": "",
                    "formatted_answer": "",
                    "graph_paths": [],
                    "retrieval": {
                        "context_text": "",
                        "evidence_items": [],
                        "graph_paths": [],
                        "matched_nodes": [],
                        "seed_nodes": [],
                    },
                }
            )
        else:
            record_id = result["record_id"]
            record = result["record"]

        existing_predictions[record_id] = record
        appended_records.append(record)
        if record.get("status") == "success":
            run_stats["success_count"] += 1
        else:
            run_stats["error_count"] += 1

    prediction_records = sort_records(existing_predictions.values())
    appended_count = len(appended_records)
    write_jsonl(output_paths["predictions_path"], prediction_records)

    summary = {
        "phase": "answer",
        "dataset_path": str(dataset_path.resolve()),
        "qa_dataset_path": str(qa_dataset_path.resolve()),
        "predictions_path": str(output_paths["predictions_path"].resolve()),
        "manifest_path": str(output_paths["build_manifest_path"].resolve()),
        "checkpoint_path": str(answer_checkpoint_path.resolve()),
        "checkpoint_loaded": bool(answer_checkpoint_loaded),
        "restored_from_checkpoint_count": restored_from_checkpoint_count,
        "range": {"start": args.start, "end": args.end, "limit": args.limit},
        "force": bool(args.force),
        "qa_defaults": {
            "max_hop": DEFAULT_MAX_HOP,
            "seed_top_k": DEFAULT_SEED_TOP_K,
            "max_context_items": DEFAULT_MAX_CONTEXT_ITEMS,
        },
        "appended_count": appended_count,
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
    selected_samples = select_qa_samples(samples, start=args.start, end=args.end, limit=args.limit)
    existing_scored = index_records_by_qa_or_sample_id(load_jsonl(output_paths["scored_results_path"]))
    score_checkpoint_path = output_paths["score_checkpoint_path"]
    score_checkpoint_loaded = score_checkpoint_path.exists()
    score_checkpoint_state = load_score_checkpoint(score_checkpoint_path)
    score_checkpoint_records = score_checkpoint_state.get("records", {})

    need_answer_judge = (not args.skip_llm_judge) and args.judge_model_mode in {"answer", "both"}
    need_retrieval_judge = (not args.skip_llm_judge) and args.judge_model_mode in {"retrieval", "both"}
    judge_model = None
    if need_answer_judge or need_retrieval_judge:
        judge_model = LLMProvider().get_llm()

    run_stats = {"selected_count": len(selected_samples), "skipped_count": 0, "success_count": 0, "error_count": 0}
    appended_records = []
    pending_payloads = []
    restored_from_checkpoint_count = 0

    for sample in selected_samples:
        record_id = str(sample.get("qa_id") or sample["sample_id"])
        current_record = existing_scored.get(record_id)
        checkpoint_record = score_checkpoint_records.get(record_id)
        if (
            not args.force
            and needs_score_work(current_record, need_answer_judge, need_retrieval_judge)
            and not needs_score_work(checkpoint_record, need_answer_judge, need_retrieval_judge)
        ):
            restored_record = dict(checkpoint_record)
            restored_record["recovered_from_score_checkpoint"] = True
            existing_scored[record_id] = restored_record
            current_record = restored_record
            restored_from_checkpoint_count += 1

        if not args.force and not needs_score_work(current_record, need_answer_judge, need_retrieval_judge):
            run_stats["skipped_count"] += 1
            continue

        prediction_record = prediction_map.get(record_id, {})
        retrieval = prediction_record.get("retrieval", {}) if isinstance(prediction_record.get("retrieval", {}), dict) else {}
        pred_answer = str(prediction_record.get("pred_answer", ""))
        context_text = str(retrieval.get("context_text", ""))

        base_record = dict(current_record or {})
        base_record.update(build_base_record(sample))
        base_record.update(
            {
                "phase": "score",
                "checkpoint_path": str(score_checkpoint_path.resolve()),
                "graph_path": prediction_record.get("graph_path", ""),
                "graph_paths": prediction_record.get("graph_paths", []),
                "pred_answer": pred_answer,
                "formatted_answer": prediction_record.get("formatted_answer", ""),
                "retrieval": retrieval,
                "official_f1": max_qa_f1_zh_score(pred_answer, sample["answers"]),
                "updated_at": utc_timestamp(),
            }
        )
        pending_payloads.append(
            {
                "sample": sample,
                "record_id": record_id,
                "prediction_record": prediction_record,
                "pred_answer": pred_answer,
                "context_text": context_text,
                "base_record": base_record,
            }
        )

    score_checkpoint_state.update(
        {
            "version": 1,
            "phase": "score",
            "qa_dataset_path": str(qa_dataset_path.resolve()),
            "output_root": str(output_paths["output_root"].resolve()),
            "predictions_path": str(output_paths["predictions_path"].resolve()),
            "scored_results_path": str(output_paths["scored_results_path"].resolve()),
            "skip_llm_judge": bool(args.skip_llm_judge),
            "judge_model_mode": args.judge_model_mode,
            "updated_at": utc_timestamp(),
        }
    )
    score_checkpoint_lock = threading.Lock()

    def score_one(payload: dict) -> dict:
        sample = payload["sample"]
        record_id = payload["record_id"]
        pred_answer = payload["pred_answer"]
        context_text = payload["context_text"]
        base_record = dict(payload["base_record"])
        try:
            if need_answer_judge:
                if pred_answer.strip():
                    prompt = ANSWER_JUDGE_PROMPT.format(
                        question=sample["question"],
                        answers=json.dumps(sample["answers"], ensure_ascii=False),
                        prediction=pred_answer,
                    )
                    base_record["answer_judge"] = run_with_rate_limit_retry(
                        lambda: invoke_binary_judge(judge_model, prompt),
                        **rate_limit_retry_kwargs(args, f"answer judge sample {record_id}"),
                    )
                else:
                    base_record["answer_judge"] = 0

            if need_retrieval_judge:
                if context_text.strip():
                    prompt = RETRIEVAL_JUDGE_PROMPT.format(
                        question=sample["question"],
                        answers=json.dumps(sample["answers"], ensure_ascii=False),
                        context=context_text,
                    )
                    base_record["retrieval_judge"] = run_with_rate_limit_retry(
                        lambda: invoke_binary_judge(judge_model, prompt),
                        **rate_limit_retry_kwargs(args, f"retrieval judge sample {record_id}"),
                    )
                else:
                    base_record["retrieval_judge"] = 0

            base_record["status"] = "success"
        except Exception as exc:
            base_record["status"] = "error"
            base_record["error"] = str(exc)
            base_record["traceback"] = traceback.format_exc()
        base_record["updated_at"] = utc_timestamp()
        with score_checkpoint_lock:
            checkpoint_score_record(
                score_checkpoint_state,
                score_checkpoint_path,
                record_id,
                base_record,
            )
        return {"record_id": record_id, "record": base_record}

    tasks = [
        LLMTask(
            kind="multifieldqa_zh_score",
            payload=payload,
            invoke_fn=score_one,
            metadata={"record_id": payload["record_id"], "sample": payload["sample"]},
        )
        for payload in pending_payloads
    ]
    task_results = LLMExecutor().invoke_batch(
        tasks,
        progress_enabled=True,
        progress_label="MFQA-ZH score",
    )
    for result in task_results:
        if isinstance(result, LLMTaskError):
            sample = result.metadata.get("sample", {})
            record_id = str(result.metadata.get("record_id") or sample.get("qa_id") or sample.get("sample_id") or "")
            record = build_task_error_record(sample, phase="score", task_error=result)
        else:
            record_id = result["record_id"]
            record = result["record"]

        with score_checkpoint_lock:
            checkpoint_score_record(
                score_checkpoint_state,
                score_checkpoint_path,
                record_id,
                record,
            )
        existing_scored[record_id] = record
        appended_records.append(record)
        if record.get("status") == "success":
            run_stats["success_count"] += 1
        else:
            run_stats["error_count"] += 1

    scored_records = sort_records(existing_scored.values())
    appended_count = len(appended_records)
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
        "checkpoint_path": str(score_checkpoint_path.resolve()),
        "checkpoint_loaded": bool(score_checkpoint_loaded),
        "restored_from_checkpoint_count": restored_from_checkpoint_count,
        "range": {"start": args.start, "end": args.end, "limit": args.limit},
        "force": bool(args.force),
        "skip_llm_judge": bool(args.skip_llm_judge),
        "judge_model_mode": args.judge_model_mode,
        "count": len(scored_records),
        "appended_count": appended_count,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-wise evaluation for LongBench MultiFieldQA-ZH.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="Path to MultiFieldQA-ZH JSONL.")
        subparser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Evaluation output directory.")
        subparser.add_argument("--start", type=int, default=0, help="Inclusive source dataset row start index.")
        subparser.add_argument("--end", type=int, default=None, help="Exclusive source dataset row end index.")
        subparser.add_argument("--limit", type=int, default=None, help="Maximum number of source samples after slicing.")
        subparser.add_argument("--force", action="store_true", help="Re-run selected samples even if already completed.")
        subparser.add_argument(
            "--rate-limit-max-retries",
            type=int,
            default=8,
            help="Maximum retries for rate-limit errors after the initial attempt.",
        )
        subparser.add_argument(
            "--rate-limit-initial-wait",
            type=float,
            default=30.0,
            help="Initial wait in seconds before retrying rate-limit errors.",
        )
        subparser.add_argument(
            "--rate-limit-max-wait",
            type=float,
            default=300.0,
            help="Maximum wait in seconds between rate-limit retries.",
        )

    build_parser_cmd = subparsers.add_parser("build", help="Build a graph for each sample context.")
    add_common_arguments(build_parser_cmd)
    build_parser_cmd.set_defaults(handler=build_stage)

    answer_parser_cmd = subparsers.add_parser("answer", help="Answer each question using the built graph.")
    add_common_arguments(answer_parser_cmd)
    answer_parser_cmd.add_argument(
        "--qa-dataset-path",
        default=DEFAULT_QA_DATASET_PATH,
        help="Path to expanded QA JSONL for answer stage.",
    )
    answer_parser_cmd.set_defaults(handler=answer_stage)

    score_parser_cmd = subparsers.add_parser("score", help="Score predictions with official F1 and optional LLM judges.")
    add_common_arguments(score_parser_cmd)
    score_parser_cmd.add_argument(
        "--qa-dataset-path",
        default=DEFAULT_QA_DATASET_PATH,
        help="Path to expanded QA JSONL for score stage.",
    )
    score_parser_cmd.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Only compute official LongBench F1 without LLM judges.",
    )
    score_parser_cmd.add_argument(
        "--judge-model-mode",
        choices=["answer", "retrieval", "both"],
        default="both",
        help="Which LLM judge to run when LLM judging is enabled.",
    )
    score_parser_cmd.set_defaults(handler=score_stage)

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
