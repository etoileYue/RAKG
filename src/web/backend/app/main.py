from __future__ import annotations

import json
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable so backend can call existing src/* modules.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import config as rakg_config  # noqa: E402

from app.config import DEFAULT_TASK_OUTPUT_ROOT, REPO_ROOT, SETTINGS  # noqa: E402
from app.db import Database  # noqa: E402
from app.schemas import (  # noqa: E402
    CancelTaskResponse,
    CreateTaskResponse,
    HealthResponse,
    KGCandidateItem,
    KGCandidateListResponse,
    KGBuildRequest,
    LogsResponse,
    QAConversationCreateRequest,
    QAConversationDetailResponse,
    QAConversationListResponse,
    QAConversationSummary,
    QADeleteConversationResponse,
    QAMessage,
    QAMessageCreateRequest,
    QAQueryRequest,
    QAQueryResponse,
    QASendMessageResponse,
    TaskDetail,
    TaskListResponse,
    TaskSummary,
)
from app.serial_executor import SerialTaskExecutor  # noqa: E402
from app.services.qa_service import QAService  # noqa: E402


db = Database(SETTINGS.db_path)
executor = SerialTaskExecutor(db=db, poll_interval_sec=SETTINGS.queue_poll_interval_sec)
qa_service = QAService()
SEED_KG_DIR = REPO_ROOT / "data" / "short" / "processed" / "RAKG_graph_re"


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.log("INFO", "system", "RAKG web backend started")
    try:
        yield
    finally:
        executor.shutdown()
        db.log("INFO", "system", "RAKG web backend stopped")


app = FastAPI(title=SETTINGS.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SETTINGS.cors_origins),
    allow_origin_regex=SETTINGS.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(f"{SETTINGS.api_prefix}/tasks/kg-build", response_model=CreateTaskResponse)
def create_kg_build_task(payload: KGBuildRequest) -> CreateTaskResponse:
    serialized_payload = payload.model_dump()
    resolved_existing = _resolve_existing_kg_path(payload.existing_kg)
    serialized_payload["existing_kg"] = str(resolved_existing) if resolved_existing else None
    _enrich_task_topic_meta(serialized_payload)

    task_id = executor.submit_kg_task(serialized_payload)
    return CreateTaskResponse(task_id=task_id)


@app.post(f"{SETTINGS.api_prefix}/tasks/kg-build/upload", response_model=CreateTaskResponse)
async def create_kg_build_task_upload(
    input_type: str = Form(...),
    json_text: str | None = Form(default=None),
    json_file: UploadFile | None = File(default=None),
    output_dir: str | None = Form(default=None),
    existing_kg: str | None = Form(default=None),
    force_rebuild: bool = Form(default=False),
) -> CreateTaskResponse:
    input_kind = input_type.strip()
    if input_kind not in {"json_text", "json_file"}:
        raise HTTPException(status_code=400, detail="input_type must be one of: json_text, json_file")

    if input_kind == "json_text":
        raw_text = (json_text or "").strip()
        if not raw_text:
            raise HTTPException(status_code=400, detail="json_text is required when input_type=json_text")
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"invalid json_text: {exc}") from exc
    else:
        if json_file is None:
            raise HTTPException(status_code=400, detail="json_file is required when input_type=json_file")
        filename = (json_file.filename or "").strip()
        if filename and not filename.lower().endswith(".json"):
            raise HTTPException(status_code=400, detail="json_file must be a .json file")
        payload_bytes = await json_file.read()
        try:
            parsed = json.loads(payload_bytes.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"json_file must be utf-8 text: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"invalid json_file: {exc}") from exc

    topics = _normalize_topics_payload(parsed)
    uploaded_json_path = _write_uploaded_topics_json(topics)
    resolved_existing = _resolve_existing_kg_path(existing_kg)

    serialized_payload: dict[str, Any] = {
        "input_type": "json_path",
        "json_path": str(uploaded_json_path),
        "output_dir": (output_dir or "").strip() or None,
        "existing_kg": str(resolved_existing) if resolved_existing else None,
        "force_rebuild": bool(force_rebuild),
        "topic_preview": topics[0]["topic"],
        "topic_count": len(topics),
        "upload_source": input_kind,
    }
    task_id = executor.submit_kg_task(serialized_payload)
    return CreateTaskResponse(task_id=task_id)


@app.get(f"{SETTINGS.api_prefix}/tasks/{{task_id}}", response_model=TaskDetail)
def get_task(task_id: str) -> TaskDetail:
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
    _enrich_task_topic_meta(task)
    return TaskDetail(**task)


@app.get(f"{SETTINGS.api_prefix}/tasks", response_model=TaskListResponse)
def list_tasks(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    task_type: str | None = Query(default=None),
    status: str | None = Query(default=None),
) -> TaskListResponse:
    result = db.list_tasks(page=page, page_size=page_size, task_type=task_type, status=status)
    for item in result["items"]:
        _enrich_task_topic_meta(item)
    return TaskListResponse(
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"],
        items=[TaskSummary(**item) for item in result["items"]],
    )


@app.post(f"{SETTINGS.api_prefix}/tasks/{{task_id}}/cancel", response_model=CancelTaskResponse)
def cancel_task(task_id: str) -> CancelTaskResponse:
    accepted, status = executor.cancel_task(task_id)
    if not accepted and status == "not_found":
        raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
    return CancelTaskResponse(task_id=task_id, accepted=accepted, status=status)


@app.post(f"{SETTINGS.api_prefix}/qa/query", response_model=QAQueryResponse)
def qa_query(payload: QAQueryRequest) -> QAQueryResponse:
    try:
        result = qa_service.ask(
            kg_path=payload.kg_path,
            question=payload.question,
            max_hop=payload.max_hop,
            seed_top_k=payload.seed_top_k,
            max_context_items=payload.max_context_items,
        )
        db.log("INFO", "qa", f"QA query executed on {payload.kg_path}")
        return QAQueryResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        db.log("ERROR", "qa", f"QA failed: {exc}")
        raise HTTPException(status_code=500, detail=f"QA failed: {exc}") from exc


@app.get(f"{SETTINGS.api_prefix}/qa/conversations", response_model=QAConversationListResponse)
def list_qa_conversations() -> QAConversationListResponse:
    items = [_normalize_qa_conversation(item) for item in db.list_qa_conversations()]
    return QAConversationListResponse(total=len(items), items=[QAConversationSummary(**item) for item in items])


@app.post(f"{SETTINGS.api_prefix}/qa/conversations", response_model=QAConversationSummary)
def create_qa_conversation(payload: QAConversationCreateRequest) -> QAConversationSummary:
    resolved_kg_path = _resolve_existing_kg_path(payload.kg_path, field_name="kg_path")
    if resolved_kg_path is None:
        raise HTTPException(status_code=400, detail="kg_path is required")

    created = db.create_qa_conversation(
        conversation_id=uuid.uuid4().hex,
        kg_path=str(resolved_kg_path),
        title=payload.title,
    )
    db.log("INFO", "qa", f"Created QA conversation {created['id']}")
    return QAConversationSummary(**_normalize_qa_conversation(created))


@app.get(f"{SETTINGS.api_prefix}/qa/conversations/{{conversation_id}}", response_model=QAConversationDetailResponse)
def get_qa_conversation(conversation_id: str) -> QAConversationDetailResponse:
    conversation = db.get_qa_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail=f"conversation not found: {conversation_id}")

    messages = db.list_qa_messages(conversation_id)
    return QAConversationDetailResponse(
        conversation=QAConversationSummary(**_normalize_qa_conversation(conversation)),
        messages=[QAMessage(**item) for item in messages],
    )


@app.post(
    f"{SETTINGS.api_prefix}/qa/conversations/{{conversation_id}}/messages",
    response_model=QASendMessageResponse,
)
def send_qa_conversation_message(conversation_id: str, payload: QAMessageCreateRequest) -> QASendMessageResponse:
    conversation = db.get_qa_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail=f"conversation not found: {conversation_id}")

    params_snapshot = {
        "max_hop": payload.max_hop,
        "seed_top_k": payload.seed_top_k,
        "max_context_items": payload.max_context_items,
    }
    try:
        result = qa_service.ask(
            kg_path=conversation["kg_path"],
            question=payload.question,
            max_hop=payload.max_hop,
            seed_top_k=payload.seed_top_k,
            max_context_items=payload.max_context_items,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        db.log("ERROR", "qa", f"QA failed in conversation {conversation_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"QA failed: {exc}") from exc

    user_message = db.create_qa_message(
        message_id=uuid.uuid4().hex,
        conversation_id=conversation_id,
        role="user",
        content=payload.question,
    )
    assistant_message = db.create_qa_message(
        message_id=uuid.uuid4().hex,
        conversation_id=conversation_id,
        role="assistant",
        content=result.get("formatted_answer", ""),
        params_snapshot=params_snapshot,
        qa_response_snapshot=result,
    )

    if not user_message or not assistant_message:
        raise HTTPException(status_code=404, detail=f"conversation not found: {conversation_id}")

    updated_conversation = db.get_qa_conversation(conversation_id)
    if not updated_conversation:
        raise HTTPException(status_code=404, detail=f"conversation not found: {conversation_id}")

    db.log("INFO", "qa", f"QA message executed in conversation {conversation_id}")
    return QASendMessageResponse(
        conversation=QAConversationSummary(**_normalize_qa_conversation(updated_conversation)),
        user_message=QAMessage(**user_message),
        assistant_message=QAMessage(**assistant_message),
    )


@app.delete(f"{SETTINGS.api_prefix}/qa/conversations/{{conversation_id}}", response_model=QADeleteConversationResponse)
def delete_qa_conversation(conversation_id: str) -> QADeleteConversationResponse:
    deleted = db.delete_qa_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"conversation not found: {conversation_id}")
    db.log("INFO", "qa", f"Deleted QA conversation {conversation_id}")
    return QADeleteConversationResponse(conversation_id=conversation_id, deleted=True)


@app.get(f"{SETTINGS.api_prefix}/logs", response_model=LogsResponse)
def get_logs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    task_id: str | None = Query(default=None),
    keyword: str | None = Query(default=None),
) -> LogsResponse:
    result = db.fetch_logs(page=page, page_size=page_size, task_id=task_id, keyword=keyword)
    return LogsResponse(**result)


@app.get(f"{SETTINGS.api_prefix}/system/health", response_model=HealthResponse)
def health() -> HealthResponse:
    stats = executor.get_queue_stats()
    model_summary = {
        "use_openai": bool(getattr(rakg_config, "USE_OPENAI", False)),
        "openai_model": getattr(rakg_config, "OPENAI_MODEL", ""),
        "openai_embedding_model": getattr(rakg_config, "OPENAI_EMBEDDING_MODEL", ""),
        "openai_similarity_model": getattr(rakg_config, "OPENAI_SIMILARITY_MODEL", ""),
        "openai_base_url": getattr(rakg_config, "base_url", ""),
        "openai_api_key_set": bool(getattr(rakg_config, "OPENAI_API_KEY", "")),
        "ollama_base_url": getattr(rakg_config, "OLLAMA_BASE_URL", ""),
        "ollama_model": getattr(rakg_config, "DEFAULT_MODEL", ""),
        "ollama_embedding_model": getattr(rakg_config, "EMBEDDING_MODEL", ""),
        "ollama_similarity_model": getattr(rakg_config, "SIMILARITY_MODEL", ""),
    }
    return HealthResponse(
        healthy=True,
        queue={
            "queued": stats.queued,
            "running": 1 if stats.running_task_id else 0,
            "running_task_id": stats.running_task_id,
        },
        model_config_summary=model_summary,
    )


@app.get(f"{SETTINGS.api_prefix}/artifacts/kg")
def read_kg_artifact(path: str = Query(..., description="Absolute or repo-relative KG JSON path")) -> dict[str, Any]:
    resolved = _resolve_repo_path(path)
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"artifact not found: {resolved}")
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {"path": str(resolved), "data": data}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid json artifact: {exc}") from exc


@app.get(f"{SETTINGS.api_prefix}/artifacts/kg/candidates", response_model=KGCandidateListResponse)
def list_kg_candidates() -> KGCandidateListResponse:
    items: list[KGCandidateItem] = []
    seen_paths: set[str] = set()

    for path in _iter_seed_kg_paths():
        canonical = str(path)
        if canonical in seen_paths:
            continue
        seen_paths.add(canonical)
        items.append(
            KGCandidateItem(
                path=canonical,
                source="seed",
                display_name=f"seed/{path.name}",
            )
        )

    succeeded_tasks = db.list_tasks_by_filters(task_type="kg_build", status="succeeded")
    for task in succeeded_tasks:
        task_id = str(task.get("task_id", ""))
        output_payload = task.get("output_payload", {})
        if not isinstance(output_payload, dict):
            continue

        raw_paths: list[str] = []
        graph_paths = output_payload.get("graph_paths")
        if isinstance(graph_paths, list):
            for candidate in graph_paths:
                if candidate is None:
                    continue
                value = str(candidate).strip()
                if value:
                    raw_paths.append(value)

        latest_graph_path = str(output_payload.get("latest_graph_path", "")).strip()
        if latest_graph_path and latest_graph_path not in raw_paths:
            raw_paths.append(latest_graph_path)

        for raw_path in raw_paths:
            resolved = _safe_resolve_repo_path(raw_path)
            if resolved is None or not resolved.exists() or not resolved.is_file():
                continue
            if resolved.suffix.lower() != ".json":
                continue

            canonical = str(resolved)
            if canonical in seen_paths:
                continue
            seen_paths.add(canonical)
            items.append(
                KGCandidateItem(
                    path=canonical,
                    source="task",
                    display_name=f"task/{task_id}/{resolved.name}",
                    task_id=task_id or None,
                )
            )

    return KGCandidateListResponse(total=len(items), items=items)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "RAKG Web API is running"}


def _resolve_existing_kg_path(raw_path: str | None, field_name: str = "existing_kg") -> Path | None:
    candidate_text = (raw_path or "").strip()
    if not candidate_text:
        return None

    resolved = _resolve_repo_path(candidate_text)
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=400, detail=f"{field_name} not found: {resolved}")
    if resolved.suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail=f"{field_name} must be a .json file path")
    return resolved


def _normalize_topics_payload(payload: Any) -> list[dict[str, str]]:
    if isinstance(payload, dict):
        candidates = [payload]
    elif isinstance(payload, list):
        candidates = payload
    else:
        raise HTTPException(status_code=400, detail="Input JSON must be an object or array of objects")

    if not candidates:
        raise HTTPException(status_code=400, detail="Input JSON must not be empty")

    normalized: list[dict[str, str]] = []
    for idx, item in enumerate(candidates, start=1):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"Item {idx} must be an object")

        topic = str(item.get("topic") or "").strip()
        content = str(item.get("content") or "").strip()
        if not topic:
            raise HTTPException(status_code=400, detail=f"Item {idx} requires non-empty `topic`")
        if not content:
            raise HTTPException(status_code=400, detail=f"Item {idx} requires non-empty `content`")

        normalized.append({"topic": topic, "content": content})
    return normalized


def _write_uploaded_topics_json(topics: list[dict[str, str]]) -> Path:
    upload_dir = (DEFAULT_TASK_OUTPUT_ROOT / "uploads").resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{uuid.uuid4().hex}.json"
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(topics, handle, ensure_ascii=False, indent=2)
    return file_path.resolve()


def _enrich_task_topic_meta(container: dict[str, Any]) -> None:
    payload: dict[str, Any]
    input_payload = container.get("input_payload")
    if isinstance(input_payload, dict):
        payload = input_payload
    else:
        payload = container

    topic_preview = str(payload.get("topic_preview") or "").strip()
    raw_topic_count = payload.get("topic_count")
    topic_count = int(raw_topic_count) if isinstance(raw_topic_count, (int, float)) and not isinstance(raw_topic_count, bool) else None
    if topic_count is not None and topic_count <= 0:
        topic_count = None

    if not topic_preview or topic_count is None:
        input_type = str(payload.get("input_type") or "").strip()
        if input_type == "text":
            topic_preview = topic_preview or (str(payload.get("topic") or "").strip() or None)
            topic_count = topic_count or (1 if topic_preview else None)
        elif input_type == "json_path":
            raw_json_path = str(payload.get("json_path") or "").strip()
            resolved = _safe_resolve_repo_path(raw_json_path) if raw_json_path else None
            if resolved is not None and resolved.exists() and resolved.is_file():
                try:
                    with resolved.open("r", encoding="utf-8") as handle:
                        topics = _normalize_topics_payload(json.load(handle))
                    if topics:
                        topic_preview = topic_preview or topics[0]["topic"]
                        topic_count = topic_count or len(topics)
                except Exception:  # noqa: BLE001
                    pass

    container["topic_preview"] = topic_preview or None
    container["topic_count"] = topic_count if topic_count is not None else None


def _resolve_repo_path(raw_path: str) -> Path:
    candidate = Path(raw_path.strip())
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    # 限制为仓库内路径，避免任意文件读取。
    if candidate != REPO_ROOT and REPO_ROOT not in candidate.parents:
        raise HTTPException(status_code=400, detail="path is outside repository")
    return candidate


def _safe_resolve_repo_path(raw_path: str) -> Path | None:
    try:
        return _resolve_repo_path(raw_path)
    except HTTPException:
        return None


def _normalize_qa_conversation(item: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    title = str(normalized.get("title") or "").strip()
    if not title:
        title = "新对话"
    normalized["title"] = title
    return normalized


def _iter_seed_kg_paths() -> list[Path]:
    if not SEED_KG_DIR.exists() or not SEED_KG_DIR.is_dir():
        return []

    def sort_key(path: Path) -> tuple[int, int | str]:
        stem = path.stem
        if stem.isdigit():
            return (0, int(stem))
        return (1, path.name)

    return sorted((item.resolve() for item in SEED_KG_DIR.glob("*.json") if item.is_file()), key=sort_key)
