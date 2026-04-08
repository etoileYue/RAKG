from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable so backend can call existing src/* modules.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import config as rakg_config  # noqa: E402

from app.config import REPO_ROOT, SETTINGS  # noqa: E402
from app.db import Database  # noqa: E402
from app.schemas import (  # noqa: E402
    CancelTaskResponse,
    CreateTaskResponse,
    HealthResponse,
    KGBuildRequest,
    LogsResponse,
    QAQueryRequest,
    QAQueryResponse,
    TaskDetail,
    TaskListResponse,
    TaskSummary,
)
from app.serial_executor import SerialTaskExecutor  # noqa: E402
from app.services.qa_service import QAService  # noqa: E402


db = Database(SETTINGS.db_path)
executor = SerialTaskExecutor(db=db, poll_interval_sec=SETTINGS.queue_poll_interval_sec)
qa_service = QAService()


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
    task_id = executor.submit_kg_task(payload.model_dump())
    return CreateTaskResponse(task_id=task_id)


@app.get(f"{SETTINGS.api_prefix}/tasks/{{task_id}}", response_model=TaskDetail)
def get_task(task_id: str) -> TaskDetail:
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
    return TaskDetail(**task)


@app.get(f"{SETTINGS.api_prefix}/tasks", response_model=TaskListResponse)
def list_tasks(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    task_type: str | None = Query(default=None),
    status: str | None = Query(default=None),
) -> TaskListResponse:
    result = db.list_tasks(page=page, page_size=page_size, task_type=task_type, status=status)
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


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "RAKG Web API is running"}


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
