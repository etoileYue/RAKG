from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


TaskStatus = Literal["queued", "running", "succeeded", "failed", "canceled"]


class KGBuildRequest(BaseModel):
    input_type: Literal["text", "json_path"]
    text: str | None = None
    topic: str = "web_topic"
    json_path: str | None = None
    output_dir: str | None = None
    existing_kg: str | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "KGBuildRequest":
        if self.input_type == "text":
            if not self.text or not self.text.strip():
                raise ValueError("text input requires non-empty `text`")
        if self.input_type == "json_path":
            if not self.json_path or not self.json_path.strip():
                raise ValueError("json_path input requires non-empty `json_path`")
        return self

    @field_validator("existing_kg")
    @classmethod
    def normalize_existing_kg(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class TaskSummary(BaseModel):
    task_id: str
    task_type: str
    status: TaskStatus
    progress: float
    message: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    topic_preview: str | None = None
    topic_count: int | None = None


class TaskDetail(TaskSummary):
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    cancel_requested: bool = False


class TaskListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[TaskSummary]


class CreateTaskResponse(BaseModel):
    task_id: str


class CancelTaskResponse(BaseModel):
    task_id: str
    accepted: bool
    status: str


class QAQueryRequest(BaseModel):
    kg_path: str
    question: str
    max_hop: int = Field(default=2, ge=1, le=3)
    seed_top_k: int = Field(default=5, ge=1, le=20)
    max_context_items: int = Field(default=30, ge=1, le=100)

    @field_validator("kg_path", "question")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be non-empty")
        return value.strip()


class QAQueryResponse(BaseModel):
    question: str
    formatted_answer: str
    answer: str
    retrieved_context: dict[str, Any]
    evidence_sources: list[dict[str, str]]
    graph_paths: list[str]
    intermediate: dict[str, Any]


MessageRole = Literal["user", "assistant"]


class QAConversationCreateRequest(BaseModel):
    kg_path: str
    title: str | None = None

    @field_validator("kg_path")
    @classmethod
    def normalize_kg_path(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("kg_path must be non-empty")
        return value.strip()

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class QAConversationSummary(BaseModel):
    id: str
    title: str
    kg_path: str
    created_at: str
    updated_at: str
    message_count: int = 0
    last_message_preview: str | None = None


class QAConversationListResponse(BaseModel):
    total: int
    items: list[QAConversationSummary]


class QAMessageCreateRequest(BaseModel):
    question: str
    max_hop: int = Field(default=2, ge=1, le=3)
    seed_top_k: int = Field(default=5, ge=1, le=20)
    max_context_items: int = Field(default=30, ge=1, le=100)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("question must be non-empty")
        return value.strip()


class QAMessage(BaseModel):
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    params_snapshot: dict[str, Any] = Field(default_factory=dict)
    qa_response_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class QAConversationDetailResponse(BaseModel):
    conversation: QAConversationSummary
    messages: list[QAMessage]


class QASendMessageResponse(BaseModel):
    conversation: QAConversationSummary
    user_message: QAMessage
    assistant_message: QAMessage


class QADeleteConversationResponse(BaseModel):
    conversation_id: str
    deleted: bool


class LogsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[dict[str, Any]]


class HealthResponse(BaseModel):
    healthy: bool
    queue: dict[str, Any]
    model_config_summary: dict[str, Any]


class KGCandidateItem(BaseModel):
    path: str
    source: Literal["seed", "task"]
    display_name: str
    task_id: str | None = None


class KGCandidateListResponse(BaseModel):
    total: int
    items: list[KGCandidateItem]
