from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TaskPriority = Literal["low", "medium", "high", "urgent"]
TaskStatus = Literal["pending", "in_progress", "completed"]
NonEmptyText = Annotated[str, Field(min_length=1)]


class ExtractedTask(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    description: NonEmptyText
    owner: str | None = None
    deadline_text: str | None = None
    priority: TaskPriority | None = None
    status: TaskStatus = "pending"
    evidence: str | None = None

    @field_validator("owner", "deadline_text", "evidence")
    @classmethod
    def optional_text_must_not_be_blank(cls, value: str | None) -> str | None:
        if value == "":
            raise ValueError("Use null instead of an empty string.")
        return value


class ExtractedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    description: NonEmptyText
    date_text: NonEmptyText
    evidence: str | None = None

    @field_validator("evidence")
    @classmethod
    def evidence_must_not_be_blank(cls, value: str | None) -> str | None:
        if value == "":
            raise ValueError("Use null instead of an empty string.")
        return value


class DocumentAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    summary: str = Field(min_length=1, max_length=500)
    tasks: list[ExtractedTask] = Field(default_factory=list)
    events: list[ExtractedEvent] = Field(default_factory=list)
    potential_blockers: list[NonEmptyText] = Field(default_factory=list)
    source_title: str | None = None

    @field_validator("source_title")
    @classmethod
    def source_title_must_not_be_blank(cls, value: str | None) -> str | None:
        if value == "":
            raise ValueError("Use null instead of an empty string.")
        return value
