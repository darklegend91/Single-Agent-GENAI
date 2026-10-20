"""Strict public contracts and a prose-free model output contract."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, StringConstraints, field_validator, model_validator

UserID = Annotated[str, StringConstraints(strict=True, pattern=r"\AU[0-9]{10}\z")]
NonemptyText = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
SourceField = Annotated[str, StringConstraints(strict=True, pattern=r"\A[A-Za-z][A-Za-z0-9_]{0,127}\z")]
AnswerStatus = Literal["answered", "insufficient_evidence", "refused"]
MetricID = Literal["monthly_settled_total_paise", "monthly_settled_complete", "monthly_settled_available_months"]
LimitationCode = Literal["missing_field", "incomplete_period", "unknown_code", "unresolved_timestamp", "aggregate_only", "no_relevant_evidence", "unsupported_question", "unverified_data_consistency"]
ClaimValue = StrictStr | StrictBool | StrictInt | Annotated[StrictFloat, Field(allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, validate_assignment=True)


class ChatRequest(Contract):
    user_id: UserID
    question: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=2000)]

    @field_validator("question")
    @classmethod
    def meaningful_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must contain non-whitespace characters")
        return value


class Evidence(Contract):
    user_id: UserID
    source_fields: list[SourceField] = Field(min_length=1, max_length=100)
    explanation: NonemptyText


class CustomerAnswer(Contract):
    status: AnswerStatus
    answer: NonemptyText
    evidence: list[Evidence] = Field(default_factory=list, max_length=100)
    limitations: list[NonemptyText] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def answered_requires_evidence(self) -> Self:
        if self.status == "answered" and not self.evidence:
            raise ValueError("Answered responses require source evidence")
        return self


class Claim(Contract):
    """A source value or approved deterministic metric; never factual prose."""

    user_id: UserID
    source_field: SourceField | None = None
    metric_id: MetricID | None = None
    value: ClaimValue
    unit: Literal["paise", "count", "boolean", "text", "code", "timestamp", "ratio"] | None = None

    @model_validator(mode="after")
    def exactly_one_reference(self) -> Self:
        if (self.source_field is None) == (self.metric_id is None):
            raise ValueError("Claim requires exactly one source field or approved metric")
        return self


class AgentResult(Contract):
    status: AnswerStatus
    claims: list[Claim] = Field(default_factory=list, max_length=100)
    limitations: list[LimitationCode] = Field(default_factory=list, max_length=20)
