from typing import Literal, TypeAlias

from pydantic import BaseModel, Field
from pydantic import model_validator

RunType: TypeAlias = Literal[
    "lecture_compile",
    "copilot_answer",
    "mcp_tool_request",
]
RunStatus: TypeAlias = Literal["running", "finished", "blocked"]
RunVerdictStatus: TypeAlias = Literal["accept", "revise", "block"]


class RunContract(BaseModel):
    run_type: RunType
    entrypoint: str
    target_ref: str
    expected_profile: str | None = None
    expected_artifacts: list[str] = Field(default_factory=list)
    writeback_allowed: bool = False
    minimum_acceptance: list[str] = Field(default_factory=list)


class PolicySnapshot(BaseModel):
    run_type: RunType
    allowed_entrypoint: str
    sandbox_level: str
    writeback_allowed: bool
    capabilities: list[str] = Field(default_factory=list)


class RunArtifact(BaseModel):
    artifact_type: str
    path_or_ref: str
    producer: str


class RunVerdict(BaseModel):
    status: RunVerdictStatus
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class RunRecord(BaseModel):
    trace_id: str
    run_type: RunType
    entrypoint: str
    target_ref: str
    run_contract: RunContract
    policy_snapshot: PolicySnapshot
    status: RunStatus
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[RunArtifact] = Field(default_factory=list)
    verdict: RunVerdict | None = None

    @model_validator(mode="after")
    def validate_consistency(self) -> "RunRecord":
        if self.run_type != self.run_contract.run_type:
            raise ValueError("run_type must match run_contract.run_type")
        if self.run_type != self.policy_snapshot.run_type:
            raise ValueError("run_type must match policy_snapshot.run_type")
        if self.entrypoint != self.run_contract.entrypoint:
            raise ValueError("entrypoint must match run_contract.entrypoint")
        if self.entrypoint != self.policy_snapshot.allowed_entrypoint:
            raise ValueError("entrypoint must match policy_snapshot.allowed_entrypoint")
        if self.target_ref != self.run_contract.target_ref:
            raise ValueError("target_ref must match run_contract.target_ref")
        if self.run_contract.writeback_allowed and not self.policy_snapshot.writeback_allowed:
            raise ValueError("run_contract.writeback_allowed cannot exceed policy_snapshot.writeback_allowed")
        if self.status == "running" and self.verdict is not None:
            return self
        if self.status in {"finished", "blocked"} and self.verdict is None:
            raise ValueError("verdict is required once a run is finished or blocked")
        return self
