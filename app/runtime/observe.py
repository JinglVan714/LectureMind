from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.runtime.contracts import (
    RunArtifact,
    RunContract,
    RunRecord,
    RunType,
    RunVerdict,
    RunVerdictStatus,
)
from app.runtime.policy import build_policy_snapshot


def create_trace_id() -> str:
    return f"rt-{uuid4().hex[:12]}"


def create_run_record(
    *,
    run_type: RunType,
    entrypoint: str,
    target_ref: str,
    expected_artifacts: list[str],
    expected_profile: str | None,
    writeback_allowed: bool,
    minimum_acceptance: list[str],
) -> RunRecord:
    contract = RunContract(
        run_type=run_type,
        entrypoint=entrypoint,
        target_ref=target_ref,
        expected_profile=expected_profile,
        expected_artifacts=expected_artifacts,
        writeback_allowed=writeback_allowed,
        minimum_acceptance=minimum_acceptance,
    )
    return RunRecord(
        trace_id=create_trace_id(),
        run_type=run_type,
        entrypoint=entrypoint,
        target_ref=target_ref,
        run_contract=contract,
        policy_snapshot=build_policy_snapshot(run_type),
        status="running",
    )


def emit_artifact(
    record: RunRecord,
    *,
    artifact_type: str,
    path_or_ref: str,
    producer: str,
) -> None:
    record.artifacts.append(
        RunArtifact(
            artifact_type=artifact_type,
            path_or_ref=path_or_ref,
            producer=producer,
        )
    )


def issue_verdict(
    *,
    status: RunVerdictStatus,
    reasons: list[str],
    warnings: list[str],
    next_actions: list[str],
    evidence_refs: list[str],
) -> RunVerdict:
    return RunVerdict(
        status=status,
        reasons=reasons,
        warnings=warnings,
        next_actions=next_actions,
        evidence_refs=evidence_refs,
    )


def write_run_record(record: RunRecord, debug_root: Path) -> Path:
    debug_root.mkdir(parents=True, exist_ok=True)
    validated_record = RunRecord.model_validate(record.model_dump(mode="json"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = debug_root / f"{validated_record.trace_id}.{validated_record.run_type}.{stamp}.run.json"
    duplicate_index = 1
    while out.exists():
        out = (
            debug_root
            / f"{validated_record.trace_id}.{validated_record.run_type}.{stamp}.{duplicate_index}.run.json"
        )
        duplicate_index += 1
    out.write_text(
        json.dumps(validated_record.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out
