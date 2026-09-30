import json

from app.runtime.contracts import RunContract, RunRecord, RunVerdict
from app.runtime.observe import (
    create_run_record,
    create_trace_id,
    emit_artifact,
    issue_verdict,
    write_run_record,
)
from app.runtime.policy import build_policy_snapshot
from pydantic import ValidationError
import pytest


def test_runtime_policy_defaults_are_run_type_specific():
    compile_policy = build_policy_snapshot("lecture_compile")
    copilot_policy = build_policy_snapshot("copilot_answer")
    mcp_policy = build_policy_snapshot("mcp_tool_request")

    assert compile_policy.writeback_allowed is True
    assert "commit_summary" in compile_policy.capabilities
    assert copilot_policy.writeback_allowed is False
    assert "call_read_tools" in copilot_policy.capabilities
    assert mcp_policy.allowed_entrypoint == "mcp"


def test_run_record_requires_contract_policy_and_verdict():
    contract = RunContract(
        run_type="copilot_answer",
        entrypoint="copilot_sse",
        target_ref="BV1demo",
        expected_profile=None,
        expected_artifacts=["copilot_answer", "anchor_validation_report"],
        writeback_allowed=False,
        minimum_acceptance=["answer_emitted"],
    )
    verdict = RunVerdict(
        status="accept",
        reasons=["ok"],
        warnings=[],
        next_actions=[],
        evidence_refs=[],
    )
    record = RunRecord(
        trace_id="rt-123",
        run_type="copilot_answer",
        entrypoint="copilot_sse",
        target_ref="BV1demo",
        run_contract=contract,
        policy_snapshot=build_policy_snapshot("copilot_answer"),
        status="finished",
        warnings=[],
        artifacts=[],
        verdict=verdict,
    )

    assert record.trace_id == "rt-123"
    assert record.verdict.status == "accept"


def test_running_run_record_allows_missing_verdict():
    contract = RunContract(
        run_type="lecture_compile",
        entrypoint="pipeline",
        target_ref="BV1demo",
        expected_profile="standard",
        expected_artifacts=["lecture_ir"],
        writeback_allowed=True,
        minimum_acceptance=["report_rendered"],
    )
    record = RunRecord(
        trace_id="rt-456",
        run_type="lecture_compile",
        entrypoint="pipeline",
        target_ref="BV1demo",
        run_contract=contract,
        policy_snapshot=build_policy_snapshot("lecture_compile"),
        status="running",
        warnings=[],
        artifacts=[],
        verdict=None,
    )

    assert record.verdict is None
    assert record.status == "running"


def test_build_policy_snapshot_rejects_unknown_run_type():
    with pytest.raises(ValueError):
        build_policy_snapshot("unknown_run_type")


def test_run_record_rejects_mismatched_contract_and_policy_fields():
    contract = RunContract(
        run_type="copilot_answer",
        entrypoint="copilot_sse",
        target_ref="BV1demo",
        expected_profile=None,
        expected_artifacts=["copilot_answer"],
        writeback_allowed=False,
        minimum_acceptance=["answer_emitted"],
    )
    policy = build_policy_snapshot("copilot_answer")
    verdict = RunVerdict(
        status="accept",
        reasons=["ok"],
        warnings=[],
        next_actions=[],
        evidence_refs=[],
    )

    with pytest.raises(ValidationError):
        RunRecord(
            trace_id="rt-789",
            run_type="mcp_tool_request",
            entrypoint="copilot_sse",
            target_ref="BV1demo",
            run_contract=contract,
            policy_snapshot=policy,
            status="finished",
            warnings=[],
            artifacts=[],
            verdict=verdict,
        )


def test_run_record_rejects_entrypoint_outside_policy_boundary():
    contract = RunContract(
        run_type="copilot_answer",
        entrypoint="mcp",
        target_ref="BV1demo",
        expected_profile=None,
        expected_artifacts=["copilot_answer"],
        writeback_allowed=False,
        minimum_acceptance=["answer_emitted"],
    )
    policy = build_policy_snapshot("copilot_answer")
    verdict = RunVerdict(
        status="accept",
        reasons=["ok"],
        warnings=[],
        next_actions=[],
        evidence_refs=[],
    )

    with pytest.raises(ValidationError):
        RunRecord(
            trace_id="rt-790",
            run_type="copilot_answer",
            entrypoint="mcp",
            target_ref="BV1demo",
            run_contract=contract,
            policy_snapshot=policy,
            status="finished",
            warnings=[],
            artifacts=[],
            verdict=verdict,
        )


def test_run_record_rejects_contract_writeback_wider_than_policy():
    contract = RunContract(
        run_type="copilot_answer",
        entrypoint="copilot_sse",
        target_ref="BV1demo",
        expected_profile=None,
        expected_artifacts=["copilot_answer"],
        writeback_allowed=True,
        minimum_acceptance=["answer_emitted"],
    )
    policy = build_policy_snapshot("copilot_answer")
    verdict = RunVerdict(
        status="accept",
        reasons=["ok"],
        warnings=[],
        next_actions=[],
        evidence_refs=[],
    )

    with pytest.raises(ValidationError):
        RunRecord(
            trace_id="rt-791",
            run_type="copilot_answer",
            entrypoint="copilot_sse",
            target_ref="BV1demo",
            run_contract=contract,
            policy_snapshot=policy,
            status="finished",
            warnings=[],
            artifacts=[],
            verdict=verdict,
        )


def test_create_trace_id_has_runtime_prefix():
    trace_id = create_trace_id()

    assert trace_id.startswith("rt-")
    assert len(trace_id) > 3


def test_create_run_record_starts_running_without_verdict():
    record = create_run_record(
        run_type="lecture_compile",
        entrypoint="pipeline",
        target_ref="BV1demo",
        expected_artifacts=["lecture_ir", "report_html"],
        expected_profile="standard",
        writeback_allowed=True,
        minimum_acceptance=["report_rendered"],
    )

    assert record.status == "running"
    assert record.verdict is None
    assert record.trace_id.startswith("rt-")
    assert record.run_contract.expected_artifacts == ["lecture_ir", "report_html"]
    assert record.policy_snapshot.run_type == "lecture_compile"


def test_write_run_record_persists_debug_json(tmp_path):
    record = create_run_record(
        run_type="lecture_compile",
        entrypoint="pipeline",
        target_ref="BV1demo",
        expected_artifacts=["lecture_ir", "report_html"],
        expected_profile="standard",
        writeback_allowed=True,
        minimum_acceptance=["report_rendered"],
    )
    emit_artifact(
        record,
        artifact_type="lecture_ir",
        path_or_ref="data/debug/BV1demo.lecture_ir.json",
        producer="ir_builder",
    )
    record.verdict = issue_verdict(
        status="accept",
        reasons=["report_rendered"],
        warnings=[],
        next_actions=[],
        evidence_refs=["lecture_ir"],
    )
    record.status = "finished"

    out = write_run_record(record, tmp_path)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert out.exists()
    assert out.parent == tmp_path
    assert out.name.endswith(".run.json")
    assert payload["trace_id"].startswith("rt-")
    assert payload["artifacts"][0]["artifact_type"] == "lecture_ir"
    assert payload["verdict"]["status"] == "accept"
    assert record.artifacts[0].producer == "ir_builder"


def test_write_run_record_rejects_invalid_post_init_state(tmp_path):
    record = create_run_record(
        run_type="lecture_compile",
        entrypoint="pipeline",
        target_ref="BV1demo",
        expected_artifacts=["lecture_ir"],
        expected_profile="standard",
        writeback_allowed=True,
        minimum_acceptance=["report_rendered"],
    )
    record.status = "finished"

    with pytest.raises(ValidationError):
        write_run_record(record, tmp_path)


def test_write_run_record_uses_unique_filenames_per_write(tmp_path):
    record = create_run_record(
        run_type="lecture_compile",
        entrypoint="pipeline",
        target_ref="BV1demo",
        expected_artifacts=["lecture_ir"],
        expected_profile="standard",
        writeback_allowed=True,
        minimum_acceptance=["report_rendered"],
    )
    record.verdict = issue_verdict(
        status="accept",
        reasons=["report_rendered"],
        warnings=[],
        next_actions=[],
        evidence_refs=["lecture_ir"],
    )
    record.status = "finished"

    first = write_run_record(record, tmp_path)
    second = write_run_record(record, tmp_path)

    assert first != second
    assert first.name != second.name
    assert first.exists()
    assert second.exists()
