from app.runtime.contracts import PolicySnapshot, RunType


def build_policy_snapshot(run_type: RunType) -> PolicySnapshot:
    if run_type == "lecture_compile":
        return PolicySnapshot(
            run_type="lecture_compile",
            allowed_entrypoint="pipeline",
            sandbox_level="S1",
            writeback_allowed=True,
            capabilities=[
                "read_lecture_assets",
                "emit_artifacts",
                "commit_summary",
                "commit_report",
                "commit_rag_index",
            ],
        )
    if run_type == "copilot_answer":
        return PolicySnapshot(
            run_type="copilot_answer",
            allowed_entrypoint="copilot_sse",
            sandbox_level="S0",
            writeback_allowed=False,
            capabilities=[
                "read_lecture_assets",
                "call_read_tools",
                "emit_artifacts",
            ],
        )
    if run_type == "mcp_tool_request":
        return PolicySnapshot(
            run_type="mcp_tool_request",
            allowed_entrypoint="mcp",
            sandbox_level="S0",
            writeback_allowed=False,
            capabilities=[
                "read_lecture_assets",
                "call_read_tools",
                "emit_artifacts",
            ],
        )
    raise ValueError(f"Unsupported run_type: {run_type}")
