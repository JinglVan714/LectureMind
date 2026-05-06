"""Run LectureMind pipeline over the multi-domain QA sample matrix.

Usage:
    python -m scripts.run_qa_matrix
    python -m scripts.run_qa_matrix --only BV1fyNKz5Egb BV16h9rBDEn9
    python -m scripts.run_qa_matrix --force
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.pipeline import Pipeline
from app.storage.db import Database


@dataclass(frozen=True)
class QASample:
    domain: str
    bv_id: str
    title: str
    url: str
    qa_focus: str
    risk: str


QA_SAMPLES: tuple[QASample, ...] = (
    QASample(
        domain="30 分钟以上 / AI 科研流程",
        bv_id="BV1vPsZzAEbe",
        title="[实用 AI] GPT/Gemini 全流程参与 idea 生成，实验设计，Paper writing | AI scientist | NotebookLM",
        url="https://www.bilibili.com/video/BV1vPsZzAEbe/",
        qa_focus="长流程是否能拆出完整闭环；是否能保留 idea → experiment → writing 的阶段关系",
        risk="长视频，可能触发字幕 token 压力",
    ),
    QASample(
        domain="概念课 / RL + Diffusion",
        bv_id="BV1TeKKeiEci",
        title="【直播回放  Online RL+Diffusion】Diffusion Actor-Critic with Entropy Regulator",
        url="https://www.bilibili.com/video/BV1TeKKeiEci/",
        qa_focus="抽象概念、符号、方法关系是否讲清；是否避免硬造操作步骤",
        risk="直播回放可能噪声高、结构松散",
    ),
    QASample(
        domain="Agent 技术课",
        bv_id="BV1fyNKz5Egb",
        title="[Modern Agent] 11 Codex 初步，系统提示词注入，会话历史管理，Context Engineering，Skills / MCP 位置",
        url="https://www.bilibili.com/video/BV1fyNKz5Egb/",
        qa_focus="Agent 概念层级、系统提示词/历史/Context/Skills/MCP 的关系是否清晰",
        risk="术语密集，需重点看术语速查和主线推进",
    ),
    QASample(
        domain="操作教程 / 饮食",
        bv_id="BV1JV96B5EFJ",
        title="唐山味儿！毛蚶子焖米饭，焖出一锅油滋滋的锅巴饭，绝了～",
        url="https://www.bilibili.com/video/BV1JV96B5EFJ/",
        qa_focus="操作步骤、材料、火候/顺序、成品判断是否能还原",
        risk="非知识讲座，考验 procedural_tutorial 适配",
    ),
    QASample(
        domain="健身科普",
        bv_id="BV16h9rBDEn9",
        title="【健身與健康】EP4. 過度訓練造成的影響 ! 休息是你最好的恢復 !",
        url="https://www.bilibili.com/video/BV16h9rBDEn9/",
        qa_focus="原因-影响-恢复建议是否闭环；是否能处理繁体/口语内容",
        risk="健康建议需避免过度医学化",
    ),
    QASample(
        domain="健身教学",
        bv_id="BV1B2Jgz8ES4",
        title="正确背部训练顺序计划完整版",
        url="https://www.bilibili.com/video/BV1B2Jgz8ES4/",
        qa_focus="动作顺序、训练计划、注意事项是否可执行；视觉证据是否有动作价值",
        risk="画面信息重要，需重点看关键帧利用",
    ),
    QASample(
        domain="医学科普",
        bv_id="BV1VjoeBfEAx",
        title="塑料餐盒+食用油，微波炉加热3分钟，释放大量毒物，危害巨大！",
        url="https://www.bilibili.com/video/BV1VjoeBfEAx/",
        qa_focus="风险链条、实验条件、结论边界是否清楚；是否避免夸大",
        risk="医学/健康内容需关注边界与免责声明式表达",
    ),
)


def _select_samples(only: list[str]) -> list[QASample]:
    if not only:
        return list(QA_SAMPLES)
    wanted = set(only)
    selected = [sample for sample in QA_SAMPLES if sample.bv_id in wanted]
    missing = sorted(wanted - {sample.bv_id for sample in selected})
    if missing:
        raise SystemExit(f"Unknown BV id(s): {', '.join(missing)}")
    return selected


def _default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return get_settings().data_dir / "debug" / f"qa_matrix_{stamp}.md"


def _format_seconds(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    return f"{minutes}m{sec:02d}s"


def _markdown_cell(text: str) -> str:
    return text.replace("\n", " ").replace("|", "\\|")


async def _run_sample(pipeline: Pipeline, sample: QASample, force_refresh: bool) -> dict[str, str]:
    started = time.monotonic()

    async def progress(p: int, msg: str) -> None:
        print(f"  [{sample.bv_id}] [{p:3d}%] {msg}", file=sys.stderr)

    try:
        report = await pipeline.run(sample.url, force_refresh=force_refresh, progress_cb=progress)
        return {
            "status": "ok",
            "duration": _format_seconds(time.monotonic() - started),
            "report": str(report),
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        logging.exception("QA sample failed: %s", sample.bv_id)
        return {
            "status": "failed",
            "duration": _format_seconds(time.monotonic() - started),
            "report": "",
            "error": str(exc).replace("\n", " ")[:500],
        }


def _write_report(path: Path, rows: list[tuple[QASample, dict[str, str]]], force_refresh: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LectureMind 多领域 QA 批量运行结果",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- force_refresh：{force_refresh}",
        "",
        "| 状态 | 领域 | BV | 标题 | QA 重点 | 风险/备注 | 耗时 | 报告 | 错误 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for sample, result in rows:
        report = _markdown_cell(result["report"] or "-")
        error = _markdown_cell(result["error"] or "-")
        lines.append(
            f"| {result['status']} | {_markdown_cell(sample.domain)} | `{sample.bv_id}` | "
            f"{_markdown_cell(sample.title)} | {_markdown_cell(sample.qa_focus)} | "
            f"{_markdown_cell(sample.risk)} | {result['duration']} | {report} | {error} |"
        )
    lines.extend([
        "",
        "## 人工 QA 提示",
        "",
        "- 主线闭环是否清楚",
        "- 类型适配是否自然",
        "- 视觉证据是否有结构价值",
        "- HTML 资源是否可直接打开",
        "- 健身/医学/健康类表达是否有边界",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def _async_main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=get_settings().log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )
    samples = _select_samples(args.only)
    settings = get_settings()
    db = Database(settings.db_path)
    await db.init()
    pipeline = Pipeline(db)
    rows: list[tuple[QASample, dict[str, str]]] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] {sample.bv_id} · {sample.domain}", file=sys.stderr)
        rows.append((sample, await _run_sample(pipeline, sample, args.force)))
    output = args.out or _default_output_path()
    _write_report(output, rows, args.force)
    failed = sum(1 for _, result in rows if result["status"] != "ok")
    print(f"QA report written to {output}")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LectureMind multi-domain QA samples sequentially.")
    parser.add_argument("--only", nargs="+", default=[], help="Only run selected BV ids from the QA matrix.")
    parser.add_argument("--force", action="store_true", help="Ignore cached reports and regenerate each selected sample.")
    parser.add_argument("--out", type=Path, default=None, help="Markdown output path for the batch result.")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
