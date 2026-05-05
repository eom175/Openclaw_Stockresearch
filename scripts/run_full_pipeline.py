#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from _bootstrap import add_src_to_path

add_src_to_path()

from policylink.paths import FULL_PIPELINE_REPORT_PATH, PROJECT_ROOT, REPORTS_DIR


SENSITIVE_TEXT_PATTERNS = [
    (re.compile(r"Bearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE), "Bearer [REDACTED]"),
    (re.compile(r"(?i)(token|secret|app[_-]?key|api[_-]?key|crtfc_key|authorization|password)\s*[:=]\s*[^\s,}&]+"), r"\1=[REDACTED]"),
    (re.compile(r"(?i)(crtfc_key=)[^&\s]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(account|acct|acnt|계좌)\s*[:=]\s*[^\s,}]+"), r"\1=[REDACTED]"),
    (re.compile(r"(?<!\d)\d{8,}(?!\d)"), "[REDACTED_NUMBER]"),
]


@dataclass
class PipelineStep:
    name: str
    argv: List[str]
    warning_only: bool = False


def redact_text(text: str, limit: int = 4000) -> str:
    redacted = text[-limit:]
    for pattern, replacement in SENSITIVE_TEXT_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted.strip()


def command_label(argv: List[str]) -> str:
    labels = []
    for item in argv:
        try:
            labels.append(str(Path(item).relative_to(PROJECT_ROOT)))
        except Exception:
            labels.append(str(item))
    return " ".join(labels)


def run_step(step: PipelineStep) -> dict:
    started = time.time()
    completed = subprocess.run(
        step.argv,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    duration = time.time() - started

    result = {
        "name": step.name,
        "command": command_label(step.argv),
        "return_code": completed.returncode,
        "duration_seconds": round(duration, 1),
        "status": "success" if completed.returncode == 0 else ("warning" if step.warning_only else "failed"),
        "stdout_tail": "",
        "stderr_tail": "",
    }

    if completed.returncode != 0:
        result["stdout_tail"] = redact_text(completed.stdout)
        result["stderr_tail"] = redact_text(completed.stderr)

    return result


def build_report(results: List[dict]) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    overall = "success" if all(item["status"] in {"success", "warning"} for item in results) else "failed"

    lines = [
        "# Full Pipeline Report",
        "",
        f"- generated_at: {generated_at}",
        f"- overall_status: {overall}",
        "- order_scripts_included: no",
        "",
        "## Steps",
        "",
    ]

    for idx, item in enumerate(results, start=1):
        lines.extend([
            f"### {idx}. {item['name']}",
            "",
            f"- status: {item['status']}",
            f"- return_code: {item['return_code']}",
            f"- duration_seconds: {item['duration_seconds']}",
            f"- command: `{item['command']}`",
        ])
        if item["status"] != "success":
            if item["stdout_tail"]:
                lines.extend(["", "stdout_tail:", "```text", item["stdout_tail"], "```"])
            if item["stderr_tail"]:
                lines.extend(["", "stderr_tail:", "```text", item["stderr_tail"], "```"])
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    python = sys.executable
    script_dir = Path(__file__).resolve().parent

    steps = [
        PipelineStep("Research - Collect 30 Days", [python, str(script_dir / "collect_research.py"), "--hours", "720", "--max-items", "50"]),
        PipelineStep("DART - Collect Disclosures", [python, str(script_dir / "collect_dart.py"), "--days", "90", "--max-stocks", "10", "--sleep", "0.3"], warning_only=True),
        PipelineStep("News - Collect Naver", [python, str(script_dir / "collect_naver_news.py"), "--days", "14", "--max-stocks", "10", "--display", "20", "--sort", "date", "--sleep", "0.3"], warning_only=True),
        PipelineStep("Yahoo - Collect Global Features", [python, str(script_dir / "collect_yahoo_finance.py"), "--days", "90", "--news-count", "10", "--max-tickers", "14", "--sleep", "0.3"], warning_only=True),
        PipelineStep("Kiwoom - Daily Mock Account Report", [python, str(script_dir / "kiwoom_daily_report.py")]),
        PipelineStep("Prices - Sync Daily", [python, str(script_dir / "sync_prices.py"), "--max-stocks", "10", "--sleep", "0.7"]),
        PipelineStep("Flows - Sync Investor Flows", [python, str(script_dir / "sync_flows.py"), "--max-stocks", "5", "--sleep", "1.2"]),
        PipelineStep("Portfolio - Recommend", [python, str(script_dir / "recommend_portfolio.py")]),
        PipelineStep("Dataset - Build Snapshot", [python, str(script_dir / "build_dataset.py")]),
        PipelineStep("Dataset - Label Future Returns", [python, str(script_dir / "label_dataset.py")]),
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for step in steps:
        print(f"[pipeline] running: {step.name}", flush=True)
        result = run_step(step)
        results.append(result)
        print(f"[pipeline] {result['status']}: {step.name}", flush=True)

    FULL_PIPELINE_REPORT_PATH.write_text(build_report(results), encoding="utf-8")
    print(f"[pipeline] report: {FULL_PIPELINE_REPORT_PATH}", flush=True)

    return 0 if all(item["status"] in {"success", "warning"} for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
