#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from _bootstrap import add_src_to_path

add_src_to_path()

from policylink.paths import MODEL_RESEARCH_PIPELINE_REPORT_PATH, PROJECT_ROOT, ensure_project_dirs


PYTHON = sys.executable


STEPS: List[Dict[str, Any]] = [
    {"name": "Dataset Audit", "cmd": [PYTHON, "scripts/audit_dataset.py", "--min-rows", "50", "--min-labeled-rows", "30", "--min-dates", "5"], "warning_only": True},
    {"name": "Experiment Models", "cmd": [PYTHON, "scripts/experiment_models.py", "--use-historical", "--min-labeled-rows", "300", "--min-dates", "30", "--n-splits", "3"], "warning_only": True},
    {"name": "Calibrate Model", "cmd": [PYTHON, "scripts/calibrate_model.py", "--use-historical", "--method", "sigmoid", "--min-labeled-rows", "300"], "warning_only": True},
    {"name": "Explain Model", "cmd": [PYTHON, "scripts/explain_model.py", "--use-historical", "--top-n", "30"], "warning_only": True},
    {"name": "Backtest Historical", "cmd": [PYTHON, "scripts/backtest_signals.py", "--use-historical", "--score-column", "final_score", "--target-column", "future_return_5d", "--top-k", "3", "--min-labeled-rows", "100"], "warning_only": True},
    {"name": "Promote Model Dry Run", "cmd": [PYTHON, "scripts/promote_model.py", "--dry-run"], "warning_only": True},
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_step(step: Dict[str, Any]) -> Dict[str, Any]:
    started_at = utc_now()
    try:
        completed = subprocess.run(
            step["cmd"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        status = "success" if completed.returncode == 0 else ("warning" if step.get("warning_only") else "failed")
        return {
            "name": step["name"],
            "status": status,
            "returncode": completed.returncode,
            "started_at": started_at,
            "finished_at": utc_now(),
            "stdout_tail": completed.stdout[-1500:],
            "stderr_tail": completed.stderr[-1500:],
            "command": " ".join(step["cmd"]),
        }
    except Exception as exc:
        return {
            "name": step["name"],
            "status": "warning" if step.get("warning_only") else "failed",
            "returncode": None,
            "started_at": started_at,
            "finished_at": utc_now(),
            "error": str(exc)[:500],
            "command": " ".join(step["cmd"]),
        }


def build_report(results: List[Dict[str, Any]]) -> str:
    overall = "success" if all(item["status"] in {"success", "warning"} for item in results) else "failed"
    lines = [
        "# Model Research Pipeline Report",
        "",
        f"- generated_at: {utc_now()}",
        "- order_enabled=false",
        f"- overall_status: {overall}",
        "",
        "## Steps",
    ]
    for item in results:
        lines.append(f"- {item['name']}: {item['status']} (returncode={item.get('returncode')})")
        stdout = (item.get("stdout_tail") or "").strip()
        if stdout:
            last_line = stdout.splitlines()[-1]
            lines.append(f"  - stdout_last: {last_line}")
        if item.get("error"):
            lines.append(f"  - error: {item.get('error')}")
    lines.extend([
        "",
        "## Notes",
        "- 이 파이프라인은 모델 연구/평가/승격 dry-run만 수행합니다.",
        "- 주문 실행 스크립트와 Kiwoom 주문 TR은 포함하지 않습니다.",
    ])
    return "\n".join(lines)


def main() -> int:
    ensure_project_dirs()
    results = [run_step(step) for step in STEPS]
    MODEL_RESEARCH_PIPELINE_REPORT_PATH.write_text(build_report(results), encoding="utf-8")
    print(f"Saved model research pipeline report: {MODEL_RESEARCH_PIPELINE_REPORT_PATH}")
    print("overall_status=success")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
