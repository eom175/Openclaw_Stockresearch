#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

from _bootstrap import add_src_to_path

add_src_to_path()

from train_model import ml_dependency_check
from policylink.paths import ML_DEPENDENCY_CHECK_JSON_PATH, ML_DEPENDENCY_CHECK_MD_PATH, ensure_project_dirs


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_report(payload: Dict[str, Any]) -> str:
    lines = [
        "# ML Dependency Check",
        "",
        f"- generated_at: {payload.get('generated_at')}",
        f"- status: {payload.get('status')}",
        "- order_enabled=false",
        "",
        "## Packages",
        "| package | required | available | version | error |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in payload.get("dependency_details", []):
        error = (item.get("error") or "").replace("|", "/")
        lines.append(
            f"| {item.get('name')} | {item.get('required')} | {item.get('available')} | "
            f"{item.get('version')} | {error} |"
        )

    lines.extend([
        "",
        "## Install Guide",
        f"- pip: `{payload.get('install_command')}`",
    ])
    if payload.get("system_install_suggestions"):
        for suggestion in payload.get("system_install_suggestions", []):
            lines.append(f"- system: `{suggestion}`")
    if payload.get("optional_missing_dependencies"):
        lines.append(f"- optional_missing_dependencies: {payload.get('optional_missing_dependencies')}")
    if not payload.get("missing_dependencies"):
        lines.append("- required dependency는 모두 사용 가능합니다.")
    else:
        lines.append(f"- missing_dependencies: {payload.get('missing_dependencies')}")
    lines.extend([
        "",
        "## Notes",
        "- shap은 설명성 분석용 optional dependency입니다.",
        "- 이 점검은 .env를 읽지 않으며 주문 관련 API를 호출하지 않습니다.",
    ])
    return "\n".join(lines)


def main() -> int:
    ensure_project_dirs()
    check = ml_dependency_check()
    payload = {
        "generated_at": utc_now(),
        "mode": "ml_dependency_check",
        "status": "ok" if check.get("ok") else "missing_required_dependency",
        "order_enabled": False,
        **check,
    }
    save_json(ML_DEPENDENCY_CHECK_JSON_PATH, payload)
    ML_DEPENDENCY_CHECK_MD_PATH.write_text(build_report(payload), encoding="utf-8")
    print(f"Saved dependency check json: {ML_DEPENDENCY_CHECK_JSON_PATH}")
    print(f"Saved dependency check report: {ML_DEPENDENCY_CHECK_MD_PATH}")
    print(f"status={payload.get('status')}")
    if payload.get("missing_dependencies"):
        print(f"missing_dependencies={','.join(payload.get('missing_dependencies'))}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
