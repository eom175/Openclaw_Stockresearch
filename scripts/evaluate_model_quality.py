#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

from _bootstrap import add_src_to_path

add_src_to_path()

from policylink.paths import (
    BACKTEST_METRICS_PATH,
    DATASET_AUDIT_JSON_PATH,
    MODEL_QUALITY_METRICS_PATH,
    MODEL_QUALITY_REPORT_PATH,
    MODEL_REGISTRY_PATH,
    MODEL_TRAINING_METRICS_PATH,
)
from policylink.utils import load_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_quality() -> Dict[str, Any]:
    audit = load_json(DATASET_AUDIT_JSON_PATH, {})
    training = load_json(MODEL_TRAINING_METRICS_PATH, {})
    registry = load_json(MODEL_REGISTRY_PATH, {})
    backtest = load_json(BACKTEST_METRICS_PATH, {})

    model_status = registry.get("model_status") or training.get("model_registry_status") or "no_model"
    quality_gate = "not_ready"
    reasons = []

    if audit.get("warning_level") == "BLOCK_TRAINING":
        reasons.append("dataset audit BLOCK_TRAINING")
    if training.get("training_status") != "trained":
        reasons.append("model is not trained")
    if backtest.get("backtest_status") != "completed":
        reasons.append("backtest is not completed")

    if not reasons:
        class_metrics = (training.get("metrics") or {}).get("outperform_5d", {})
        precision_at_3 = class_metrics.get("precision_at_3")
        mean_return_top3 = class_metrics.get("mean_future_return_top3")
        if precision_at_3 is not None and precision_at_3 >= 0.55 and (mean_return_top3 or 0) > 0:
            quality_gate = "candidate_for_review"
        else:
            quality_gate = "weak_model_review_only"
            reasons.append("validation ranking metrics are not strong enough")

    return {
        "generated_at": utc_now(),
        "mode": "model_quality_evaluation",
        "model_status": model_status,
        "quality_gate": quality_gate,
        "reasons": reasons,
        "audit_warning_level": audit.get("warning_level"),
        "training_status": training.get("training_status"),
        "backtest_status": backtest.get("backtest_status"),
        "best_threshold": registry.get("best_threshold"),
        "order_enabled": False,
    }


def build_report(result: Dict[str, Any]) -> str:
    lines = [
        "# Model Quality Report",
        "",
        f"- generated_at: {result.get('generated_at')}",
        "- 실제 주문 실행: 비활성화",
        "- order_enabled=false",
        f"- model_status: {result.get('model_status')}",
        f"- quality_gate: {result.get('quality_gate')}",
        f"- audit_warning_level: {result.get('audit_warning_level')}",
        f"- training_status: {result.get('training_status')}",
        f"- backtest_status: {result.get('backtest_status')}",
        f"- best_threshold: {result.get('best_threshold')}",
        "",
        "## 판단",
    ]
    if result.get("reasons"):
        for reason in result["reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("- 모델 품질은 검토 후보 수준입니다.")
    lines.append("- ML 모델은 주문 실행자가 아니라 scoring engine입니다.")
    return "\n".join(lines)


def main() -> int:
    result = evaluate_quality()
    save_json(MODEL_QUALITY_METRICS_PATH, result)
    MODEL_QUALITY_REPORT_PATH.write_text(build_report(result), encoding="utf-8")
    print(f"Saved model quality metrics: {MODEL_QUALITY_METRICS_PATH}")
    print(f"Saved model quality report: {MODEL_QUALITY_REPORT_PATH}")
    print(f"quality_gate={result.get('quality_gate')}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
