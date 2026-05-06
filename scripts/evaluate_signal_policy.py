#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from _bootstrap import add_src_to_path

add_src_to_path()

from policylink.paths import (
    BACKTEST_METRICS_PATH,
    ML_SIGNALS_JSON_PATH,
    MODEL_REGISTRY_PATH,
    MODEL_TRAINING_METRICS_PATH,
    SIGNAL_POLICY_JSON_PATH,
    SIGNAL_POLICY_REPORT_PATH,
    ensure_project_dirs,
)
from policylink.utils import load_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def metric(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key)
    try:
        return float(value)
    except Exception:
        return default


def evaluate_policy(backtest: Dict[str, Any], training: Dict[str, Any], registry: Dict[str, Any], ml_signals: Dict[str, Any]) -> Dict[str, Any]:
    full_rule = backtest.get("metrics") or {}
    ml = backtest.get("ml_walk_forward_metrics") or {}
    same_rule = ml.get("same_window_rule_metrics") or {}
    ensemble = ml.get("same_window_ensemble_metrics") or {}

    blockers: List[str] = []
    warnings: List[str] = []
    ml_selected_rows = int(ml.get("selected_rows") or 0)
    ml_date_count = int(ml.get("date_count") or 0)
    ensemble_selected_rows = int(ensemble.get("selected_rows") or 0)
    ensemble_date_count = int(ensemble.get("date_count") or 0)
    calibrated = bool(ml_signals.get("calibrated", False))
    model_source = str(ml_signals.get("model_source") or "unknown")

    if ml.get("status") != "completed":
        blockers.append("ml_walk_forward_unavailable")
    if ml_selected_rows < 100:
        blockers.append("ml_selected_rows_too_small")
    if ml_date_count < 30:
        blockers.append("ml_validation_dates_too_small")
    if not calibrated:
        blockers.append("calibration_missing")
    if model_source == "fallback":
        warnings.append("fallback_model")
    if metric(ml, "excess_net_return") <= metric(same_rule, "excess_net_return"):
        blockers.append("ml_not_superior_on_excess_return")

    ensemble_beats_rule = metric(ensemble, "excess_net_return") > metric(same_rule, "excess_net_return")
    ensemble_beats_ml = metric(ensemble, "excess_net_return") > metric(ml, "excess_net_return")
    ensemble_sample_ok = ensemble_selected_rows >= 100 and ensemble_date_count >= 30

    recommended_policy = "rule_first_ml_modifier"
    primary_signal = "rule"
    rule_weight = 0.75
    ml_weight = 0.25
    reason = "ML validation sample is small, uncalibrated, or not superior to the rule baseline."

    if ensemble_beats_rule and ensemble_sample_ok:
        recommended_policy = "ensemble_candidate"
        primary_signal = "ensemble"
        rule_weight = 0.60 if calibrated and model_source == "active" else 0.75
        ml_weight = 0.40 if calibrated and model_source == "active" else 0.25
        reason = "Ensemble beats same-window rule baseline with enough validation samples."
    if ensemble_beats_rule and ensemble_beats_ml and ensemble_sample_ok and calibrated:
        recommended_policy = "ensemble_primary_candidate"
        primary_signal = "ensemble"
        rule_weight = 0.60
        ml_weight = 0.40
        reason = "Calibrated ensemble is superior on same-window excess net return."

    payload = {
        "generated_at": utc_now(),
        "mode": "signal_policy_evaluation",
        "recommended_policy": recommended_policy,
        "primary_signal": primary_signal,
        "rule_weight": rule_weight,
        "ml_weight": ml_weight,
        "auto_order_allowed": False,
        "reason": reason,
        "promotion_blockers": blockers,
        "warnings": warnings,
        "model_source": model_source,
        "calibrated": calibrated,
        "metrics": {
            "full_period_rule": {
                "selected_rows": full_rule.get("selected_rows"),
                "date_count": full_rule.get("date_count"),
                "top_k_mean_net_return": full_rule.get("top_k_mean_net_return"),
                "excess_net_return": full_rule.get("excess_net_return"),
            },
            "same_window_rule": same_rule,
            "same_window_ml": ml,
            "same_window_ensemble": ensemble,
        },
        "training_status": training.get("training_status"),
        "model_status": registry.get("model_status"),
        "order_enabled": False,
    }
    return payload


def update_registry(policy: Dict[str, Any]) -> None:
    registry = load_json(MODEL_REGISTRY_PATH, {})
    registry["generated_at"] = utc_now()
    registry["signal_policy"] = {
        "primary_signal": policy.get("primary_signal", "rule"),
        "recommended_policy": policy.get("recommended_policy", "rule_first_ml_modifier"),
        "reason": policy.get("reason"),
        "ml_weight": policy.get("ml_weight", 0.25),
        "rule_weight": policy.get("rule_weight", 0.75),
        "promotion_blockers": policy.get("promotion_blockers", []),
        "auto_order_allowed": False,
    }
    save_json(MODEL_REGISTRY_PATH, registry)


def build_report(policy: Dict[str, Any]) -> str:
    lines = [
        "# Signal Policy Report",
        "",
        f"- generated_at: {policy.get('generated_at')}",
        "- order_enabled=false",
        "- auto_order_allowed=false",
        f"- recommended_policy: {policy.get('recommended_policy')}",
        f"- primary_signal: {policy.get('primary_signal')}",
        f"- rule_weight: {policy.get('rule_weight')}",
        f"- ml_weight: {policy.get('ml_weight')}",
        f"- calibrated: {policy.get('calibrated')}",
        f"- model_source: {policy.get('model_source')}",
        f"- reason: {policy.get('reason')}",
        "",
        "## Promotion Blockers",
    ]
    blockers = policy.get("promotion_blockers") or []
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- none")

    metrics = policy.get("metrics") or {}
    same_rule = metrics.get("same_window_rule") or {}
    ml = metrics.get("same_window_ml") or {}
    ensemble = metrics.get("same_window_ensemble") or {}
    lines.extend([
        "",
        "## Same-window Metrics",
        "| signal | selected_rows | date_count | top_k_net | excess_net | precision_at_k | hit_rate | drawdown | stability |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for label, item in [("rule", same_rule), ("ml", ml), ("ensemble", ensemble)]:
        lines.append(
            f"| {label} | {item.get('selected_rows')} | {item.get('date_count')} | "
            f"{item.get('top_k_mean_net_return')} | {item.get('excess_net_return')} | "
            f"{item.get('precision_at_k')} | {item.get('top_k_hit_rate')} | "
            f"{item.get('top_k_avg_drawdown')} | {item.get('stability_score')} |"
        )
    lines.extend([
        "",
        "## Decision",
        "- ML은 현재 primary signal이 아니라 rule-first scoring의 보조 modifier로 사용합니다.",
        "- 자동주문은 비활성화 상태를 유지합니다.",
    ])
    return "\n".join(lines)


def main() -> int:
    ensure_project_dirs()
    backtest = load_json(BACKTEST_METRICS_PATH, {})
    training = load_json(MODEL_TRAINING_METRICS_PATH, {})
    registry = load_json(MODEL_REGISTRY_PATH, {})
    ml_signals = load_json(ML_SIGNALS_JSON_PATH, {})
    policy = evaluate_policy(backtest, training, registry, ml_signals)
    save_json(SIGNAL_POLICY_JSON_PATH, policy)
    SIGNAL_POLICY_REPORT_PATH.write_text(build_report(policy), encoding="utf-8")
    update_registry(policy)
    print(f"Saved signal policy json: {SIGNAL_POLICY_JSON_PATH}")
    print(f"Saved signal policy report: {SIGNAL_POLICY_REPORT_PATH}")
    print(f"recommended_policy={policy.get('recommended_policy')}")
    print(f"primary_signal={policy.get('primary_signal')}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
