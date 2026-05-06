#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from _bootstrap import add_src_to_path

add_src_to_path()

from policylink.paths import (
    ACTIVE_FEATURE_COLUMNS_PATH,
    ACTIVE_CLASSIFIER_JOBLIB_PATH,
    ACTIVE_MODELS_DIR,
    ACTIVE_XGB_OUTPERFORM_MODEL_PATH,
    BACKTEST_METRICS_PATH,
    CALIBRATION_REPORT_JSON_PATH,
    EXPERIMENT_RESULTS_PATH,
    HISTORICAL_DATASET_AUDIT_JSON_PATH,
    MODEL_PROMOTION_REPORT_PATH,
    MODEL_REGISTRY_PATH,
    PAPER_TRADING_REPORT_JSON_PATH,
    SIGNAL_POLICY_JSON_PATH,
    ensure_project_dirs,
)
from policylink.utils import load_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def find_best_experiment(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    best_id = (results.get("best_experiment") or {}).get("experiment_id")
    if not best_id:
        return None
    for item in results.get("experiments", []):
        if item.get("experiment_id") == best_id:
            return item
    return None


def promotion_decision(
    best: Optional[Dict[str, Any]],
    args: argparse.Namespace,
    audit: Dict[str, Any],
    backtest: Dict[str, Any],
    signal_policy: Dict[str, Any],
    calibration: Dict[str, Any],
    paper_trading: Dict[str, Any],
) -> Dict[str, Any]:
    reasons = []
    if not best:
        reasons.append("best experiment가 없습니다.")
    metrics = (best or {}).get("metrics_mean") or {}
    ml = backtest.get("ml_walk_forward_metrics") or {}
    same_rule = ml.get("same_window_rule_metrics") or {}
    ensemble = ml.get("same_window_ensemble_metrics") or {}
    calibration_improved = calibration.get("improved") is True
    ml_weight = float(signal_policy.get("ml_weight", 0.25) or 0.25)
    paper_metrics = paper_trading.get("metrics") if isinstance(paper_trading.get("metrics"), dict) else {}

    if best and best.get("status") != "completed":
        reasons.append("best experiment status가 completed가 아닙니다.")
    if best and (metrics.get("precision_at_3") or 0.0) < args.min_precision_at_3:
        reasons.append(f"precision_at_3 {metrics.get('precision_at_3')} < {args.min_precision_at_3}")
    if best and (metrics.get("top3_mean_future_return_5d") or 0.0) <= args.min_top3_mean_return:
        reasons.append(f"top3_mean_future_return_5d {metrics.get('top3_mean_future_return_5d')} <= {args.min_top3_mean_return}")
    if best and args.require_positive_excess_return and (metrics.get("excess_return_top3_vs_all") or 0.0) <= 0:
        reasons.append(f"excess_return_top3_vs_all {metrics.get('excess_return_top3_vs_all')} <= 0")
    drawdown = metrics.get("top3_avg_future_max_drawdown_5d")
    if best and drawdown is not None and drawdown < args.max_drawdown_top3:
        reasons.append(f"top3_avg_future_max_drawdown_5d {drawdown} < {args.max_drawdown_top3}")
    if best and (metrics.get("fold_count") or 0) < 3:
        reasons.append(f"fold_count {metrics.get('fold_count')} < 3")
    if audit.get("audit_status") == "BLOCK_TRAINING" and audit.get("leakage_candidate_columns"):
        reasons.append("historical dataset audit가 leakage 의심 컬럼으로 BLOCK_TRAINING 상태입니다.")
    if (ensemble.get("excess_net_return") or 0.0) <= (same_rule.get("excess_net_return") or 0.0):
        reasons.append("same-window ensemble excess_net_return이 rule excess_net_return보다 높지 않습니다.")
    if (ensemble.get("top_k_hit_rate") or 0.0) < (same_rule.get("top_k_hit_rate") or 0.0):
        reasons.append("same-window ensemble top_k_hit_rate가 rule보다 낮습니다.")
    if (ensemble.get("selected_rows") or 0) < 100:
        reasons.append(f"ensemble selected_rows {ensemble.get('selected_rows')} < 100")
    if (ensemble.get("date_count") or 0) < 30:
        reasons.append(f"ensemble validation date_count {ensemble.get('date_count')} < 30")
    if (ensemble.get("stability_score") or 0.0) <= 0:
        reasons.append(f"ensemble stability_score {ensemble.get('stability_score')} <= 0")
    ensemble_drawdown = ensemble.get("top_k_avg_drawdown")
    if ensemble_drawdown is not None and ensemble_drawdown < args.max_drawdown_top3:
        reasons.append(f"ensemble top_k_avg_drawdown {ensemble_drawdown} < {args.max_drawdown_top3}")
    if not calibration_improved and ml_weight > 0.25:
        reasons.append("calibration이 개선되지 않은 상태에서 ML weight가 0.25를 초과합니다.")
    if signal_policy.get("promotion_blockers"):
        for item in signal_policy.get("promotion_blockers", []):
            if item == "calibration_missing" and calibration_improved:
                continue
            reasons.append(f"signal_policy blocker: {item}")
    if (paper_metrics.get("labeled_trades") or 0) < 30:
        reasons.append(f"paper_trading labeled_trades {paper_metrics.get('labeled_trades')} < 30")
    if (paper_metrics.get("mean_net_return_5d") or 0.0) <= 0:
        reasons.append(f"paper_trading mean_net_return_5d {paper_metrics.get('mean_net_return_5d')} <= 0")
    if (paper_metrics.get("hit_rate_5d") or 0.0) < 0.55:
        reasons.append(f"paper_trading hit_rate_5d {paper_metrics.get('hit_rate_5d')} < 0.55")
    drawdown = paper_metrics.get("avg_max_drawdown_5d")
    if drawdown is None or drawdown <= -0.05:
        reasons.append(f"paper_trading avg_max_drawdown_5d {drawdown} <= -0.05")
    return {"promoted": not reasons, "reasons": reasons}


def copy_active_artifacts(best: Dict[str, Any]) -> Dict[str, str]:
    paths = best.get("artifact_paths") or {}
    model_path = Path(paths.get("model_path") or "")
    feature_path = Path(paths.get("feature_columns_path") or "")
    if not model_path.exists() or not feature_path.exists():
        raise FileNotFoundError("best experiment artifact가 없습니다.")

    ACTIVE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if model_path.suffix == ".json":
        target_model = ACTIVE_XGB_OUTPERFORM_MODEL_PATH
    else:
        target_model = ACTIVE_CLASSIFIER_JOBLIB_PATH
    shutil.copy2(model_path, target_model)
    shutil.copy2(feature_path, ACTIVE_FEATURE_COLUMNS_PATH)
    return {"active_model_path": str(target_model), "active_feature_columns_path": str(ACTIVE_FEATURE_COLUMNS_PATH)}


def update_registry(best: Dict[str, Any], copied: Dict[str, str], calibration: Dict[str, Any]) -> None:
    registry = load_json(MODEL_REGISTRY_PATH, {})
    registry.update({
        "generated_at": utc_now(),
        "model_status": "active",
        "active_model_version": best.get("experiment_id"),
        "active_experiment_id": best.get("experiment_id"),
        "best_threshold": best.get("best_threshold", 0.60),
        "active_model": copied,
        "active_metrics": best.get("metrics_mean", {}),
        "latest_promotion_status": "promoted",
        "calibration": {
            "status": calibration.get("calibration_status"),
            "saved_model_path": calibration.get("saved_model_path"),
            "improved": calibration.get("improved"),
        },
        "order_enabled": False,
    })
    save_json(MODEL_REGISTRY_PATH, registry)


def promote(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_project_dirs()
    results = load_json(EXPERIMENT_RESULTS_PATH, {})
    calibration = load_json(CALIBRATION_REPORT_JSON_PATH, {})
    audit = load_json(HISTORICAL_DATASET_AUDIT_JSON_PATH, {})
    backtest = load_json(BACKTEST_METRICS_PATH, {})
    signal_policy = load_json(SIGNAL_POLICY_JSON_PATH, {})
    paper_trading = load_json(PAPER_TRADING_REPORT_JSON_PATH, {})
    best = find_best_experiment(results)
    decision = promotion_decision(best, args, audit, backtest, signal_policy, calibration, paper_trading)
    copied: Dict[str, str] = {}
    promoted = bool(decision["promoted"])
    if promoted and not args.dry_run:
        copied = copy_active_artifacts(best or {})
        update_registry(best or {}, copied, calibration)
    return {
        "generated_at": utc_now(),
        "mode": "model_promotion",
        "dry_run": bool(args.dry_run),
        "promotion_status": (
            "promoted"
            if promoted and not args.dry_run
            else ("would_promote" if promoted else ("dry_run_rejected" if args.dry_run else "rejected"))
        ),
        "promoted": promoted and not args.dry_run,
        "best_experiment": best,
        "same_window_metrics": {
            "rule": ((backtest.get("ml_walk_forward_metrics") or {}).get("same_window_rule_metrics") or {}),
            "ml": backtest.get("ml_walk_forward_metrics") or {},
            "ensemble": ((backtest.get("ml_walk_forward_metrics") or {}).get("same_window_ensemble_metrics") or {}),
        },
        "recommended_signal_policy": signal_policy,
        "paper_trading_metrics": paper_trading.get("metrics", {}),
        "reasons": decision["reasons"],
        "active_paths": copied,
        "criteria": {
            "min_precision_at_3": args.min_precision_at_3,
            "min_top3_mean_return": args.min_top3_mean_return,
            "max_drawdown_top3": args.max_drawdown_top3,
            "require_positive_excess_return": args.require_positive_excess_return,
            "min_paper_labeled_trades": 30,
            "min_paper_hit_rate_5d": 0.55,
            "min_paper_mean_net_return_5d": 0.0,
            "min_paper_avg_max_drawdown_5d": -0.05,
        },
        "order_enabled": False,
    }


def build_report(result: Dict[str, Any]) -> str:
    lines = [
        "# Model Promotion Report",
        "",
        f"- generated_at: {result.get('generated_at')}",
        "- order_enabled=false",
        f"- dry_run: {result.get('dry_run')}",
        f"- promotion_status: {result.get('promotion_status')}",
        f"- promoted: {result.get('promoted')}",
        "",
        "## Decision Reasons",
    ]
    reasons = result.get("reasons") or []
    if reasons:
        for reason in reasons:
            lines.append(f"- {reason}")
    else:
        lines.append("- 기준 충족")
    best = result.get("best_experiment") or {}
    same_window = result.get("same_window_metrics") or {}
    lines.extend(["", "## Best Experiment Metrics"])
    if best:
        lines.append(f"- experiment_id: {best.get('experiment_id')}")
        lines.append(f"- best_threshold: {best.get('best_threshold')}")
        for key, value in (best.get("metrics_mean") or {}).items():
            if key in {"precision_at_3", "top3_mean_future_return_5d", "excess_return_top3_vs_all", "top3_avg_future_max_drawdown_5d", "fold_count"}:
                lines.append(f"- {key}: {value}")
    else:
        lines.append("- best experiment 없음")
    lines.extend([
        "",
        "## Same-window Metrics Comparison",
        "| signal | selected_rows | date_count | excess_net | hit_rate | drawdown | stability |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for label in ["rule", "ml", "ensemble"]:
        item = same_window.get(label) or {}
        lines.append(
            f"| {label} | {item.get('selected_rows')} | {item.get('date_count')} | "
            f"{item.get('excess_net_return')} | {item.get('top_k_hit_rate')} | "
            f"{item.get('top_k_avg_drawdown')} | {item.get('stability_score')} |"
        )
    policy = result.get("recommended_signal_policy") or {}
    paper = result.get("paper_trading_metrics") or {}
    lines.extend([
        "",
        "## Recommended Signal Policy",
        f"- recommended_policy: {policy.get('recommended_policy')}",
        f"- primary_signal: {policy.get('primary_signal')}",
        f"- rule_weight: {policy.get('rule_weight')}",
        f"- ml_weight: {policy.get('ml_weight')}",
        "",
        "## Paper Trading Metrics",
        f"- labeled_trades: {paper.get('labeled_trades')}",
        f"- mean_net_return_5d: {paper.get('mean_net_return_5d')}",
        f"- hit_rate_5d: {paper.get('hit_rate_5d')}",
        f"- avg_max_drawdown_5d: {paper.get('avg_max_drawdown_5d')}",
        "",
        "## Next",
        "- 승격된 active model이 있어도 risk_guard 없이 주문 후보를 강화하지 않습니다.",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote best experiment model into active model paths if criteria pass.")
    parser.add_argument("--min-precision-at-3", type=float, default=0.55)
    parser.add_argument("--min-top3-mean-return", type=float, default=0.0)
    parser.add_argument("--max-drawdown-top3", type=float, default=-0.05)
    parser.add_argument("--require-positive-excess-return", default="true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.require_positive_excess_return = str(args.require_positive_excess_return).lower() not in {"false", "0", "no"}
    return args


def main() -> int:
    args = parse_args()
    result = promote(args)
    MODEL_PROMOTION_REPORT_PATH.write_text(build_report(result), encoding="utf-8")
    print(f"Saved promotion report: {MODEL_PROMOTION_REPORT_PATH}")
    print(f"promotion_status={result.get('promotion_status')}")
    print(f"promoted={str(result.get('promoted')).lower()}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
