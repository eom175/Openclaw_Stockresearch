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
    CALIBRATION_REPORT_JSON_PATH,
    EXPERIMENT_RESULTS_PATH,
    HISTORICAL_DATASET_AUDIT_JSON_PATH,
    MODEL_PROMOTION_REPORT_PATH,
    MODEL_REGISTRY_PATH,
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


def promotion_decision(best: Optional[Dict[str, Any]], args: argparse.Namespace, audit: Dict[str, Any]) -> Dict[str, Any]:
    if not best:
        return {"promoted": False, "reasons": ["best experiment가 없습니다."]}
    reasons = []
    metrics = best.get("metrics_mean") or {}
    if best.get("status") != "completed":
        reasons.append("best experiment status가 completed가 아닙니다.")
    if (metrics.get("precision_at_3") or 0.0) < args.min_precision_at_3:
        reasons.append(f"precision_at_3 {metrics.get('precision_at_3')} < {args.min_precision_at_3}")
    if (metrics.get("top3_mean_future_return_5d") or 0.0) <= args.min_top3_mean_return:
        reasons.append(f"top3_mean_future_return_5d {metrics.get('top3_mean_future_return_5d')} <= {args.min_top3_mean_return}")
    if args.require_positive_excess_return and (metrics.get("excess_return_top3_vs_all") or 0.0) <= 0:
        reasons.append(f"excess_return_top3_vs_all {metrics.get('excess_return_top3_vs_all')} <= 0")
    drawdown = metrics.get("top3_avg_future_max_drawdown_5d")
    if drawdown is not None and drawdown < args.max_drawdown_top3:
        reasons.append(f"top3_avg_future_max_drawdown_5d {drawdown} < {args.max_drawdown_top3}")
    if (metrics.get("fold_count") or 0) < 3:
        reasons.append(f"fold_count {metrics.get('fold_count')} < 3")
    if audit.get("audit_status") == "BLOCK_TRAINING" and audit.get("leakage_candidate_columns"):
        reasons.append("historical dataset audit가 leakage 의심 컬럼으로 BLOCK_TRAINING 상태입니다.")
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
    best = find_best_experiment(results)
    decision = promotion_decision(best, args, audit)
    copied: Dict[str, str] = {}
    promoted = bool(decision["promoted"])
    if promoted and not args.dry_run:
        copied = copy_active_artifacts(best or {})
        update_registry(best or {}, copied, calibration)
    return {
        "generated_at": utc_now(),
        "mode": "model_promotion",
        "dry_run": bool(args.dry_run),
        "promotion_status": "promoted" if promoted and not args.dry_run else ("would_promote" if promoted else "rejected"),
        "promoted": promoted and not args.dry_run,
        "best_experiment": best,
        "reasons": decision["reasons"],
        "active_paths": copied,
        "criteria": {
            "min_precision_at_3": args.min_precision_at_3,
            "min_top3_mean_return": args.min_top3_mean_return,
            "max_drawdown_top3": args.max_drawdown_top3,
            "require_positive_excess_return": args.require_positive_excess_return,
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
    lines.extend(["", "## Best Experiment Metrics"])
    if best:
        lines.append(f"- experiment_id: {best.get('experiment_id')}")
        lines.append(f"- best_threshold: {best.get('best_threshold')}")
        for key, value in (best.get("metrics_mean") or {}).items():
            if key in {"precision_at_3", "top3_mean_future_return_5d", "excess_return_top3_vs_all", "top3_avg_future_max_drawdown_5d", "fold_count"}:
                lines.append(f"- {key}: {value}")
    else:
        lines.append("- best experiment 없음")
    lines.extend(["", "## Next", "- 승격된 active model이 있어도 risk_guard 없이 주문 후보를 강화하지 않습니다."])
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
