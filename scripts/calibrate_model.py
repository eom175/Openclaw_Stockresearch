#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _bootstrap import add_src_to_path

add_src_to_path()

from experiment_models import dataset_path, read_dataset
from train_model import build_matrix, labeled_frame
from policylink.paths import (
    ACTIVE_CALIBRATED_OUTPERFORM_MODEL_PATH,
    ACTIVE_CLASSIFIER_JOBLIB_PATH,
    ACTIVE_FEATURE_COLUMNS_PATH,
    ACTIVE_XGB_OUTPERFORM_MODEL_PATH,
    CALIBRATION_REPORT_JSON_PATH,
    CALIBRATION_REPORT_MD_PATH,
    EXPERIMENT_RESULTS_PATH,
    MODEL_REGISTRY_PATH,
    ensure_project_dirs,
)
from policylink.utils import load_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def no_op(reason: str, dataset: Optional[Path] = None) -> Dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "mode": "probability_calibration",
        "calibration_status": "no_op",
        "dataset_path": str(dataset) if dataset else None,
        "reason": reason,
        "before": {},
        "after": {},
        "reliability_bins": [],
        "saved_model_path": None,
        "order_enabled": False,
    }


def best_experiment_artifacts() -> Tuple[Optional[str], Dict[str, str]]:
    result = load_json(EXPERIMENT_RESULTS_PATH, {})
    best = result.get("best_experiment") or {}
    experiment_id = best.get("experiment_id")
    if not experiment_id:
        return None, {}
    for item in result.get("experiments", []):
        if item.get("experiment_id") == experiment_id and item.get("status") == "completed":
            return experiment_id, item.get("artifact_paths") or {}
    return None, {}


def active_artifacts() -> Tuple[Optional[str], Dict[str, str]]:
    if ACTIVE_CALIBRATED_OUTPERFORM_MODEL_PATH.exists():
        return "active_calibrated", {"model_path": str(ACTIVE_CALIBRATED_OUTPERFORM_MODEL_PATH), "feature_columns_path": str(ACTIVE_FEATURE_COLUMNS_PATH)}
    if ACTIVE_XGB_OUTPERFORM_MODEL_PATH.exists() and ACTIVE_FEATURE_COLUMNS_PATH.exists():
        return "active", {"model_path": str(ACTIVE_XGB_OUTPERFORM_MODEL_PATH), "feature_columns_path": str(ACTIVE_FEATURE_COLUMNS_PATH)}
    if ACTIVE_CLASSIFIER_JOBLIB_PATH.exists() and ACTIVE_FEATURE_COLUMNS_PATH.exists():
        return "active", {"model_path": str(ACTIVE_CLASSIFIER_JOBLIB_PATH), "feature_columns_path": str(ACTIVE_FEATURE_COLUMNS_PATH)}
    return best_experiment_artifacts()


def load_model(model_path: str):
    path = Path(model_path)
    if path.suffix == ".joblib":
        import joblib

        return joblib.load(path)
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(path)
    return model


def load_feature_meta(path: str) -> Dict[str, Any]:
    return load_json(Path(path), {})


def probability_metrics(y_true, prob) -> Dict[str, Any]:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    result = {}
    try:
        result["log_loss"] = float(log_loss(y_true, prob, labels=[0, 1]))
    except Exception:
        result["log_loss"] = None
    try:
        result["brier_score"] = float(brier_score_loss(y_true, prob))
    except Exception:
        result["brier_score"] = None
    try:
        result["roc_auc"] = float(roc_auc_score(y_true, prob)) if len(set(y_true)) > 1 else None
    except Exception:
        result["roc_auc"] = None
    return result


def ranking_metrics(frame, prob) -> Dict[str, Any]:
    import pandas as pd

    work = frame.copy()
    work["prob"] = prob
    selected = []
    for _, group in work.groupby("snapshot_date"):
        selected.append(group.sort_values("prob", ascending=False).head(3))
    if not selected:
        return {}
    top = pd.concat(selected)
    returns = pd.to_numeric(top["future_return_5d"], errors="coerce")
    return {
        "precision_at_3": float(pd.to_numeric(top["future_outperform_5d"], errors="coerce").mean()),
        "top3_mean_return": float(returns.mean()),
    }


def reliability_bins(y_true, prob, n_bins: int) -> List[Dict[str, Any]]:
    import pandas as pd

    frame = pd.DataFrame({"y": y_true, "prob": prob})
    bins = []
    for idx in range(n_bins):
        low = idx / n_bins
        high = (idx + 1) / n_bins
        if idx == n_bins - 1:
            selected = frame[(frame["prob"] >= low) & (frame["prob"] <= high)]
        else:
            selected = frame[(frame["prob"] >= low) & (frame["prob"] < high)]
        bins.append({
            "bin_low": round(low, 4),
            "bin_high": round(high, 4),
            "predicted_prob_mean": None if selected.empty else round(float(selected["prob"].mean()), 6),
            "actual_positive_rate": None if selected.empty else round(float(selected["y"].mean()), 6),
            "count": int(len(selected)),
        })
    return bins


def calibrate(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_project_dirs()
    path = dataset_path(args)
    df = read_dataset(path)
    labeled = labeled_frame(df)
    if len(labeled) < args.min_labeled_rows:
        return no_op(f"labeled rows {len(labeled)} < min_labeled_rows={args.min_labeled_rows}", path)

    source, artifacts = active_artifacts()
    if not source or not artifacts.get("model_path") or not artifacts.get("feature_columns_path"):
        return no_op("active model 또는 best experiment model이 없습니다.", path)

    feature_meta = load_feature_meta(artifacts["feature_columns_path"])
    encoded_columns = feature_meta.get("encoded_columns") or []
    if not encoded_columns:
        return no_op("feature metadata에 encoded_columns가 없습니다.", path)

    labeled = labeled.sort_values(["snapshot_date", "stock_code"])
    n = len(labeled)
    train_end = max(1, int(n * 0.60))
    cal_end = max(train_end + 1, int(n * 0.80))
    if cal_end >= n:
        return no_op("train/calibration/test split을 만들 데이터가 부족합니다.", path)

    X, _ = build_matrix(
        labeled,
        feature_meta.get("numeric_columns", []),
        feature_meta.get("categorical_columns", []),
        feature_meta.get("numeric_medians", {}),
        encoded_columns,
    )
    y = labeled["future_outperform_5d"].astype(int)
    model = load_model(artifacts["model_path"])
    before_prob = model.predict_proba(X.iloc[cal_end:])[:, 1]

    from sklearn.calibration import CalibratedClassifierCV

    try:
        calibrated = CalibratedClassifierCV(model, method=args.method, cv="prefit")
        calibrated.fit(X.iloc[train_end:cal_end], y.iloc[train_end:cal_end])
    except TypeError:
        calibrated = CalibratedClassifierCV(estimator=model, method=args.method, cv="prefit")
        calibrated.fit(X.iloc[train_end:cal_end], y.iloc[train_end:cal_end])

    after_prob = calibrated.predict_proba(X.iloc[cal_end:])[:, 1]
    test = labeled.iloc[cal_end:].copy()
    before = probability_metrics(y.iloc[cal_end:], before_prob)
    before.update(ranking_metrics(test, before_prob))
    after = probability_metrics(y.iloc[cal_end:], after_prob)
    after.update(ranking_metrics(test, after_prob))

    improved = (
        after.get("brier_score") is not None
        and before.get("brier_score") is not None
        and after["brier_score"] <= before["brier_score"]
    )
    saved_path = None
    if improved:
        import joblib

        ACTIVE_CALIBRATED_OUTPERFORM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrated, ACTIVE_CALIBRATED_OUTPERFORM_MODEL_PATH)
        saved_path = str(ACTIVE_CALIBRATED_OUTPERFORM_MODEL_PATH)

    return {
        "generated_at": utc_now(),
        "mode": "probability_calibration",
        "calibration_status": "completed",
        "dataset_path": str(path),
        "model_source": source,
        "method": args.method,
        "row_count": int(len(labeled)),
        "test_row_count": int(len(test)),
        "before": before,
        "after": after,
        "reliability_bins": reliability_bins(y.iloc[cal_end:], after_prob, args.n_bins),
        "improved": improved,
        "saved_model_path": saved_path,
        "order_enabled": False,
    }


def build_report(result: Dict[str, Any]) -> str:
    lines = [
        "# Calibration Report",
        "",
        f"- generated_at: {result.get('generated_at')}",
        "- order_enabled=false",
        f"- calibration_status: {result.get('calibration_status')}",
        f"- dataset_path: {result.get('dataset_path')}",
        "",
    ]
    if result.get("calibration_status") == "no_op":
        lines.append("## No-op")
        lines.append(f"- reason: {result.get('reason')}")
        return "\n".join(lines)
    lines.extend([
        "## Before",
        json.dumps(result.get("before", {}), ensure_ascii=False, indent=2),
        "",
        "## After",
        json.dumps(result.get("after", {}), ensure_ascii=False, indent=2),
        "",
        "## Reliability Table",
    ])
    for row in result.get("reliability_bins", []):
        lines.append(f"- {row['bin_low']}-{row['bin_high']}: pred={row['predicted_prob_mean']} actual={row['actual_positive_rate']} count={row['count']}")
    lines.append(f"- calibrated model saved: {result.get('saved_model_path')}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate outperform probability for active/best classifier.")
    parser.add_argument("--dataset-path")
    parser.add_argument("--use-historical", action="store_true")
    parser.add_argument("--method", choices=["sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--min-labeled-rows", type=int, default=300)
    parser.add_argument("--n-bins", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = calibrate(args)
    save_json(CALIBRATION_REPORT_JSON_PATH, result)
    CALIBRATION_REPORT_MD_PATH.write_text(build_report(result), encoding="utf-8")
    print(f"Saved calibration json: {CALIBRATION_REPORT_JSON_PATH}")
    print(f"Saved calibration report: {CALIBRATION_REPORT_MD_PATH}")
    print(f"calibration_status={result.get('calibration_status')}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
