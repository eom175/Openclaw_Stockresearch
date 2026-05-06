#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _bootstrap import add_src_to_path

add_src_to_path()

from train_model import build_matrix, labeled_frame, ml_dependency_check
from policylink.paths import (
    ACTIVE_CALIBRATED_OUTPERFORM_MODEL_PATH,
    ACTIVE_CLASSIFIER_JOBLIB_PATH,
    ACTIVE_FEATURE_COLUMNS_PATH,
    ACTIVE_XGB_OUTPERFORM_MODEL_PATH,
    CALIBRATION_REPORT_JSON_PATH,
    CALIBRATION_REPORT_MD_PATH,
    FEATURE_COLUMNS_PATH,
    HISTORICAL_MODEL_DATASET_CSV_PATH,
    MODEL_DATASET_CSV_PATH,
    MODEL_REGISTRY_PATH,
    SIGNAL_POLICY_JSON_PATH,
    XGB_OUTPERFORM_MODEL_PATH,
    ensure_project_dirs,
)
from policylink.utils import load_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def dataset_path(args: argparse.Namespace) -> Path:
    if args.dataset_path:
        return Path(args.dataset_path)
    if args.use_historical:
        return HISTORICAL_MODEL_DATASET_CSV_PATH
    return MODEL_DATASET_CSV_PATH


def read_dataset(path: Path):
    import pandas as pd

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def no_op(status: str, reason: str, dataset: Optional[Path] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "generated_at": utc_now(),
        "mode": "probability_calibration",
        "calibration_status": status,
        "dataset_path": str(dataset) if dataset else None,
        "reason": reason,
        "before": {},
        "after": {},
        "reliability_bins": [],
        "improved": False,
        "saved_model_path": None,
        "order_enabled": False,
    }
    if extra:
        payload.update(extra)
    return payload


def candidate_model_artifacts(registry: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, str]]:
    if ACTIVE_XGB_OUTPERFORM_MODEL_PATH.exists() and ACTIVE_FEATURE_COLUMNS_PATH.exists():
        return "active_uncalibrated", {
            "model_path": str(ACTIVE_XGB_OUTPERFORM_MODEL_PATH),
            "feature_columns_path": str(ACTIVE_FEATURE_COLUMNS_PATH),
        }
    if ACTIVE_CLASSIFIER_JOBLIB_PATH.exists() and ACTIVE_FEATURE_COLUMNS_PATH.exists():
        return "active_uncalibrated", {
            "model_path": str(ACTIVE_CLASSIFIER_JOBLIB_PATH),
            "feature_columns_path": str(ACTIVE_FEATURE_COLUMNS_PATH),
        }
    if XGB_OUTPERFORM_MODEL_PATH.exists() and FEATURE_COLUMNS_PATH.exists():
        return "fallback_uncalibrated", {
            "model_path": str(XGB_OUTPERFORM_MODEL_PATH),
            "feature_columns_path": str(FEATURE_COLUMNS_PATH),
        }
    return None, {}


def load_classifier(path: str):
    model_path = Path(path)
    if model_path.suffix == ".joblib":
        import joblib

        return joblib.load(model_path)
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(model_path)
    return model


def split_by_date(labeled, min_dates: int):
    dates = list(labeled["snapshot_date"].astype(str).drop_duplicates())
    if len(dates) < min_dates:
        return None
    train_end = max(1, int(len(dates) * 0.60))
    cal_end = max(train_end + 1, int(len(dates) * 0.80))
    if cal_end >= len(dates):
        return None
    train_dates = set(dates[:train_end])
    cal_dates = set(dates[train_end:cal_end])
    test_dates = set(dates[cal_end:])
    train_idx = labeled.index[labeled["snapshot_date"].astype(str).isin(train_dates)].tolist()
    cal_idx = labeled.index[labeled["snapshot_date"].astype(str).isin(cal_dates)].tolist()
    test_idx = labeled.index[labeled["snapshot_date"].astype(str).isin(test_dates)].tolist()
    if not train_idx or not cal_idx or not test_idx:
        return None
    return train_idx, cal_idx, test_idx, {
        "train_dates": len(train_dates),
        "calibration_dates": len(cal_dates),
        "test_dates": len(test_dates),
        "train_rows": len(train_idx),
        "calibration_rows": len(cal_idx),
        "test_rows": len(test_idx),
    }


def probability_metrics(y_true, prob) -> Dict[str, Any]:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    result: Dict[str, Any] = {}
    try:
        result["log_loss"] = round(float(log_loss(y_true, prob, labels=[0, 1])), 6)
    except Exception:
        result["log_loss"] = None
    try:
        result["brier_score"] = round(float(brier_score_loss(y_true, prob)), 6)
    except Exception:
        result["brier_score"] = None
    try:
        result["roc_auc"] = round(float(roc_auc_score(y_true, prob)), 6) if len(set(y_true)) > 1 else None
    except Exception:
        result["roc_auc"] = None
    try:
        result["average_precision"] = round(float(average_precision_score(y_true, prob)), 6) if len(set(y_true)) > 1 else None
    except Exception:
        result["average_precision"] = None
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
        "precision_at_3": round(float(pd.to_numeric(top["future_outperform_5d"], errors="coerce").mean()), 6),
        "top3_mean_future_return_5d": round(float(returns.mean()), 6),
        "selected_rows_at_3": int(len(top)),
        "date_count": int(work["snapshot_date"].nunique()),
    }


def reliability_bins(y_true, prob, n_bins: int) -> List[Dict[str, Any]]:
    import pandas as pd

    frame = pd.DataFrame({"y": y_true, "prob": prob})
    rows = []
    for idx in range(max(1, n_bins)):
        low = idx / n_bins
        high = (idx + 1) / n_bins
        selected = frame[(frame["prob"] >= low) & (frame["prob"] <= high)] if idx == n_bins - 1 else frame[(frame["prob"] >= low) & (frame["prob"] < high)]
        rows.append({
            "bin_low": round(low, 4),
            "bin_high": round(high, 4),
            "predicted_prob_mean": None if selected.empty else round(float(selected["prob"].mean()), 6),
            "actual_positive_rate": None if selected.empty else round(float(selected["y"].mean()), 6),
            "count": int(len(selected)),
        })
    return rows


def is_improved(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    log_before = before.get("log_loss")
    log_after = after.get("log_loss")
    brier_before = before.get("brier_score")
    brier_after = after.get("brier_score")
    log_better = log_before is not None and log_after is not None and log_after < log_before
    brier_better = brier_before is not None and brier_after is not None and brier_after < brier_before
    return bool(log_better or brier_better)


def update_registry(result: Dict[str, Any]) -> None:
    registry = load_json(MODEL_REGISTRY_PATH, {})
    signal_policy = registry.get("signal_policy") if isinstance(registry.get("signal_policy"), dict) else {}
    policy_file = load_json(SIGNAL_POLICY_JSON_PATH, {})
    if result.get("improved"):
        for policy in [signal_policy, policy_file]:
            if isinstance(policy, dict):
                blockers = [item for item in policy.get("promotion_blockers", []) if item != "calibration_missing"]
                policy["promotion_blockers"] = blockers
                policy["calibrated"] = True
                policy["calibration_status"] = result.get("calibration_status")
                if policy.get("reason"):
                    policy["reason"] = str(policy["reason"]).replace(", uncalibrated", "").replace("uncalibrated, ", "")
    if policy_file:
        save_json(SIGNAL_POLICY_JSON_PATH, policy_file)
    registry["generated_at"] = utc_now()
    registry["calibration"] = {
        "status": result.get("calibration_status"),
        "method": result.get("method"),
        "improved": bool(result.get("improved")),
        "model_path": result.get("saved_model_path"),
        "metrics_before": result.get("before", {}),
        "metrics_after": result.get("after", {}),
        "model_source": result.get("model_source"),
        "updated_at": utc_now(),
    }
    if signal_policy:
        registry["signal_policy"] = signal_policy
    save_json(MODEL_REGISTRY_PATH, registry)


def calibrate(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_project_dirs()
    path = dataset_path(args)
    dependency = ml_dependency_check()
    if not dependency.get("ok"):
        return no_op("no_op_dependency_unavailable", "ML dependency를 사용할 수 없습니다.", path, dependency)

    df = read_dataset(path)
    labeled = labeled_frame(df)
    date_count = int(labeled["snapshot_date"].nunique()) if not labeled.empty and "snapshot_date" in labeled.columns else 0
    if len(labeled) < args.min_labeled_rows:
        return no_op(
            "no_op_calibration_insufficient_data",
            f"labeled rows {len(labeled)} < min_labeled_rows={args.min_labeled_rows}",
            path,
            {"labeled_rows": len(labeled), "date_count": date_count},
        )
    if date_count < args.min_dates:
        return no_op(
            "no_op_calibration_insufficient_data",
            f"snapshot dates {date_count} < min_dates={args.min_dates}",
            path,
            {"labeled_rows": len(labeled), "date_count": date_count},
        )

    registry = load_json(MODEL_REGISTRY_PATH, {})
    source, artifacts = candidate_model_artifacts(registry)
    if not source or not artifacts.get("model_path") or not artifacts.get("feature_columns_path"):
        return no_op("no_op_no_model", "calibration 대상 모델 파일이 없습니다.", path, {"labeled_rows": len(labeled), "date_count": date_count})

    feature_meta = load_json(Path(artifacts["feature_columns_path"]), {})
    encoded_columns = feature_meta.get("encoded_columns") or []
    if not encoded_columns:
        return no_op("no_op_no_model", "feature metadata에 encoded_columns가 없습니다.", path)

    labeled = labeled.sort_values(["snapshot_date", "stock_code"] if "stock_code" in labeled.columns else ["snapshot_date"])
    split = split_by_date(labeled, args.min_dates)
    if split is None:
        return no_op("no_op_calibration_insufficient_data", "train/calibration/test split을 만들 수 없습니다.", path)
    train_idx, cal_idx, test_idx, split_meta = split

    X, _ = build_matrix(
        labeled,
        feature_meta.get("numeric_columns", []),
        feature_meta.get("categorical_columns", []),
        feature_meta.get("numeric_medians", {}),
        encoded_columns,
    )
    y = labeled["future_outperform_5d"].astype(int)
    if len(set(y.loc[cal_idx])) < 2 or len(set(y.loc[test_idx])) < 2:
        return no_op(
            "no_op_calibration_insufficient_data",
            "calibration/test split target이 한 클래스뿐입니다.",
            path,
            {"labeled_rows": len(labeled), "date_count": date_count, "split": split_meta},
        )

    model = load_classifier(artifacts["model_path"])
    before_prob = model.predict_proba(X.loc[test_idx])[:, 1]

    from sklearn.calibration import CalibratedClassifierCV

    try:
        calibrated = CalibratedClassifierCV(model, method=args.method, cv="prefit")
        calibrated.fit(X.loc[cal_idx], y.loc[cal_idx])
    except TypeError:
        calibrated = CalibratedClassifierCV(estimator=model, method=args.method, cv="prefit")
        calibrated.fit(X.loc[cal_idx], y.loc[cal_idx])

    after_prob = calibrated.predict_proba(X.loc[test_idx])[:, 1]
    test = labeled.loc[test_idx].copy()
    before = probability_metrics(y.loc[test_idx], before_prob)
    before.update(ranking_metrics(test, before_prob))
    after = probability_metrics(y.loc[test_idx], after_prob)
    after.update(ranking_metrics(test, after_prob))
    improved = is_improved(before, after)
    saved_path = None
    if improved and args.save_if_improved:
        import joblib

        ACTIVE_CALIBRATED_OUTPERFORM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrated, ACTIVE_CALIBRATED_OUTPERFORM_MODEL_PATH)
        shutil.copy2(Path(artifacts["feature_columns_path"]), ACTIVE_FEATURE_COLUMNS_PATH)
        saved_path = str(ACTIVE_CALIBRATED_OUTPERFORM_MODEL_PATH)

    result = {
        "generated_at": utc_now(),
        "mode": "probability_calibration",
        "calibration_status": "completed" if improved else "calibration_not_improved",
        "dataset_path": str(path),
        "model_source": source,
        "method": args.method,
        "row_count": int(len(labeled)),
        "date_count": date_count,
        "split": split_meta,
        "before": before,
        "after": after,
        "reliability_bins_before": reliability_bins(y.loc[test_idx], before_prob, args.n_bins),
        "reliability_bins": reliability_bins(y.loc[test_idx], after_prob, args.n_bins),
        "improved": improved,
        "saved_model_path": saved_path,
        "save_if_improved": bool(args.save_if_improved),
        "order_enabled": False,
    }
    update_registry(result)
    return result


def build_report(result: Dict[str, Any]) -> str:
    lines = [
        "# Calibration Report",
        "",
        f"- generated_at: {result.get('generated_at')}",
        "- order_enabled=false",
        f"- calibration_status: {result.get('calibration_status')}",
        f"- dataset_path: {result.get('dataset_path')}",
        f"- model_source: {result.get('model_source')}",
        f"- method: {result.get('method')}",
        f"- improved: {result.get('improved')}",
        f"- saved_model_path: {result.get('saved_model_path')}",
        "",
    ]
    if str(result.get("calibration_status", "")).startswith("no_op"):
        lines.append("## No-op")
        lines.append(f"- reason: {result.get('reason')}")
        return "\n".join(lines)
    if result.get("calibration_status") == "calibration_not_improved":
        lines.append("## Calibration Not Improved")
        lines.append("- 보정 후 log_loss 또는 brier_score가 개선되지 않아 calibrated model을 저장하지 않았습니다.")
        lines.append("")
    lines.extend([
        "## Split",
        json.dumps(result.get("split", {}), ensure_ascii=False, indent=2),
        "",
        "## Before",
        json.dumps(result.get("before", {}), ensure_ascii=False, indent=2),
        "",
        "## After",
        json.dumps(result.get("after", {}), ensure_ascii=False, indent=2),
        "",
        "## Reliability Table After",
    ])
    for row in result.get("reliability_bins", []):
        lines.append(f"- {row['bin_low']}-{row['bin_high']}: pred={row['predicted_prob_mean']} actual={row['actual_positive_rate']} count={row['count']}")
    lines.extend([
        "",
        "## Notes",
        "- Calibration은 probability 품질 검증 단계이며 주문 실행과 무관합니다.",
        "- 개선되지 않은 calibrated model은 active 경로에 저장하지 않습니다.",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate outperform probability for active/fallback classifier.")
    parser.add_argument("--dataset-path")
    parser.add_argument("--use-historical", action="store_true")
    parser.add_argument("--method", choices=["sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--min-labeled-rows", type=int, default=300)
    parser.add_argument("--min-dates", type=int, default=30)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--save-if-improved", default="true")
    args = parser.parse_args()
    args.save_if_improved = str(args.save_if_improved).lower() not in {"false", "0", "no"}
    return args


def main() -> int:
    args = parse_args()
    result = calibrate(args)
    save_json(CALIBRATION_REPORT_JSON_PATH, result)
    CALIBRATION_REPORT_MD_PATH.write_text(build_report(result), encoding="utf-8")
    print(f"Saved calibration json: {CALIBRATION_REPORT_JSON_PATH}")
    print(f"Saved calibration report: {CALIBRATION_REPORT_MD_PATH}")
    print(f"calibration_status={result.get('calibration_status')}")
    print(f"improved={str(result.get('improved')).lower()}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
