#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _bootstrap import add_src_to_path

add_src_to_path()

from train_model import (
    build_matrix,
    finite_float,
    labeled_frame,
    ml_dependency_check,
    walk_forward_splits,
    xgboost_unavailable_reason,
)
from policylink.paths import (
    EXPERIMENT_REPORT_PATH,
    EXPERIMENT_RESULTS_PATH,
    HISTORICAL_MODEL_DATASET_CSV_PATH,
    MODEL_DATASET_CSV_PATH,
    MODEL_EXPERIMENTS_DIR,
    ensure_project_dirs,
)


THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
TARGET = "future_outperform_5d"

FEATURE_GROUPS: Dict[str, List[str]] = {
    "account": ["is_holding", "holding_weight", "account_cash_weight", "account_invested_weight", "pending_or_reserved_amount"],
    "research": ["research_score", "risk_score", "opportunity_score", "macro_pressure_score", "final_score", "target_weight", "recommendation_rank"],
    "price": ["return_1d", "return_5d", "return_20d", "volatility_20d", "ma20_gap", "ma60_gap", "drawdown_20d", "volume_ratio_20", "trend_label", "price_risk_label"],
    "flow": ["foreign_net_1d", "foreign_net_5d", "foreign_net_20d", "institution_net_1d", "institution_net_5d", "institution_net_20d", "combined_net_1d", "combined_net_5d", "combined_net_20d", "combined_net_5d_to_avg_volume_20", "combined_net_20d_to_avg_volume_20", "foreign_weight", "foreign_limit_exhaustion_rate", "flow_label"],
    "dart": ["dart_"],
    "naver": ["naver_"],
    "yahoo": ["yahoo_"],
}

FEATURE_SETS: Dict[str, List[str]] = {
    "price_only": ["price"],
    "price_flow": ["price", "flow"],
    "price_flow_dart": ["price", "flow", "dart"],
    "price_flow_news": ["price", "flow", "naver"],
    "price_flow_yahoo": ["price", "flow", "yahoo"],
    "price_flow_dart_news_yahoo": ["price", "flow", "dart", "naver", "yahoo"],
    "all_features": ["account", "research", "price", "flow", "dart", "naver", "yahoo"],
    "no_news_no_disclosure": ["account", "research", "price", "flow", "yahoo"],
    "no_yahoo": ["account", "research", "price", "flow", "dart", "naver"],
    "no_research": ["price", "flow", "dart", "naver", "yahoo"],
}

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "xgb_classifier_default": {"kind": "xgb", "params": {"max_depth": 3, "min_child_weight": 3, "learning_rate": 0.03, "n_estimators": 500}},
    "xgb_classifier_conservative": {"kind": "xgb", "params": {"max_depth": 2, "min_child_weight": 5, "learning_rate": 0.03, "n_estimators": 500}},
    "logistic_regression_baseline": {"kind": "logistic", "params": {}},
    "random_forest_baseline": {"kind": "random_forest", "params": {}},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def experiment_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def dataset_path(args: argparse.Namespace) -> Path:
    if args.dataset_path:
        return Path(args.dataset_path)
    if args.use_historical and HISTORICAL_MODEL_DATASET_CSV_PATH.exists():
        return HISTORICAL_MODEL_DATASET_CSV_PATH
    if HISTORICAL_MODEL_DATASET_CSV_PATH.exists():
        return HISTORICAL_MODEL_DATASET_CSV_PATH
    return MODEL_DATASET_CSV_PATH


def read_dataset(path: Path):
    import pandas as pd

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def no_op(
    reason: str,
    dataset: Path,
    row_count: int = 0,
    labeled_rows: int = 0,
    date_count: int = 0,
    dependency_check: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dependency_check = dependency_check or {}
    return {
        "generated_at": utc_now(),
        "mode": "model_experiment",
        "experiment_status": "no_op",
        "dataset_path": str(dataset),
        "row_count": row_count,
        "labeled_row_count": labeled_rows,
        "date_range": {},
        "date_count": date_count,
        "experiments": [],
        "best_experiment": None,
        "reason": reason,
        "missing_dependencies": dependency_check.get("missing_dependencies", []),
        "optional_missing_dependencies": dependency_check.get("optional_missing_dependencies", []),
        "dependency_details": dependency_check.get("dependency_details", []),
        "install_command": dependency_check.get("install_command"),
        "system_install_suggestions": dependency_check.get("system_install_suggestions", []),
        "order_enabled": False,
    }


def group_columns(df, group_names: List[str]) -> List[str]:
    columns: List[str] = []
    for group in group_names:
        selectors = FEATURE_GROUPS.get(group, [])
        for selector in selectors:
            matched = [column for column in df.columns if column == selector or column.startswith(selector)]
            for column in matched:
                if column not in columns:
                    columns.append(column)
    return columns


def split_feature_columns(df, columns: List[str]) -> Tuple[List[str], List[str], Dict[str, float]]:
    import pandas as pd

    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    medians: Dict[str, float] = {}
    for column in columns:
        if column not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_columns.append(column)
            median = pd.to_numeric(df[column], errors="coerce").median()
            medians[column] = 0.0 if math.isnan(float(median)) else float(median)
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().mean() >= 0.80:
            df[column] = converted
            numeric_columns.append(column)
            median = converted.median()
            medians[column] = 0.0 if math.isnan(float(median)) else float(median)
        else:
            categorical_columns.append(column)
    return numeric_columns, categorical_columns, medians


def make_estimator(model_name: str, random_state: int):
    config = MODEL_CONFIGS[model_name]
    kind = config["kind"]
    if kind == "xgb":
        from xgboost import XGBClassifier

        params = dict(config["params"])
        return XGBClassifier(
            **params,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        )
    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"unknown model kind: {kind}")


def classification_metrics(y_true, prob) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score

    pred = (prob >= 0.5).astype(int)
    result = {
        "accuracy": finite_float(accuracy_score(y_true, pred)),
        "precision": finite_float(precision_score(y_true, pred, zero_division=0)),
        "recall": finite_float(recall_score(y_true, pred, zero_division=0)),
        "f1": finite_float(f1_score(y_true, pred, zero_division=0)),
        "positive_rate": finite_float(float(sum(y_true)) / len(y_true)) if len(y_true) else None,
        "predicted_positive_rate": finite_float(float(sum(pred)) / len(pred)) if len(pred) else None,
    }
    try:
        result["roc_auc"] = finite_float(roc_auc_score(y_true, prob)) if len(set(y_true)) > 1 else None
    except Exception:
        result["roc_auc"] = None
    try:
        result["average_precision"] = finite_float(average_precision_score(y_true, prob)) if len(set(y_true)) > 1 else None
    except Exception:
        result["average_precision"] = None
    try:
        result["log_loss"] = finite_float(log_loss(y_true, prob, labels=[0, 1]))
    except Exception:
        result["log_loss"] = None
    return result


def ranking_metrics(validation_rows, prob_column: str = "outperform_prob", top_k: int = 3) -> Dict[str, Any]:
    import pandas as pd

    result: Dict[str, Any] = {}
    if validation_rows.empty:
        return result
    all_returns = pd.to_numeric(validation_rows["future_return_5d"], errors="coerce")
    cost_return = 25.0 / 10000.0
    result["all_mean_future_return_5d"] = finite_float(all_returns.mean())
    result["all_mean_net_return_5d"] = finite_float(all_returns.mean() - cost_return)
    for k in [1, 3, 5]:
        selected = []
        for _, group in validation_rows.groupby("snapshot_date"):
            selected.append(group.sort_values(prob_column, ascending=False).head(k))
        if not selected:
            continue
        top = pd.concat(selected)
        top_returns = pd.to_numeric(top["future_return_5d"], errors="coerce")
        result[f"precision_at_{k}"] = finite_float(pd.to_numeric(top["future_outperform_5d"], errors="coerce").mean())
        result[f"top{k}_mean_future_return_5d"] = finite_float(top_returns.mean())
        result[f"top{k}_mean_net_return_5d"] = finite_float(top_returns.mean() - cost_return)
        result[f"top{k}_hit_rate"] = finite_float((top_returns > 0).mean())
        if "future_max_drawdown_5d" in top.columns:
            result[f"top{k}_avg_future_max_drawdown_5d"] = finite_float(pd.to_numeric(top["future_max_drawdown_5d"], errors="coerce").mean())
    if result.get(f"top{top_k}_mean_future_return_5d") is not None and result.get("all_mean_future_return_5d") is not None:
        result[f"excess_return_top{top_k}_vs_all"] = finite_float(result[f"top{top_k}_mean_future_return_5d"] - result["all_mean_future_return_5d"])
    if result.get(f"top{top_k}_mean_net_return_5d") is not None and result.get("all_mean_net_return_5d") is not None:
        result[f"excess_net_return_top{top_k}_vs_all"] = finite_float(result[f"top{top_k}_mean_net_return_5d"] - result["all_mean_net_return_5d"])
    return result


def threshold_sweep(validation_rows) -> Tuple[List[Dict[str, Any]], float]:
    if validation_rows.empty:
        return [], 0.60
    rows = []
    for threshold in THRESHOLDS:
        selected = validation_rows.loc[validation_rows["outperform_prob"] >= threshold]
        if selected.empty:
            rows.append({"threshold": threshold, "selected_count": 0, "precision": None, "mean_future_return_5d": None, "avg_future_max_drawdown_5d": None, "hit_rate": None})
            continue
        returns = selected["future_return_5d"]
        rows.append({
            "threshold": threshold,
            "selected_count": int(len(selected)),
            "precision": finite_float(selected["future_outperform_5d"].mean()),
            "mean_future_return_5d": finite_float(returns.mean()),
            "avg_future_max_drawdown_5d": finite_float(selected["future_max_drawdown_5d"].mean()) if "future_max_drawdown_5d" in selected.columns else None,
            "hit_rate": finite_float((returns > 0).mean()),
        })
    viable = [row for row in rows if row["selected_count"] >= 5 and row["precision"] is not None]
    if not viable:
        viable = [row for row in rows if row["selected_count"] > 0 and row["precision"] is not None]
    if not viable:
        return rows, 0.60
    best = sorted(viable, key=lambda row: (row.get("precision") or 0.0, row.get("mean_future_return_5d") or -999), reverse=True)[0]
    return rows, float(best["threshold"])


def mean_metrics(metrics_by_fold: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {"fold_count": len(metrics_by_fold)}
    keys = sorted({key for item in metrics_by_fold for key in item.keys() if key != "fold"})
    for key in keys:
        values = [item.get(key) for item in metrics_by_fold if isinstance(item.get(key), (int, float))]
        result[key] = finite_float(sum(values) / len(values)) if values else None
    return result


def save_model_artifact(experiment_dir: Path, model_name: str, model: Any, feature_meta: Dict[str, Any]) -> Dict[str, str]:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    feature_path = experiment_dir / "feature_columns.json"
    save_json(feature_path, feature_meta)
    config = MODEL_CONFIGS[model_name]
    if config["kind"] == "xgb":
        model_path = experiment_dir / "xgb_outperform_5d.json"
        model.save_model(model_path)
    else:
        import joblib

        model_path = experiment_dir / "classifier.joblib"
        joblib.dump(model, model_path)
    return {"model_path": str(model_path), "feature_columns_path": str(feature_path)}


def run_single_experiment(
    labeled,
    splits,
    feature_set: str,
    model_name: str,
    args: argparse.Namespace,
    run_id: str,
) -> Dict[str, Any]:
    experiment_id = f"{run_id}_{feature_set}_{model_name}"
    notes: List[str] = []
    columns = group_columns(labeled, FEATURE_SETS[feature_set])
    if not columns:
        return {"experiment_id": experiment_id, "feature_set": feature_set, "model_name": model_name, "status": "skipped", "metrics_mean": {}, "metrics_by_fold": [], "threshold_results": [], "best_threshold": 0.60, "feature_count": 0, "notes": ["no usable feature columns"]}

    if MODEL_CONFIGS[model_name]["kind"] == "xgb":
        reason = xgboost_unavailable_reason()
        if reason:
            return {"experiment_id": experiment_id, "feature_set": feature_set, "model_name": model_name, "status": "skipped", "metrics_mean": {}, "metrics_by_fold": [], "threshold_results": [], "best_threshold": 0.60, "feature_count": len(columns), "notes": [reason]}
    if model_name == "random_forest_baseline" and len(labeled) < 500:
        return {"experiment_id": experiment_id, "feature_set": feature_set, "model_name": model_name, "status": "skipped", "metrics_mean": {}, "metrics_by_fold": [], "threshold_results": [], "best_threshold": 0.60, "feature_count": len(columns), "notes": ["random forest skipped for small dataset"]}

    work = labeled.copy()
    numeric_columns, categorical_columns, medians = split_feature_columns(work, columns)
    X, encoded_columns = build_matrix(work, numeric_columns, categorical_columns, medians)
    if X.empty or len(encoded_columns) == 0:
        return {"experiment_id": experiment_id, "feature_set": feature_set, "model_name": model_name, "status": "skipped", "metrics_mean": {}, "metrics_by_fold": [], "threshold_results": [], "best_threshold": 0.60, "feature_count": 0, "notes": ["empty design matrix"]}

    y = work[TARGET].astype(int)
    metrics_by_fold = []
    validation_parts = []
    try:
        for fold_no, (train_idx, val_idx) in enumerate(splits, start=1):
            if len(set(y.loc[train_idx])) < 2:
                notes.append(f"fold {fold_no} skipped: one-class train")
                continue
            model = make_estimator(model_name, args.random_state)
            model.fit(X.loc[train_idx], y.loc[train_idx])
            prob = model.predict_proba(X.loc[val_idx])[:, 1]
            fold_metrics = classification_metrics(y.loc[val_idx], prob)
            fold_metrics["fold"] = fold_no
            fold_metrics["selected_count_per_fold"] = int((prob >= 0.60).sum())
            metrics_by_fold.append(fold_metrics)
            validation = work.loc[val_idx].copy()
            validation["outperform_prob"] = prob
            validation_parts.append(validation)
        if not validation_parts:
            return {"experiment_id": experiment_id, "feature_set": feature_set, "model_name": model_name, "status": "skipped", "metrics_mean": {}, "metrics_by_fold": [], "threshold_results": [], "best_threshold": 0.60, "feature_count": len(encoded_columns), "notes": notes or ["no valid validation folds"]}
        import pandas as pd

        validation_rows = pd.concat(validation_parts)
        metrics_mean = mean_metrics(metrics_by_fold)
        metrics_mean.update(ranking_metrics(validation_rows, top_k=args.top_k))
        threshold_results, best_threshold = threshold_sweep(validation_rows)
        final_model = make_estimator(model_name, args.random_state)
        final_model.fit(X, y)
        artifact_paths = save_model_artifact(
            MODEL_EXPERIMENTS_DIR / experiment_id,
            model_name,
            final_model,
            {
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "numeric_medians": medians,
                "encoded_columns": encoded_columns,
                "target_column": TARGET,
                "feature_set": feature_set,
                "model_name": model_name,
                "trained_at": utc_now(),
            },
        )
        return {
            "experiment_id": experiment_id,
            "feature_set": feature_set,
            "model_name": model_name,
            "status": "completed",
            "metrics_mean": metrics_mean,
            "metrics_by_fold": metrics_by_fold,
            "threshold_results": threshold_results,
            "best_threshold": best_threshold,
            "feature_count": len(encoded_columns),
            "artifact_paths": artifact_paths,
            "notes": notes,
        }
    except Exception as exc:
        return {"experiment_id": experiment_id, "feature_set": feature_set, "model_name": model_name, "status": "failed", "metrics_mean": {}, "metrics_by_fold": metrics_by_fold, "threshold_results": [], "best_threshold": 0.60, "feature_count": len(encoded_columns), "notes": [str(exc)[:400]]}


def choose_best(experiments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    completed = [item for item in experiments if item.get("status") == "completed"]
    if not completed:
        return None
    best = sorted(
        completed,
        key=lambda item: (
            item.get("metrics_mean", {}).get("precision_at_3") or 0.0,
            item.get("metrics_mean", {}).get("top3_mean_future_return_5d") or -999.0,
            item.get("metrics_mean", {}).get("roc_auc") or 0.0,
        ),
        reverse=True,
    )[0]
    return {"experiment_id": best.get("experiment_id"), "selection_reason": "max precision_at_3, then top3 mean return, then roc_auc"}


def run_experiments(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_project_dirs()
    path = dataset_path(args)
    df = read_dataset(path)
    dependency = ml_dependency_check()
    if not dependency.get("ok"):
        return no_op(
            "ML dependency를 사용할 수 없어 모델 실험을 보류합니다.",
            path,
            len(df),
            0,
            0,
            dependency_check=dependency,
        )
    labeled = labeled_frame(df)
    labeled_rows = int(len(labeled))
    date_count = int(labeled["snapshot_date"].nunique()) if not labeled.empty else 0
    if df.empty:
        return no_op("dataset file missing or empty", path)
    if labeled_rows < args.min_labeled_rows:
        return no_op(f"labeled rows {labeled_rows} < min_labeled_rows={args.min_labeled_rows}", path, len(df), labeled_rows, date_count)
    if date_count < args.min_dates:
        return no_op(f"snapshot dates {date_count} < min_dates={args.min_dates}", path, len(df), labeled_rows, date_count)
    if len(set(labeled[TARGET].astype(int))) < 2:
        return no_op("classification target has one class", path, len(df), labeled_rows, date_count)
    splits = walk_forward_splits(labeled, args.n_splits, test_size_days=5, gap_days=1)
    if not splits:
        return no_op("walk-forward splits unavailable", path, len(df), labeled_rows, date_count)

    feature_sets = list(FEATURE_SETS)
    model_names = list(MODEL_CONFIGS)
    if args.quick:
        feature_sets = ["price_only", "price_flow", "all_features"]
        model_names = ["logistic_regression_baseline", "xgb_classifier_default"]
    run_id = experiment_timestamp()
    experiments = [
        run_single_experiment(labeled, splits, feature_set, model_name, args, run_id)
        for feature_set in feature_sets
        for model_name in model_names
    ]
    return {
        "generated_at": utc_now(),
        "mode": "model_experiment",
        "experiment_status": "completed",
        "dataset_path": str(path),
        "row_count": int(len(df)),
        "labeled_row_count": labeled_rows,
        "date_count": date_count,
        "date_range": {"min_snapshot_date": str(labeled["snapshot_date"].min()), "max_snapshot_date": str(labeled["snapshot_date"].max())},
        "experiments": experiments,
        "best_experiment": choose_best(experiments),
        "order_enabled": False,
    }


def build_report(result: Dict[str, Any]) -> str:
    lines = [
        "# Model Experiment Report",
        "",
        f"- generated_at: {result.get('generated_at')}",
        "- order_enabled=false",
        f"- experiment_status: {result.get('experiment_status')}",
        f"- dataset_path: {result.get('dataset_path')}",
        f"- row_count: {result.get('row_count', 0)}",
        f"- labeled_row_count: {result.get('labeled_row_count', 0)}",
        f"- date_count: {result.get('date_count', 0)}",
        "",
    ]
    if result.get("experiment_status") == "no_op":
        lines.append("## No-op")
        lines.append(f"- reason: {result.get('reason')}")
        if result.get("missing_dependencies"):
            lines.append(f"- missing_dependencies: {result.get('missing_dependencies')}")
            lines.append(f"- install_command: {result.get('install_command')}")
        if result.get("system_install_suggestions"):
            lines.append(f"- system_install_suggestions: {result.get('system_install_suggestions')}")
        return "\n".join(lines)

    lines.append("## Feature Set / Model Results")
    lines.append("| feature_set | model | status | precision_at_3 | top3_net_return | roc_auc | best_threshold | notes |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for item in result.get("experiments", []):
        metrics = item.get("metrics_mean", {})
        notes = "; ".join(item.get("notes", [])[:2])
        lines.append(
            f"| {item.get('feature_set')} | {item.get('model_name')} | {item.get('status')} | "
            f"{metrics.get('precision_at_3')} | {metrics.get('top3_mean_net_return_5d')} | "
            f"{metrics.get('roc_auc')} | {item.get('best_threshold')} | {notes} |"
        )
    lines.extend(["", "## Best Experiment"])
    best = result.get("best_experiment")
    if best:
        lines.append(f"- experiment_id: {best.get('experiment_id')}")
        lines.append(f"- reason: {best.get('selection_reason')}")
    else:
        lines.append("- completed experiment가 없어 best experiment를 고르지 않았습니다.")
    lines.extend([
        "",
        "## Notes",
        "- 검증은 snapshot_date 기준 walk-forward 방식이며 랜덤 분할을 쓰지 않습니다.",
        "- 모델 실험은 주문을 실행하지 않습니다.",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare feature groups and classifier configurations for model research.")
    parser.add_argument("--dataset-path")
    parser.add_argument("--use-historical", action="store_true")
    parser.add_argument("--min-labeled-rows", type=int, default=300)
    parser.add_argument("--min-dates", type=int, default=30)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_experiments(args)
    save_json(EXPERIMENT_RESULTS_PATH, result)
    EXPERIMENT_REPORT_PATH.write_text(build_report(result), encoding="utf-8")
    print(f"Saved experiment results: {EXPERIMENT_RESULTS_PATH}")
    print(f"Saved experiment report: {EXPERIMENT_REPORT_PATH}")
    print(f"experiment_status={result.get('experiment_status')}")
    print(f"best_experiment={(result.get('best_experiment') or {}).get('experiment_id')}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
