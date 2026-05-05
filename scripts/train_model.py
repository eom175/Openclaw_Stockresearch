#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from _bootstrap import add_src_to_path

add_src_to_path()

from policylink.paths import (
    DATASET_AUDIT_JSON_PATH,
    FEATURE_COLUMNS_PATH,
    HISTORICAL_MODEL_DATASET_CSV_PATH,
    MODEL_DATASET_CSV_PATH,
    MODEL_EXPERIMENTS_DIR,
    MODEL_QUALITY_METRICS_PATH,
    MODEL_QUALITY_REPORT_PATH,
    MODEL_REGISTRY_PATH,
    MODEL_TRAINING_METRICS_PATH,
    MODEL_TRAINING_REPORT_PATH,
    XGB_DRAWDOWN_METADATA_PATH,
    XGB_DRAWDOWN_MODEL_PATH,
    XGB_OUTPERFORM_METADATA_PATH,
    XGB_OUTPERFORM_MODEL_PATH,
    XGB_RETURN_METADATA_PATH,
    XGB_RETURN_MODEL_PATH,
    ensure_project_dirs,
)
from policylink.utils import load_json


TARGET_COLUMNS = {
    "classification": "future_outperform_5d",
    "return_regression": "future_return_5d",
    "drawdown_regression": "future_max_drawdown_5d",
}

EXACT_EXCLUDE_COLUMNS = {
    "generated_at",
    "stock_name",
    "snapshot_date",
    "future_return_1d",
    "future_return_5d",
    "future_return_20d",
    "future_outperform_5d",
    "future_max_drawdown_5d",
    "label_status",
    "label_updated_at",
}

LEAKAGE_TOKENS = ["future", "target", "label"]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
FEATURE_GROUPS = {
    "base_price": ["return_", "volatility", "ma20", "ma60", "drawdown", "volume_ratio", "latest_close"],
    "flow": ["flow_", "foreign_", "institution_", "combined_net"],
    "dart": ["dart_"],
    "naver": ["naver_"],
    "yahoo": ["yahoo_"],
    "research": ["research_", "risk_score", "opportunity_score", "macro_pressure_score"],
    "account": ["account_", "holding_", "pending_or_reserved", "is_holding"],
}

ML_DEPENDENCIES = [
    {"name": "pandas", "module": "pandas", "required": True},
    {"name": "numpy", "module": "numpy", "required": True},
    {"name": "scikit-learn", "module": "sklearn", "required": True},
    {"name": "xgboost", "module": "xgboost", "required": True},
    {"name": "joblib", "module": "joblib", "required": True},
    {"name": "shap", "module": "shap", "required": False},
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_json(path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def finite_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return round(numeric, 6)
    except Exception:
        return None


def resolve_dataset_path(args: argparse.Namespace):
    if args.dataset_path:
        from pathlib import Path

        return Path(args.dataset_path)
    if args.use_historical:
        return HISTORICAL_MODEL_DATASET_CSV_PATH
    return MODEL_DATASET_CSV_PATH


def read_dataset(args: argparse.Namespace):
    import pandas as pd

    dataset_path = resolve_dataset_path(args)
    if not dataset_path.exists():
        return pd.DataFrame()
    return pd.read_csv(dataset_path)


def labeled_frame(df):
    import pandas as pd

    if df.empty:
        return df
    required = ["label_status", "future_return_5d", "future_outperform_5d", "snapshot_date"]
    if any(column not in df.columns for column in required):
        return df.iloc[0:0].copy()

    mask = df["label_status"].isin(["partially_labeled", "labeled"])
    mask &= pd.to_numeric(df["future_return_5d"], errors="coerce").notna()
    mask &= pd.to_numeric(df["future_outperform_5d"], errors="coerce").notna()
    result = df.loc[mask].copy()
    result["future_return_5d"] = pd.to_numeric(result["future_return_5d"], errors="coerce")
    result["future_outperform_5d"] = pd.to_numeric(result["future_outperform_5d"], errors="coerce").astype(int)
    if "future_max_drawdown_5d" in result.columns:
        result["future_max_drawdown_5d"] = pd.to_numeric(result["future_max_drawdown_5d"], errors="coerce")
    result = result.sort_values(["snapshot_date", "stock_code"] if "stock_code" in result.columns else ["snapshot_date"])
    return result


def is_excluded_feature(column: str) -> bool:
    lowered = column.lower()
    if column in EXACT_EXCLUDE_COLUMNS:
        return True
    return any(token in lowered for token in LEAKAGE_TOKENS)


def choose_feature_columns(df, max_features: Optional[int] = None) -> Tuple[List[str], List[str], Dict[str, float]]:
    import pandas as pd

    candidate_columns = [column for column in df.columns if not is_excluded_feature(column)]
    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    numeric_medians: Dict[str, float] = {}

    for column in candidate_columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
            numeric_medians[column] = finite_float(pd.to_numeric(series, errors="coerce").median()) or 0.0
            continue

        converted = pd.to_numeric(series, errors="coerce")
        non_null_ratio = converted.notna().mean() if len(converted) else 0
        if non_null_ratio >= 0.80:
            df[column] = converted
            numeric_columns.append(column)
            numeric_medians[column] = finite_float(converted.median()) or 0.0
        else:
            categorical_columns.append(column)

    if max_features and max_features > 0:
        numeric_columns = numeric_columns[:max_features]
        remaining = max_features - len(numeric_columns)
        categorical_columns = categorical_columns[: max(0, remaining)]
        numeric_medians = {key: value for key, value in numeric_medians.items() if key in numeric_columns}

    return numeric_columns, categorical_columns, numeric_medians


def build_matrix(
    df,
    numeric_columns: List[str],
    categorical_columns: List[str],
    numeric_medians: Optional[Dict[str, float]] = None,
    encoded_columns: Optional[List[str]] = None,
):
    import pandas as pd

    numeric_medians = numeric_medians or {}
    parts = []
    if numeric_columns:
        numeric = df.reindex(columns=numeric_columns).apply(pd.to_numeric, errors="coerce")
        for column in numeric_columns:
            numeric[column] = numeric[column].fillna(numeric_medians.get(column, 0.0))
        parts.append(numeric)
    if categorical_columns:
        categorical = df.reindex(columns=categorical_columns).fillna("missing").astype(str)
        encoded = pd.get_dummies(categorical, columns=categorical_columns, dummy_na=False)
        parts.append(encoded)

    if parts:
        matrix = pd.concat(parts, axis=1)
    else:
        matrix = pd.DataFrame(index=df.index)

    if encoded_columns is None:
        encoded_columns = list(matrix.columns)
    matrix = matrix.reindex(columns=encoded_columns, fill_value=0.0)
    return matrix.astype(float), encoded_columns


def walk_forward_splits(df, n_splits: int, test_size_days: int, gap_days: int) -> List[Tuple[List[int], List[int]]]:
    dates = list(df["snapshot_date"].astype(str).drop_duplicates())
    if len(dates) < 3:
        return []

    test_size_days = max(1, test_size_days)
    gap_days = max(0, gap_days)
    splits = []
    max_splits = min(n_splits, max(1, (len(dates) - gap_days - 1) // test_size_days))

    for split_no in range(max_splits, 0, -1):
        test_end = len(dates) - (max_splits - split_no) * test_size_days
        test_start = max(0, test_end - test_size_days)
        train_end = max(0, test_start - gap_days)
        if train_end <= 0 or test_start >= test_end:
            continue

        train_dates = set(dates[:train_end])
        test_dates = set(dates[test_start:test_end])
        train_idx = df.index[df["snapshot_date"].astype(str).isin(train_dates)].tolist()
        test_idx = df.index[df["snapshot_date"].astype(str).isin(test_dates)].tolist()
        if train_idx and test_idx:
            splits.append((train_idx, test_idx))

    splits.reverse()
    return splits[:n_splits]


def classification_metrics(y_true, y_pred, y_prob) -> Dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    result = {
        "accuracy": finite_float(accuracy_score(y_true, y_pred)),
        "precision": finite_float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": finite_float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": finite_float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": finite_float(sum(y_true) / len(y_true)) if len(y_true) else None,
        "predicted_positive_rate": finite_float(sum(y_pred) / len(y_pred)) if len(y_pred) else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    try:
        result["roc_auc"] = finite_float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None
    except Exception:
        result["roc_auc"] = None
    try:
        result["average_precision"] = finite_float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else None
    except Exception:
        result["average_precision"] = None
    return result


def regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    try:
        correlation = float(y_true.corr(y_pred)) if hasattr(y_true, "corr") else None
    except Exception:
        correlation = None
    return {
        "mae": finite_float(mean_absolute_error(y_true, y_pred)),
        "rmse": finite_float(math.sqrt(mse)),
        "r2": finite_float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None,
        "prediction_actual_correlation": finite_float(correlation),
    }


def average_fold_metrics(folds: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not folds:
        return {"fold_count": 0}

    keys = sorted({key for fold in folds for key in fold.keys() if key not in {"fold", "confusion_matrix"}})
    result: Dict[str, Any] = {"fold_count": len(folds), "folds": folds}
    for key in keys:
        values = [fold.get(key) for fold in folds if isinstance(fold.get(key), (int, float))]
        result[key] = finite_float(sum(values) / len(values)) if values else None

    matrices = [fold.get("confusion_matrix") for fold in folds if isinstance(fold.get("confusion_matrix"), list)]
    if matrices:
        result["confusion_matrix_sum"] = [
            [sum(matrix[i][j] for matrix in matrices) for j in range(2)]
            for i in range(2)
        ]
    return result


def ranking_metrics(validation_rows, prob_column: str = "outperform_prob") -> Dict[str, Any]:
    import pandas as pd

    if validation_rows.empty:
        return {}

    metrics: Dict[str, Any] = {}
    for k in [1, 3, 5]:
        selected = []
        for _, group in validation_rows.groupby("snapshot_date"):
            selected.append(group.sort_values(prob_column, ascending=False).head(k))
        if not selected:
            continue
        top = pd.concat(selected)
        actual = pd.to_numeric(top["future_outperform_5d"], errors="coerce")
        returns = pd.to_numeric(top["future_return_5d"], errors="coerce")
        drawdown = pd.to_numeric(top.get("future_max_drawdown_5d"), errors="coerce") if "future_max_drawdown_5d" in top.columns else None
        metrics[f"precision_at_{k}"] = finite_float(actual.mean())
        metrics[f"mean_future_return_top{k}"] = finite_float(returns.mean())
        metrics[f"hit_rate_top{k}"] = finite_float((returns > 0).mean())
        if drawdown is not None:
            metrics[f"average_drawdown_top{k}"] = finite_float(drawdown.mean())
    return metrics


def threshold_tuning(validation_rows) -> Dict[str, Any]:
    if validation_rows.empty:
        return {"best_threshold": 0.60, "thresholds": []}

    rows = []
    for threshold in THRESHOLDS:
        selected = validation_rows.loc[validation_rows["outperform_prob"] >= threshold]
        if selected.empty:
            rows.append({
                "threshold": threshold,
                "selected_count": 0,
                "precision": None,
                "mean_future_return_5d": None,
                "max_drawdown_avg": None,
            })
            continue

        rows.append({
            "threshold": threshold,
            "selected_count": int(len(selected)),
            "precision": finite_float(selected["future_outperform_5d"].mean()),
            "mean_future_return_5d": finite_float(selected["future_return_5d"].mean()),
            "max_drawdown_avg": finite_float(selected["future_max_drawdown_5d"].mean()) if "future_max_drawdown_5d" in selected.columns else None,
        })

    viable = [row for row in rows if row["selected_count"] >= 3 and row["precision"] is not None]
    if not viable:
        viable = [row for row in rows if row["selected_count"] > 0 and row["precision"] is not None]
    if not viable:
        return {"best_threshold": 0.60, "thresholds": rows}

    best = sorted(
        viable,
        key=lambda row: (row.get("precision") or 0.0, row.get("mean_future_return_5d") or -999.0),
        reverse=True,
    )[0]
    return {"best_threshold": best["threshold"], "thresholds": rows}


def fit_classifier(X, y, labeled, splits, random_state: int):
    from xgboost import XGBClassifier

    if len(set(y)) < 2:
        return None, {"skipped": True, "reason": "classification target has one class"}, None

    folds = []
    validation_parts = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        y_train = y.loc[train_idx]
        if len(set(y_train)) < 2:
            continue
        model = XGBClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            early_stopping_rounds=30,
        )
        model.fit(X.loc[train_idx], y_train, eval_set=[(X.loc[test_idx], y.loc[test_idx])], verbose=False)
        y_prob = model.predict_proba(X.loc[test_idx])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        fold_metrics = classification_metrics(y.loc[test_idx], y_pred, y_prob)
        fold_metrics["fold"] = fold_idx
        folds.append(fold_metrics)
        fold_rows = labeled.loc[test_idx].copy()
        fold_rows["outperform_prob"] = y_prob
        validation_parts.append(fold_rows)

    final_model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    final_model.fit(X, y)

    metrics = average_fold_metrics(folds)
    if validation_parts:
        import pandas as pd

        validation_rows = pd.concat(validation_parts)
        metrics.update(ranking_metrics(validation_rows))
        metrics["threshold_tuning"] = threshold_tuning(validation_rows)
    else:
        validation_rows = None
        metrics["threshold_tuning"] = {"best_threshold": 0.60, "thresholds": []}

    return final_model, metrics, validation_rows


def fit_regressor(X, y, splits, random_state: int):
    from xgboost import XGBRegressor

    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        model = XGBRegressor(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            early_stopping_rounds=30,
        )
        model.fit(X.loc[train_idx], y.loc[train_idx], eval_set=[(X.loc[test_idx], y.loc[test_idx])], verbose=False)
        y_pred = model.predict(X.loc[test_idx])
        fold_metrics = regression_metrics(y.loc[test_idx], y_pred)
        fold_metrics["fold"] = fold_idx
        folds.append(fold_metrics)

    final_model = XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )
    final_model.fit(X, y)
    return final_model, average_fold_metrics(folds)


def feature_importance(model, encoded_columns: List[str]) -> List[Dict[str, Any]]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    items = [
        {"feature": feature, "importance": finite_float(value) or 0.0}
        for feature, value in zip(encoded_columns, importances)
    ]
    items.sort(key=lambda x: x["importance"], reverse=True)
    return items[:30]


def ablation_summary(args, labeled, numeric_columns: List[str], categorical_columns: List[str]) -> Dict[str, Any]:
    if args.skip_ablation:
        return {"skipped": True, "reason": "--skip-ablation enabled"}
    if len(labeled) < max(150, args.min_labeled_rows) or labeled["snapshot_date"].nunique() < max(8, args.min_dates):
        return {"skipped": True, "reason": "데이터가 부족해 feature ablation을 생략합니다."}

    available = numeric_columns + categorical_columns
    result = {}
    for group, prefixes in FEATURE_GROUPS.items():
        removed = [column for column in available if any(column.startswith(prefix) or prefix in column for prefix in prefixes)]
        result[group] = {"removed_feature_count": len(removed), "status": "planned"}
    return result


def ml_dependency_check() -> Dict[str, Any]:
    details = []
    missing_required = []
    optional_missing = []
    system_install_suggestions = []
    install_command = ".venv/bin/pip install pandas numpy scikit-learn xgboost joblib shap"

    for dependency in ML_DEPENDENCIES:
        name = dependency["name"]
        module_name = dependency["module"]
        required = bool(dependency["required"])
        item = {
            "name": name,
            "module": module_name,
            "required": required,
            "available": False,
            "version": None,
            "error": None,
        }
        try:
            module = importlib.import_module(module_name)
            item["available"] = True
            item["version"] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            first_line = message.splitlines()[0]
            item["error"] = first_line[:500]
            if required:
                missing_required.append(name)
            else:
                optional_missing.append(name)
            if name == "xgboost" and ("libomp" in message.lower() or "openmp" in message.lower()):
                system_install_suggestions.append("brew install libomp")
        details.append(item)

    return {
        "ok": not missing_required,
        "missing_dependencies": missing_required,
        "optional_missing_dependencies": optional_missing,
        "dependency_details": details,
        "install_command": install_command,
        "system_install_suggestions": sorted(set(system_install_suggestions)),
    }


def xgboost_unavailable_reason() -> Optional[str]:
    check = ml_dependency_check()
    if check.get("ok"):
        return None
    missing = ", ".join(check.get("missing_dependencies") or [])
    details = [
        item.get("error")
        for item in check.get("dependency_details", [])
        if item.get("required") and not item.get("available") and item.get("error")
    ]
    detail = f" ({details[0]})" if details else ""
    return f"ML dependency를 사용할 수 없습니다: {missing}{detail}"


def save_model_metadata(path, model_name: str, metrics: Dict[str, Any], row_count: int, date_range: Dict[str, Any]) -> None:
    save_json(path, {
        "model_name": model_name,
        "trained_at": utc_now(),
        "row_count": row_count,
        "date_range": date_range,
        "metrics": metrics,
        "order_enabled": False,
    })


def registry_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    existing = load_json(MODEL_REGISTRY_PATH, {})
    registry = {
        "generated_at": result.get("generated_at"),
        "model_status": existing.get("model_status") if existing.get("active_model") else result.get("model_registry_status", result.get("training_status")),
        "active_model_version": existing.get("active_model_version") if existing.get("active_model") else result.get("active_model_version"),
        "latest_train_status": result.get("training_status"),
        "latest_train_registry_status": result.get("model_registry_status"),
        "latest_train_dataset_path": result.get("dataset_path"),
        "latest_train_experiment_id": result.get("train_experiment_id"),
        "best_threshold": result.get("best_threshold", 0.60),
        "row_count": result.get("row_count", 0),
        "labeled_row_count": result.get("labeled_rows", 0),
        "date_range": {
            "min_snapshot_date": (result.get("date_range") or {}).get("min"),
            "max_snapshot_date": (result.get("date_range") or {}).get("max"),
        },
        "models": {
            "outperform_5d": {
                "path": str(XGB_OUTPERFORM_MODEL_PATH),
                "metrics": (result.get("metrics") or {}).get("outperform_5d", {}),
            },
            "return_5d": {
                "path": str(XGB_RETURN_MODEL_PATH),
                "metrics": (result.get("metrics") or {}).get("return_5d", {}),
            },
            "drawdown_5d": {
                "path": str(XGB_DRAWDOWN_MODEL_PATH),
                "metrics": (result.get("metrics") or {}).get("drawdown_5d", {}),
            },
        },
        "feature_columns_path": str(FEATURE_COLUMNS_PATH),
        "order_enabled": False,
    }
    if existing.get("active_model"):
        registry["active_model"] = existing.get("active_model")
        registry["active_experiment_id"] = existing.get("active_experiment_id")
        registry["active_metrics"] = existing.get("active_metrics")
    return registry


def no_op_result(
    status: str,
    reason: str,
    row_count: int = 0,
    labeled_rows: int = 0,
    date_count: int = 0,
    dependency_check: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = utc_now()
    dependency_check = dependency_check or {}
    return {
        "generated_at": now,
        "mode": "xgboost_training",
        "training_status": "no_op",
        "model_registry_status": status,
        "active_model_version": None,
        "best_threshold": 0.60,
        "reason": reason,
        "row_count": row_count,
        "labeled_rows": labeled_rows,
        "date_count": date_count,
        "date_range": {},
        "dataset_path": None,
        "models": {},
        "metrics": {},
        "feature_importance_top30": {},
        "ablation": {"skipped": True, "reason": reason},
        "missing_dependencies": dependency_check.get("missing_dependencies", []),
        "optional_missing_dependencies": dependency_check.get("optional_missing_dependencies", []),
        "dependency_details": dependency_check.get("dependency_details", []),
        "install_command": dependency_check.get("install_command"),
        "system_install_suggestions": dependency_check.get("system_install_suggestions", []),
        "order_enabled": False,
    }


def build_training_report(result: Dict[str, Any]) -> str:
    lines = [
        "# XGBoost 모델 학습 리포트",
        "",
        f"- 생성 시각 UTC: {result.get('generated_at')}",
        "- 실제 주문 실행: 비활성화",
        "- order_enabled=false",
        f"- training_status: {result.get('training_status')}",
        f"- model_registry_status: {result.get('model_registry_status')}",
        f"- dataset_path: {result.get('dataset_path')}",
        "",
    ]
    if result.get("training_status") == "no_op":
        lines.extend([
            "## 학습 보류",
            f"- 사유: {result.get('reason')}",
            f"- 전체 row 수: {result.get('row_count', 0)}",
            f"- 학습 가능 labeled row 수: {result.get('labeled_rows', 0)}",
            f"- 학습 가능 snapshot_date 수: {result.get('date_count', 0)}",
        ])
        if result.get("missing_dependencies"):
            lines.append(f"- missing_dependencies: {result.get('missing_dependencies')}")
            lines.append(f"- install_command: {result.get('install_command')}")
        if result.get("optional_missing_dependencies"):
            lines.append(f"- optional_missing_dependencies: {result.get('optional_missing_dependencies')}")
        if result.get("system_install_suggestions"):
            lines.append(f"- system_install_suggestions: {result.get('system_install_suggestions')}")
        unavailable = [
            item for item in result.get("dependency_details", [])
            if not item.get("available")
        ]
        if unavailable:
            lines.append("")
            lines.append("## Dependency Details")
            for item in unavailable:
                lines.append(f"- {item.get('name')}: {item.get('error')}")
        return "\n".join(lines)

    lines.extend([
        "## 학습 요약",
        f"- active_model_version: {result.get('active_model_version')}",
        f"- 학습 row 수: {result.get('labeled_rows')}",
        f"- date_range: {result.get('date_range')}",
        f"- best_threshold: {result.get('best_threshold')}",
        "",
        "## 검증 Metrics",
    ])
    for model_name, metrics in result.get("metrics", {}).items():
        lines.append(f"### {model_name}")
        for key, value in metrics.items():
            if key in {"folds", "threshold_tuning"}:
                continue
            lines.append(f"- {key}: {value}")
        if model_name == "outperform_5d" and metrics.get("threshold_tuning"):
            lines.append(f"- threshold_tuning: {metrics.get('threshold_tuning')}")

    lines.append("")
    lines.append("## Feature Ablation")
    lines.append(json.dumps(result.get("ablation", {}), ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("## Feature Importance Top 30")
    for model_name, items in result.get("feature_importance_top30", {}).items():
        lines.append(f"### {model_name}")
        for item in items[:30]:
            lines.append(f"- {item.get('feature')}: {item.get('importance')}")
    lines.append("")
    lines.append("ML 신호는 주문 확정이 아니라 후보 scoring 보조 지표입니다.")
    return "\n".join(lines)


def build_quality_report(result: Dict[str, Any]) -> str:
    lines = [
        "# Model Quality Report",
        "",
        f"- generated_at: {utc_now()}",
        "- order_enabled=false",
        f"- model_status: {result.get('model_registry_status')}",
        f"- training_status: {result.get('training_status')}",
        f"- best_threshold: {result.get('best_threshold')}",
        "",
        "## 품질 판단",
    ]
    if result.get("training_status") != "trained":
        lines.append(f"- 모델 품질 검증 보류: {result.get('reason')}")
    else:
        class_metrics = (result.get("metrics") or {}).get("outperform_5d", {})
        lines.append(f"- roc_auc: {class_metrics.get('roc_auc')}")
        lines.append(f"- average_precision: {class_metrics.get('average_precision')}")
        lines.append(f"- precision_at_3: {class_metrics.get('precision_at_3')}")
        lines.append(f"- mean_future_return_top3: {class_metrics.get('mean_future_return_top3')}")
    lines.append("- 모델 성능이 낮거나 데이터가 부족하면 주문 후보를 강화하지 않습니다.")
    return "\n".join(lines)


def write_outputs(result: Dict[str, Any]) -> None:
    save_json(MODEL_TRAINING_METRICS_PATH, result)
    MODEL_TRAINING_REPORT_PATH.write_text(build_training_report(result), encoding="utf-8")
    registry = registry_from_result(result)
    save_json(MODEL_REGISTRY_PATH, registry)
    quality = {
        "generated_at": utc_now(),
        "model_status": registry.get("model_status"),
        "training_status": result.get("training_status"),
        "best_threshold": registry.get("best_threshold"),
        "quality_gate": "not_ready" if result.get("training_status") != "trained" else "review_required",
        "metrics": result.get("metrics", {}),
        "order_enabled": False,
    }
    save_json(MODEL_QUALITY_METRICS_PATH, quality)
    MODEL_QUALITY_REPORT_PATH.write_text(build_quality_report(result), encoding="utf-8")


def train(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_project_dirs()
    dataset_path = resolve_dataset_path(args)
    df = read_dataset(args)
    if df.empty:
        result = no_op_result("no_op_insufficient_data", f"{dataset_path} 파일이 없거나 비어 있습니다.")
        result["dataset_path"] = str(dataset_path)
        return result

    labeled = labeled_frame(df)
    labeled_rows = len(labeled)
    date_count = labeled["snapshot_date"].nunique() if not labeled.empty and "snapshot_date" in labeled.columns else 0

    audit = load_json(DATASET_AUDIT_JSON_PATH, {}) if not args.use_historical and not args.dataset_path else {}
    if audit.get("warning_level") == "BLOCK_TRAINING" and not args.force:
        result = no_op_result(
            "blocked_by_audit",
            "dataset audit가 BLOCK_TRAINING 상태입니다: " + "; ".join(audit.get("blocking_reasons", [])[:5]),
            row_count=len(df),
            labeled_rows=labeled_rows,
            date_count=date_count,
        )
        result["dataset_path"] = str(dataset_path)
        return result

    if not args.force and labeled_rows < args.min_labeled_rows:
        result = no_op_result(
            "no_op_insufficient_data",
            f"학습 가능 labeled row 수가 {labeled_rows}개로 min_labeled_rows={args.min_labeled_rows}보다 적습니다.",
            row_count=len(df),
            labeled_rows=labeled_rows,
            date_count=date_count,
        )
        result["dataset_path"] = str(dataset_path)
        return result
    if not args.force and date_count < args.min_dates:
        result = no_op_result(
            "no_op_insufficient_data",
            f"snapshot_date 수가 {date_count}개로 min_dates={args.min_dates}보다 적습니다.",
            row_count=len(df),
            labeled_rows=labeled_rows,
            date_count=date_count,
        )
        result["dataset_path"] = str(dataset_path)
        return result
    if labeled_rows < 8 or date_count < 3:
        result = no_op_result(
            "no_op_insufficient_data",
            "walk-forward 검증에 필요한 최소 row/date가 부족합니다.",
            row_count=len(df),
            labeled_rows=labeled_rows,
            date_count=date_count,
        )
        result["dataset_path"] = str(dataset_path)
        return result

    dependency_check = ml_dependency_check()
    if not dependency_check.get("ok"):
        dependency_reason = xgboost_unavailable_reason() or "ML dependency를 사용할 수 없습니다."
        result = no_op_result(
            "no_op_dependency_unavailable",
            dependency_reason,
            len(df),
            labeled_rows,
            date_count,
            dependency_check=dependency_check,
        )
        result["dataset_path"] = str(dataset_path)
        return result

    numeric_columns, categorical_columns, numeric_medians = choose_feature_columns(labeled, args.max_features)
    X, encoded_columns = build_matrix(labeled, numeric_columns, categorical_columns, numeric_medians)
    if X.empty or not encoded_columns:
        result = no_op_result("no_op_insufficient_data", "사용 가능한 feature column이 없습니다.", len(df), labeled_rows, date_count)
        result["dataset_path"] = str(dataset_path)
        return result

    splits = walk_forward_splits(labeled, args.n_splits, args.test_size_days, args.gap_days)
    if not splits:
        result = no_op_result("no_op_insufficient_data", "walk-forward split을 만들 수 없습니다.", len(df), labeled_rows, date_count)
        result["dataset_path"] = str(dataset_path)
        return result

    date_range = {"min": str(labeled["snapshot_date"].min()), "max": str(labeled["snapshot_date"].max())}
    active_version = model_version()
    train_experiment_id = f"train_{active_version}"
    train_experiment_dir = MODEL_EXPERIMENTS_DIR / train_experiment_id
    train_experiment_dir.mkdir(parents=True, exist_ok=True)
    metrics: Dict[str, Any] = {}
    models: Dict[str, str] = {}
    experiment_models: Dict[str, str] = {}
    importances: Dict[str, Any] = {}

    y_class = labeled[TARGET_COLUMNS["classification"]].astype(int)
    classifier, class_metrics, _ = fit_classifier(X, y_class, labeled, splits, args.random_state)
    metrics["outperform_5d"] = class_metrics
    threshold_info = class_metrics.get("threshold_tuning", {})
    best_threshold = threshold_info.get("best_threshold", 0.60)
    if classifier is not None:
        classifier.save_model(XGB_OUTPERFORM_MODEL_PATH)
        classifier.save_model(train_experiment_dir / "xgb_outperform_5d.json")
        models["outperform_5d"] = str(XGB_OUTPERFORM_MODEL_PATH)
        experiment_models["outperform_5d"] = str(train_experiment_dir / "xgb_outperform_5d.json")
        importances["outperform_5d"] = feature_importance(classifier, encoded_columns)
        save_model_metadata(XGB_OUTPERFORM_METADATA_PATH, "xgb_outperform_5d", class_metrics, labeled_rows, date_range)

    y_return = labeled[TARGET_COLUMNS["return_regression"]].astype(float)
    return_model, return_metrics = fit_regressor(X, y_return, splits, args.random_state)
    return_model.save_model(XGB_RETURN_MODEL_PATH)
    return_model.save_model(train_experiment_dir / "xgb_return_5d.json")
    models["return_5d"] = str(XGB_RETURN_MODEL_PATH)
    experiment_models["return_5d"] = str(train_experiment_dir / "xgb_return_5d.json")
    metrics["return_5d"] = return_metrics
    importances["return_5d"] = feature_importance(return_model, encoded_columns)
    save_model_metadata(XGB_RETURN_METADATA_PATH, "xgb_return_5d", return_metrics, labeled_rows, date_range)

    if "future_max_drawdown_5d" in labeled.columns and labeled["future_max_drawdown_5d"].notna().sum() >= max(8, args.n_splits + 3):
        drawdown_data = labeled.loc[labeled["future_max_drawdown_5d"].notna()].copy()
        X_drawdown, _ = build_matrix(drawdown_data, numeric_columns, categorical_columns, numeric_medians, encoded_columns)
        drawdown_splits = walk_forward_splits(drawdown_data, args.n_splits, args.test_size_days, args.gap_days)
        if drawdown_splits:
            y_drawdown = drawdown_data["future_max_drawdown_5d"].astype(float)
            drawdown_model, drawdown_metrics = fit_regressor(X_drawdown, y_drawdown, drawdown_splits, args.random_state)
            drawdown_model.save_model(XGB_DRAWDOWN_MODEL_PATH)
            drawdown_model.save_model(train_experiment_dir / "xgb_drawdown_5d.json")
            models["drawdown_5d"] = str(XGB_DRAWDOWN_MODEL_PATH)
            experiment_models["drawdown_5d"] = str(train_experiment_dir / "xgb_drawdown_5d.json")
            metrics["drawdown_5d"] = drawdown_metrics
            importances["drawdown_5d"] = feature_importance(drawdown_model, encoded_columns)
            save_model_metadata(XGB_DRAWDOWN_METADATA_PATH, "xgb_drawdown_5d", drawdown_metrics, len(drawdown_data), date_range)
        else:
            metrics["drawdown_5d"] = {"skipped": True, "reason": "drawdown walk-forward split unavailable"}
    else:
        metrics["drawdown_5d"] = {"skipped": True, "reason": "future_max_drawdown_5d labeled rows are insufficient"}

    feature_columns = {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_medians": numeric_medians,
        "encoded_columns": encoded_columns,
        "target_columns": TARGET_COLUMNS,
        "trained_at": utc_now(),
        "row_count": labeled_rows,
        "date_range": date_range,
        "active_model_version": active_version,
    }
    save_json(FEATURE_COLUMNS_PATH, feature_columns)
    save_json(train_experiment_dir / "feature_columns.json", feature_columns)

    ablation = ablation_summary(args, labeled, numeric_columns, categorical_columns)
    return {
        "generated_at": utc_now(),
        "mode": "xgboost_training",
        "dataset_path": str(dataset_path),
        "training_status": "trained" if models else "no_op",
        "model_registry_status": "trained" if models else "no_op_insufficient_data",
        "active_model_version": active_version if models else None,
        "train_experiment_id": train_experiment_id if models else None,
        "best_threshold": best_threshold,
        "row_count": len(df),
        "labeled_rows": labeled_rows,
        "date_count": date_count,
        "date_range": date_range,
        "models": models,
        "experiment_artifacts": {
            "experiment_dir": str(train_experiment_dir),
            "models": experiment_models,
            "feature_columns_path": str(train_experiment_dir / "feature_columns.json"),
        } if models else {},
        "metrics": metrics,
        "feature_importance_top30": importances,
        "ablation": ablation,
        "feature_columns_path": str(FEATURE_COLUMNS_PATH),
        "missing_dependencies": [],
        "optional_missing_dependencies": dependency_check.get("optional_missing_dependencies", []),
        "dependency_details": dependency_check.get("dependency_details", []),
        "order_enabled": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and validate XGBoost models from labeled model_dataset.csv.")
    parser.add_argument("--dataset-path")
    parser.add_argument("--use-historical", action="store_true")
    parser.add_argument("--min-labeled-rows", type=int, default=100)
    parser.add_argument("--min-dates", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--test-size-days", type=int, default=5)
    parser.add_argument("--gap-days", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--max-features", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = train(args)
    write_outputs(result)
    print(f"Saved training metrics: {MODEL_TRAINING_METRICS_PATH}")
    print(f"Saved training report: {MODEL_TRAINING_REPORT_PATH}")
    print(f"Saved model registry: {MODEL_REGISTRY_PATH}")
    print(f"training_status={result.get('training_status')}")
    print(f"model_status={result.get('model_registry_status')}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
