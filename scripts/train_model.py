#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from _bootstrap import add_src_to_path

add_src_to_path()

from policylink.paths import (
    FEATURE_COLUMNS_PATH,
    MODEL_DATASET_CSV_PATH,
    MODEL_TRAINING_METRICS_PATH,
    MODEL_TRAINING_REPORT_PATH,
    MODELS_DIR,
    XGB_DRAWDOWN_METADATA_PATH,
    XGB_DRAWDOWN_MODEL_PATH,
    XGB_OUTPERFORM_METADATA_PATH,
    XGB_OUTPERFORM_MODEL_PATH,
    XGB_RETURN_METADATA_PATH,
    XGB_RETURN_MODEL_PATH,
    ensure_project_dirs,
)


TARGET_COLUMNS = {
    "classification": "future_outperform_5d",
    "return_regression": "future_return_5d",
    "drawdown_regression": "future_max_drawdown_5d",
}

EXCLUDE_COLUMNS = {
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def no_op_result(reason: str, row_count: int = 0, labeled_rows: int = 0, date_count: int = 0) -> Dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "mode": "xgboost_training",
        "training_status": "no_op",
        "reason": reason,
        "row_count": row_count,
        "labeled_rows": labeled_rows,
        "date_count": date_count,
        "models": {},
        "metrics": {},
        "feature_importance_top30": {},
        "order_enabled": False,
    }


def save_json(path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_report(result: Dict[str, Any]) -> str:
    lines = [
        "# XGBoost 모델 학습 리포트",
        "",
        f"- 생성 시각 UTC: {result.get('generated_at')}",
        "- 실제 주문 실행: 비활성화",
        "- order_enabled=false",
        f"- training_status: {result.get('training_status')}",
        "",
    ]

    if result.get("training_status") == "no_op":
        lines.extend([
            "## 학습 보류",
            f"- 사유: {result.get('reason')}",
            f"- 전체 row 수: {result.get('row_count', 0)}",
            f"- 학습 가능 labeled row 수: {result.get('labeled_rows', 0)}",
            f"- 학습 가능 snapshot_date 수: {result.get('date_count', 0)}",
            "",
            "라벨이 충분히 누적되면 같은 스크립트가 XGBClassifier/XGBRegressor를 학습합니다.",
        ])
        return "\n".join(lines)

    lines.extend([
        "## 학습 요약",
        f"- 전체 row 수: {result.get('row_count', 0)}",
        f"- 학습 row 수: {result.get('labeled_rows', 0)}",
        f"- snapshot_date 수: {result.get('date_count', 0)}",
        f"- date_range: {result.get('date_range')}",
        "",
        "## 검증 Metrics",
    ])

    for model_name, metrics in result.get("metrics", {}).items():
        lines.append(f"### {model_name}")
        for key, value in metrics.items():
            if key == "folds":
                continue
            lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("## Feature Importance Top 30")
    for model_name, items in result.get("feature_importance_top30", {}).items():
        lines.append(f"### {model_name}")
        for item in items[:30]:
            lines.append(f"- {item.get('feature')}: {item.get('importance')}")

    lines.extend([
        "",
        "## 주의사항",
        "- ML 신호는 매수/매도 확정이 아니라 추천 후보 scoring 보조 지표입니다.",
        "- 모델 학습은 주문 API를 호출하지 않습니다.",
    ])
    return "\n".join(lines)


def write_result(result: Dict[str, Any]) -> None:
    save_json(MODEL_TRAINING_METRICS_PATH, result)
    MODEL_TRAINING_REPORT_PATH.write_text(build_report(result), encoding="utf-8")


def read_dataset():
    import pandas as pd

    if not MODEL_DATASET_CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MODEL_DATASET_CSV_PATH)


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


def choose_feature_columns(df) -> Tuple[List[str], List[str]]:
    import pandas as pd

    candidate_columns = [column for column in df.columns if column not in EXCLUDE_COLUMNS]
    numeric_columns: List[str] = []
    categorical_columns: List[str] = []

    for column in candidate_columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
            continue

        converted = pd.to_numeric(series, errors="coerce")
        non_null_ratio = converted.notna().mean() if len(converted) else 0
        if non_null_ratio >= 0.80:
            df[column] = converted
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return numeric_columns, categorical_columns


def build_matrix(df, numeric_columns: List[str], categorical_columns: List[str], encoded_columns: Optional[List[str]] = None):
    import pandas as pd

    parts = []
    if numeric_columns:
        numeric = df.reindex(columns=numeric_columns).apply(pd.to_numeric, errors="coerce").fillna(0.0)
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


def finite_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return round(numeric, 6)
    except Exception:
        return None


def classification_metrics(y_true, y_pred, y_prob) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    result = {
        "accuracy": finite_float(accuracy_score(y_true, y_pred)),
        "precision": finite_float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": finite_float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": finite_float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        if len(set(y_true)) > 1:
            result["roc_auc"] = finite_float(roc_auc_score(y_true, y_prob))
        else:
            result["roc_auc"] = None
    except Exception:
        result["roc_auc"] = None
    return result


def regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": finite_float(mean_absolute_error(y_true, y_pred)),
        "rmse": finite_float(math.sqrt(mse)),
        "r2": finite_float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None,
    }


def average_metrics(folds: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not folds:
        return {"fold_count": 0}

    keys = sorted({key for fold in folds for key in fold.keys() if key != "fold"})
    result: Dict[str, Any] = {"fold_count": len(folds)}
    for key in keys:
        values = [fold.get(key) for fold in folds if isinstance(fold.get(key), (int, float))]
        result[key] = finite_float(sum(values) / len(values)) if values else None
    result["folds"] = folds
    return result


def fit_classifier(X, y, n_splits: int, random_state: int) -> Tuple[Any, Dict[str, Any]]:
    from sklearn.model_selection import TimeSeriesSplit
    from xgboost import XGBClassifier

    if len(set(y)) < 2:
        return None, {"skipped": True, "reason": "classification target has one class"}

    folds = []
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        y_train = y.iloc[train_idx]
        if len(set(y_train)) < 2:
            continue

        model = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
        )
        model.fit(X.iloc[train_idx], y_train)
        y_pred = model.predict(X.iloc[test_idx])
        y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        fold_metrics = classification_metrics(y.iloc[test_idx], y_pred, y_prob)
        fold_metrics["fold"] = fold_idx
        folds.append(fold_metrics)

    final_model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
    )
    final_model.fit(X, y)
    return final_model, average_metrics(folds)


def fit_regressor(X, y, n_splits: int, random_state: int) -> Tuple[Any, Dict[str, Any]]:
    from sklearn.model_selection import TimeSeriesSplit
    from xgboost import XGBRegressor

    folds = []
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        model = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X.iloc[test_idx])
        fold_metrics = regression_metrics(y.iloc[test_idx], y_pred)
        fold_metrics["fold"] = fold_idx
        folds.append(fold_metrics)

    final_model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=random_state,
    )
    final_model.fit(X, y)
    return final_model, average_metrics(folds)


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


def save_model_metadata(path, model_name: str, metrics: Dict[str, Any], row_count: int, date_range: Dict[str, Any]) -> None:
    save_json(path, {
        "model_name": model_name,
        "trained_at": utc_now(),
        "row_count": row_count,
        "date_range": date_range,
        "metrics": metrics,
        "order_enabled": False,
    })


def train(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_project_dirs()
    df = read_dataset()
    if df.empty:
        return no_op_result("data/model_dataset.csv가 없거나 비어 있습니다.")

    labeled = labeled_frame(df)
    labeled_rows = len(labeled)
    date_count = labeled["snapshot_date"].nunique() if not labeled.empty and "snapshot_date" in labeled.columns else 0

    if not args.force and labeled_rows < args.min_labeled_rows:
        return no_op_result(
            f"학습 가능 labeled row 수가 {labeled_rows}개로 min_labeled_rows={args.min_labeled_rows}보다 적습니다.",
            row_count=len(df),
            labeled_rows=labeled_rows,
            date_count=date_count,
        )

    if not args.force and date_count < args.min_dates:
        return no_op_result(
            f"snapshot_date 수가 {date_count}개로 min_dates={args.min_dates}보다 적습니다.",
            row_count=len(df),
            labeled_rows=labeled_rows,
            date_count=date_count,
        )

    if labeled_rows < 5 or date_count < 2:
        return no_op_result(
            "TimeSeriesSplit 학습에 필요한 최소 row/date가 부족합니다.",
            row_count=len(df),
            labeled_rows=labeled_rows,
            date_count=date_count,
        )

    numeric_columns, categorical_columns = choose_feature_columns(labeled)
    X, encoded_columns = build_matrix(labeled, numeric_columns, categorical_columns)
    if X.empty or len(encoded_columns) == 0:
        return no_op_result(
            "사용 가능한 feature column이 없습니다.",
            row_count=len(df),
            labeled_rows=labeled_rows,
            date_count=date_count,
        )

    n_splits = min(args.n_splits, max(2, date_count - 1), max(2, labeled_rows - 1))
    n_splits = max(2, n_splits)
    date_range = {
        "min": str(labeled["snapshot_date"].min()),
        "max": str(labeled["snapshot_date"].max()),
    }

    metrics: Dict[str, Any] = {}
    models: Dict[str, str] = {}
    importances: Dict[str, Any] = {}

    y_class = labeled[TARGET_COLUMNS["classification"]].astype(int)
    classifier, class_metrics = fit_classifier(X, y_class, n_splits, args.random_state)
    metrics["outperform_5d"] = class_metrics
    if classifier is not None:
        classifier.save_model(XGB_OUTPERFORM_MODEL_PATH)
        models["outperform_5d"] = str(XGB_OUTPERFORM_MODEL_PATH)
        importances["outperform_5d"] = feature_importance(classifier, encoded_columns)
        save_model_metadata(XGB_OUTPERFORM_METADATA_PATH, "xgb_outperform_5d", class_metrics, labeled_rows, date_range)

    y_return = labeled[TARGET_COLUMNS["return_regression"]].astype(float)
    return_model, return_metrics = fit_regressor(X, y_return, n_splits, args.random_state)
    return_model.save_model(XGB_RETURN_MODEL_PATH)
    models["return_5d"] = str(XGB_RETURN_MODEL_PATH)
    metrics["return_5d"] = return_metrics
    importances["return_5d"] = feature_importance(return_model, encoded_columns)
    save_model_metadata(XGB_RETURN_METADATA_PATH, "xgb_return_5d", return_metrics, labeled_rows, date_range)

    if "future_max_drawdown_5d" in labeled.columns and labeled["future_max_drawdown_5d"].notna().sum() >= max(5, args.n_splits + 2):
        drawdown_data = labeled.loc[labeled["future_max_drawdown_5d"].notna()].copy()
        X_drawdown, _ = build_matrix(drawdown_data, numeric_columns, categorical_columns, encoded_columns)
        y_drawdown = drawdown_data["future_max_drawdown_5d"].astype(float)
        drawdown_splits = min(n_splits, max(2, len(drawdown_data) - 1))
        drawdown_model, drawdown_metrics = fit_regressor(X_drawdown, y_drawdown, drawdown_splits, args.random_state)
        drawdown_model.save_model(XGB_DRAWDOWN_MODEL_PATH)
        models["drawdown_5d"] = str(XGB_DRAWDOWN_MODEL_PATH)
        metrics["drawdown_5d"] = drawdown_metrics
        importances["drawdown_5d"] = feature_importance(drawdown_model, encoded_columns)
        save_model_metadata(XGB_DRAWDOWN_METADATA_PATH, "xgb_drawdown_5d", drawdown_metrics, len(drawdown_data), date_range)
    else:
        metrics["drawdown_5d"] = {"skipped": True, "reason": "future_max_drawdown_5d labeled rows are insufficient"}

    feature_columns = {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "encoded_columns": encoded_columns,
        "target_columns": TARGET_COLUMNS,
        "trained_at": utc_now(),
        "row_count": labeled_rows,
        "date_range": date_range,
    }
    save_json(FEATURE_COLUMNS_PATH, feature_columns)

    return {
        "generated_at": utc_now(),
        "mode": "xgboost_training",
        "training_status": "trained" if models else "no_op",
        "row_count": len(df),
        "labeled_rows": labeled_rows,
        "date_count": date_count,
        "date_range": date_range,
        "models": models,
        "metrics": metrics,
        "feature_importance_top30": importances,
        "feature_columns_path": str(FEATURE_COLUMNS_PATH),
        "order_enabled": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost models from labeled model_dataset.csv.")
    parser.add_argument("--min-labeled-rows", type=int, default=100)
    parser.add_argument("--min-dates", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = train(args)
    write_result(result)
    print(f"Saved training metrics: {MODEL_TRAINING_METRICS_PATH}")
    print(f"Saved training report: {MODEL_TRAINING_REPORT_PATH}")
    print(f"training_status={result.get('training_status')}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
