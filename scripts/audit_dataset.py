#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from _bootstrap import add_src_to_path

add_src_to_path()

from policylink.paths import DATASET_AUDIT_JSON_PATH, DATASET_AUDIT_REPORT_PATH, MODEL_DATASET_CSV_PATH


LABEL_STATUSES = ["labeled", "partially_labeled", "pending_future_prices", "unlabeled"]
LEAKAGE_TOKENS = ["future", "target", "label", "next", "tomorrow"]
DATE_LEAKAGE_TOKENS = ["after", "post", "forward", "realized", "actual"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_dataset():
    import pandas as pd

    if not MODEL_DATASET_CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MODEL_DATASET_CSV_PATH)


def numeric_summary(series) -> Dict[str, Optional[float]]:
    import pandas as pd

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {"mean": None, "median": None, "std": None}
    return {
        "mean": round(float(numeric.mean()), 6),
        "median": round(float(numeric.median()), 6),
        "std": round(float(numeric.std()), 6) if len(numeric) > 1 else 0.0,
    }


def parse_date_series(series):
    import pandas as pd

    text = series.astype(str).str.replace("-", "", regex=False)
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def missing_feature_presence(df, prefix: str) -> Dict[str, Any]:
    columns = [column for column in df.columns if column.startswith(prefix)]
    if not columns:
        return {"column_count": 0, "all_missing_rows": len(df), "columns": []}
    all_missing = df[columns].isna().all(axis=1).sum()
    return {"column_count": len(columns), "all_missing_rows": int(all_missing), "columns": columns[:30]}


def audit_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    import pandas as pd

    df = read_dataset()
    generated_at = utc_now()
    if df.empty:
        return {
            "generated_at": generated_at,
            "mode": "dataset_audit",
            "warning_level": "BLOCK_TRAINING",
            "can_train": False,
            "blocking_reasons": ["data/model_dataset.csv가 없거나 비어 있습니다."],
            "row_count": 0,
            "order_enabled": False,
        }

    row_count = len(df)
    label_counts = {
        status: int((df.get("label_status") == status).sum()) if "label_status" in df.columns else 0
        for status in LABEL_STATUSES
    }
    labeled_mask = df.get("label_status", pd.Series([""] * row_count)).isin(["labeled", "partially_labeled"])
    labeled_mask &= pd.to_numeric(df.get("future_return_5d"), errors="coerce").notna() if "future_return_5d" in df.columns else False
    labeled_mask &= pd.to_numeric(df.get("future_outperform_5d"), errors="coerce").notna() if "future_outperform_5d" in df.columns else False
    labeled = df.loc[labeled_mask].copy()

    snapshot_dates = int(df["snapshot_date"].nunique()) if "snapshot_date" in df.columns else 0
    labeled_dates = int(labeled["snapshot_date"].nunique()) if "snapshot_date" in labeled.columns else 0
    stock_count = int(df["stock_code"].nunique()) if "stock_code" in df.columns else 0
    sector_distribution = df["sector"].fillna("missing").value_counts().to_dict() if "sector" in df.columns else {}

    class_balance = {}
    if "future_outperform_5d" in labeled.columns:
        class_balance = (
            pd.to_numeric(labeled["future_outperform_5d"], errors="coerce")
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
        )
        class_balance = {str(key): int(value) for key, value in class_balance.items()}

    missing_ratio = []
    for column in df.columns:
        missing_ratio.append({"column": column, "missing_ratio": round(float(df[column].isna().mean()), 6)})
    missing_ratio.sort(key=lambda x: x["missing_ratio"], reverse=True)

    constant_features = [
        column for column in df.columns
        if column not in {"snapshot_date", "stock_code"} and df[column].nunique(dropna=False) <= 1
    ]

    high_cardinality = []
    for column in df.select_dtypes(include=["object"]).columns:
        unique_count = int(df[column].nunique(dropna=True))
        if unique_count > min(30, max(10, row_count // 2)):
            high_cardinality.append({"column": column, "unique_count": unique_count})

    duplicate_count = 0
    duplicate_examples: List[Dict[str, Any]] = []
    if {"snapshot_date", "stock_code"}.issubset(df.columns):
        duplicate_mask = df.duplicated(["snapshot_date", "stock_code"], keep=False)
        duplicate_count = int(duplicate_mask.sum())
        duplicate_examples = df.loc[duplicate_mask, ["snapshot_date", "stock_code"]].head(10).to_dict("records")

    leakage_candidates = [
        column for column in df.columns
        if any(token in column.lower() for token in LEAKAGE_TOKENS)
    ]
    date_leakage_candidates = [
        column for column in df.columns
        if any(token in column.lower() for token in DATE_LEAKAGE_TOKENS)
    ]

    freshness = {"stale_row_count": 0, "missing_date_row_count": 0}
    if {"latest_date", "snapshot_date"}.issubset(df.columns):
        latest = parse_date_series(df["latest_date"])
        snapshot = parse_date_series(df["snapshot_date"])
        diff_days = (snapshot - latest).dt.days
        freshness = {
            "stale_row_count": int((diff_days > 5).sum()),
            "missing_date_row_count": int(diff_days.isna().sum()),
            "max_lag_days": None if diff_days.dropna().empty else int(diff_days.max()),
        }

    blocking_reasons: List[str] = []
    warnings: List[str] = []

    if row_count < args.min_rows:
        warnings.append(f"전체 row 수가 {row_count}개로 min_rows={args.min_rows}보다 적습니다.")
    if len(labeled) < args.min_labeled_rows:
        blocking_reasons.append(f"labeled rows {len(labeled)}개 < min_labeled_rows={args.min_labeled_rows}")
    if labeled_dates < args.min_dates:
        blocking_reasons.append(f"labeled snapshot_date {labeled_dates}개 < min_dates={args.min_dates}")
    if len(class_balance) <= 1:
        blocking_reasons.append("future_outperform_5d가 한 클래스만 있거나 라벨이 없습니다.")
    if duplicate_count > max(5, row_count * 0.05):
        blocking_reasons.append("snapshot_date+stock_code 중복이 심각합니다.")
    if leakage_candidates:
        warnings.append("컬럼명 기준 leakage 후보가 있습니다. train_model.py는 future/target/label 계열 컬럼을 feature에서 제외합니다.")

    warning_level = "OK"
    if warnings:
        warning_level = "WARN"
    if blocking_reasons:
        warning_level = "BLOCK_TRAINING"

    return {
        "generated_at": generated_at,
        "mode": "dataset_audit",
        "warning_level": warning_level,
        "can_train": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "row_count": row_count,
        "labeled_row_count": int(len(labeled)),
        "label_status_counts": label_counts,
        "snapshot_date_count": snapshot_dates,
        "labeled_snapshot_date_count": labeled_dates,
        "stock_code_count": stock_count,
        "sector_distribution": {str(key): int(value) for key, value in sector_distribution.items()},
        "future_outperform_5d_class_balance": class_balance,
        "future_return_5d_summary": numeric_summary(labeled["future_return_5d"]) if "future_return_5d" in labeled.columns else {},
        "feature_missing_ratio_top30": missing_ratio[:30],
        "constant_features": constant_features[:100],
        "high_cardinality_categorical_features": high_cardinality,
        "duplicate_key": {
            "duplicate_row_count": duplicate_count,
            "examples": duplicate_examples,
        },
        "leakage_candidate_columns": leakage_candidates,
        "date_leakage_candidate_columns": date_leakage_candidates,
        "price_data_freshness": freshness,
        "feature_presence": {
            "dart": missing_feature_presence(df, "dart_"),
            "naver": missing_feature_presence(df, "naver_"),
            "yahoo": missing_feature_presence(df, "yahoo_"),
        },
        "order_enabled": False,
    }


def build_markdown(result: Dict[str, Any]) -> str:
    lines = [
        "# Dataset Audit Report",
        "",
        f"- 생성 시각 UTC: {result.get('generated_at')}",
        "- 실제 주문 실행: 비활성화",
        "- order_enabled=false",
        f"- 데이터셋 상태: {result.get('warning_level')}",
        f"- 학습 가능 여부: {'yes' if result.get('can_train') else 'no'}",
        "",
        "## 라벨 상태",
        f"- 전체 row 수: {result.get('row_count', 0)}",
        f"- 학습 가능 labeled row 수: {result.get('labeled_row_count', 0)}",
        f"- snapshot_date 수: {result.get('snapshot_date_count', 0)}",
        f"- labeled snapshot_date 수: {result.get('labeled_snapshot_date_count', 0)}",
        f"- stock_code 수: {result.get('stock_code_count', 0)}",
        f"- label_status_counts: {result.get('label_status_counts', {})}",
        "",
        "## 학습 차단 사유",
    ]

    if result.get("blocking_reasons"):
        for reason in result["blocking_reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("- 없음")

    lines.extend(["", "## 누락 피처"])
    for key, item in result.get("feature_presence", {}).items():
        lines.append(f"- {key}: columns={item.get('column_count')} / all_missing_rows={item.get('all_missing_rows')}")

    lines.extend([
        "",
        "## Class Balance",
        f"- future_outperform_5d: {result.get('future_outperform_5d_class_balance', {})}",
        f"- future_return_5d summary: {result.get('future_return_5d_summary', {})}",
        "",
        "## Leakage Warning",
    ])
    leakage = result.get("leakage_candidate_columns", [])
    if leakage:
        lines.append(f"- leakage 후보 컬럼: {', '.join(leakage[:30])}")
        lines.append("- train_model.py는 future/target/label 계열 컬럼을 feature에서 제외합니다.")
    else:
        lines.append("- 컬럼명 기준 leakage 후보 없음")

    lines.extend([
        "",
        "## 다음 조치",
        "- 라벨이 부족하면 며칠간 스냅샷을 더 누적하고 label_dataset.py를 재실행합니다.",
        "- BLOCK_TRAINING 상태에서는 모델 성능을 신뢰하지 않고 주문 후보 강화에 쓰지 않습니다.",
        "- 중복 key와 stale price row가 늘어나면 데이터 생성 순서를 점검합니다.",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit model_dataset.csv for training quality and leakage risk.")
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument("--min-labeled-rows", type=int, default=30)
    parser.add_argument("--min-dates", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_dataset(args)
    save_json(DATASET_AUDIT_JSON_PATH, result)
    DATASET_AUDIT_REPORT_PATH.write_text(build_markdown(result), encoding="utf-8")
    print(f"Saved dataset audit json: {DATASET_AUDIT_JSON_PATH}")
    print(f"Saved dataset audit report: {DATASET_AUDIT_REPORT_PATH}")
    print(f"warning_level={result.get('warning_level')}")
    print(f"can_train={str(result.get('can_train')).lower()}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
