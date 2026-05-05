#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from _bootstrap import add_src_to_path

add_src_to_path()

from policylink.paths import (
    HISTORICAL_DATASET_AUDIT_JSON_PATH,
    HISTORICAL_DATASET_AUDIT_REPORT_PATH,
    HISTORICAL_MODEL_DATASET_CSV_PATH,
    ensure_project_dirs,
)


LEAKAGE_TOKENS = ["future", "target", "next", "tomorrow"]
ALLOWED_TARGET_COLUMNS = {
    "future_return_1d",
    "future_return_5d",
    "future_return_20d",
    "future_outperform_5d",
    "future_max_drawdown_5d",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def top_missing_ratios(df, limit: int = 30) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    ratios = df.isna().mean().sort_values(ascending=False).head(limit)
    return [{"column": str(column), "missing_ratio": round(float(value), 6)} for column, value in ratios.items()]


def leakage_candidates(columns: List[str]) -> List[str]:
    result = []
    for column in columns:
        lowered = column.lower()
        if column in ALLOWED_TARGET_COLUMNS:
            continue
        if any(token in lowered for token in LEAKAGE_TOKENS):
            result.append(column)
    return result


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    import pandas as pd

    if not HISTORICAL_MODEL_DATASET_CSV_PATH.exists():
        return {
            "generated_at": utc_now(),
            "audit_status": "missing_dataset",
            "can_train": False,
            "blocking_reasons": [f"{HISTORICAL_MODEL_DATASET_CSV_PATH} 파일이 없습니다."],
            "order_enabled": False,
        }

    df = pd.read_csv(HISTORICAL_MODEL_DATASET_CSV_PATH)
    row_count = int(len(df))
    stock_count = int(df["stock_code"].nunique()) if "stock_code" in df.columns else 0
    date_count = int(df["snapshot_date"].nunique()) if "snapshot_date" in df.columns else 0
    labeled_mask = pd.Series([False] * len(df))
    if {"future_return_5d", "future_outperform_5d"}.issubset(df.columns):
        labeled_mask = pd.to_numeric(df["future_return_5d"], errors="coerce").notna()
        labeled_mask &= pd.to_numeric(df["future_outperform_5d"], errors="coerce").notna()
    labeled_rows = int(labeled_mask.sum())
    labeled = df.loc[labeled_mask].copy()

    class_balance: Dict[str, int] = {}
    if not labeled.empty and "future_outperform_5d" in labeled.columns:
        counts = pd.to_numeric(labeled["future_outperform_5d"], errors="coerce").value_counts(dropna=False).to_dict()
        class_balance = {str(key): int(value) for key, value in counts.items()}

    sector_counts = {}
    if "sector" in df.columns:
        sector_counts = {str(key): int(value) for key, value in df["sector"].fillna("missing").value_counts().to_dict().items()}

    coverage_by_date = {}
    if {"snapshot_date", "stock_code"}.issubset(df.columns):
        coverage_by_date = {
            str(date): int(count)
            for date, count in df.groupby("snapshot_date")["stock_code"].nunique().describe().to_dict().items()
        }

    missing = top_missing_ratios(df)
    leakage = leakage_candidates(list(df.columns))
    blocking_reasons = []
    warnings = []

    if row_count < args.min_rows:
        blocking_reasons.append(f"row 수 {row_count} < {args.min_rows}")
    if labeled_rows < args.min_labeled_rows:
        blocking_reasons.append(f"labeled row 수 {labeled_rows} < {args.min_labeled_rows}")
    if date_count < args.min_dates:
        blocking_reasons.append(f"snapshot_date 수 {date_count} < {args.min_dates}")
    if stock_count < args.min_stocks:
        blocking_reasons.append(f"stock 수 {stock_count} < {args.min_stocks}")
    if len(class_balance) < 2:
        blocking_reasons.append("future_outperform_5d class가 한쪽만 존재하거나 없습니다.")
    if leakage:
        warnings.append("feature 후보에 target/leakage 의심 컬럼명이 있습니다. train_model.py에서 제외되어야 합니다.")

    status = "PASS" if not blocking_reasons else "BLOCK_TRAINING"
    return {
        "generated_at": utc_now(),
        "audit_status": status,
        "can_train": not blocking_reasons,
        "row_count": row_count,
        "stock_count": stock_count,
        "snapshot_date_count": date_count,
        "labeled_row_count": labeled_rows,
        "class_balance": class_balance,
        "sector_counts": sector_counts,
        "date_universe_coverage_summary": coverage_by_date,
        "missing_ratio_top30": missing,
        "leakage_candidate_columns": leakage,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "criteria": {
            "rows": args.min_rows,
            "dates": args.min_dates,
            "stocks": args.min_stocks,
            "labeled_rows": args.min_labeled_rows,
        },
        "order_enabled": False,
    }


def build_report(result: Dict[str, Any]) -> str:
    lines = [
        "# Historical Dataset Audit Report",
        "",
        f"- generated_at: {result.get('generated_at')}",
        "- 실제 주문 실행: 비활성화",
        "- order_enabled=false",
        f"- audit_status: {result.get('audit_status')}",
        f"- can_train: {result.get('can_train')}",
        f"- row_count: {result.get('row_count', 0)}",
        f"- labeled_row_count: {result.get('labeled_row_count', 0)}",
        f"- snapshot_date_count: {result.get('snapshot_date_count', 0)}",
        f"- stock_count: {result.get('stock_count', 0)}",
        "",
        "## Class Balance",
    ]
    for key, value in (result.get("class_balance") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Blocking Reasons"])
    reasons = result.get("blocking_reasons") or []
    if reasons:
        for reason in reasons:
            lines.append(f"- {reason}")
    else:
        lines.append("- 없음")
    lines.extend(["", "## Leakage Warning"])
    leakage = result.get("leakage_candidate_columns") or []
    if leakage:
        for column in leakage[:30]:
            lines.append(f"- {column}")
    else:
        lines.append("- 컬럼명 기준 leakage 의심 항목 없음")
    lines.extend(["", "## Top Missing Ratio"])
    for item in (result.get("missing_ratio_top30") or [])[:15]:
        lines.append(f"- {item.get('column')}: {item.get('missing_ratio')}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit historical point-in-time model dataset.")
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--min-labeled-rows", type=int, default=100)
    parser.add_argument("--min-dates", type=int, default=60)
    parser.add_argument("--min-stocks", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_project_dirs()
    result = audit(args)
    save_json(HISTORICAL_DATASET_AUDIT_JSON_PATH, result)
    HISTORICAL_DATASET_AUDIT_REPORT_PATH.write_text(build_report(result), encoding="utf-8")
    print(f"Saved historical audit json: {HISTORICAL_DATASET_AUDIT_JSON_PATH}")
    print(f"Saved historical audit report: {HISTORICAL_DATASET_AUDIT_REPORT_PATH}")
    print(f"audit_status={result.get('audit_status')}")
    print(f"can_train={result.get('can_train')}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
