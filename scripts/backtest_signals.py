#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from _bootstrap import add_src_to_path

add_src_to_path()

from policylink.paths import BACKTEST_METRICS_PATH, BACKTEST_REPORT_PATH, HISTORICAL_MODEL_DATASET_CSV_PATH, ML_SIGNALS_JSON_PATH, MODEL_DATASET_CSV_PATH, MODEL_REGISTRY_PATH
from policylink.utils import load_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


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


def no_op(reason: str, labeled_rows: int = 0) -> Dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "mode": "signal_backtest",
        "backtest_status": "no_op",
        "reason": reason,
        "labeled_rows": labeled_rows,
        "metrics": {},
        "order_enabled": False,
    }


def labeled_frame(df, target_column: str, score_column: str):
    import pandas as pd

    if df.empty or target_column not in df.columns or score_column not in df.columns:
        return df.iloc[0:0].copy()
    mask = df.get("label_status", pd.Series([""] * len(df))).isin(["labeled", "partially_labeled"])
    mask &= pd.to_numeric(df[target_column], errors="coerce").notna()
    mask &= pd.to_numeric(df[score_column], errors="coerce").notna()
    result = df.loc[mask].copy()
    result[target_column] = pd.to_numeric(result[target_column], errors="coerce")
    result[score_column] = pd.to_numeric(result[score_column], errors="coerce")
    if "future_max_drawdown_5d" in result.columns:
        result["future_max_drawdown_5d"] = pd.to_numeric(result["future_max_drawdown_5d"], errors="coerce")
    return result.sort_values(["snapshot_date", score_column])


def turnover_proxy(selected_by_date: Dict[str, set]) -> float:
    dates = sorted(selected_by_date.keys())
    if len(dates) <= 1:
        return 0.0
    changes = []
    for prev, cur in zip(dates, dates[1:]):
        prev_set = selected_by_date[prev]
        cur_set = selected_by_date[cur]
        if not prev_set and not cur_set:
            changes.append(0.0)
            continue
        overlap = len(prev_set & cur_set)
        base = max(len(prev_set), len(cur_set), 1)
        changes.append(1.0 - overlap / base)
    return round(sum(changes) / len(changes), 6)


def run_backtest(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = resolve_dataset_path(args)
    df = read_dataset(args)
    labeled = labeled_frame(df, args.target_column, args.score_column)
    if len(labeled) < args.min_labeled_rows:
        result = no_op(
            f"labeled rows {len(labeled)}개 < min_labeled_rows={args.min_labeled_rows}",
            labeled_rows=len(labeled),
        )
        result["dataset_path"] = str(dataset_path)
        return result

    selected_parts = []
    selected_by_date: Dict[str, set] = {}
    for snapshot_date, group in labeled.groupby("snapshot_date"):
        top = group.sort_values(args.score_column, ascending=False).head(args.top_k)
        selected_parts.append(top)
        selected_by_date[str(snapshot_date)] = set(top.get("stock_code", []))

    if not selected_parts:
        result = no_op("snapshot_date별 top_k 선택 결과가 없습니다.", labeled_rows=len(labeled))
        result["dataset_path"] = str(dataset_path)
        return result

    import pandas as pd

    selected = pd.concat(selected_parts)
    target = labeled[args.target_column]
    top_target = selected[args.target_column]
    drawdown = selected["future_max_drawdown_5d"] if "future_max_drawdown_5d" in selected.columns else None

    metrics = {
        "score_column": args.score_column,
        "target_column": args.target_column,
        "top_k": args.top_k,
        "date_count": int(labeled["snapshot_date"].nunique()) if "snapshot_date" in labeled.columns else 0,
        "labeled_rows": int(len(labeled)),
        "selected_rows": int(len(selected)),
        "top_k_mean_return": round(float(top_target.mean()), 6),
        "all_mean_return": round(float(target.mean()), 6),
        "excess_return": round(float(top_target.mean() - target.mean()), 6),
        "hit_rate": round(float((target > 0).mean()), 6),
        "top_k_hit_rate": round(float((top_target > 0).mean()), 6),
        "top_k_avg_drawdown": round(float(drawdown.mean()), 6) if drawdown is not None and drawdown.notna().any() else None,
        "turnover_proxy": turnover_proxy(selected_by_date),
    }

    registry = load_json(MODEL_REGISTRY_PATH, {})
    ml_signals = load_json(ML_SIGNALS_JSON_PATH, {})
    return {
        "generated_at": utc_now(),
        "mode": "signal_backtest",
        "dataset_path": str(dataset_path),
        "backtest_status": "completed",
        "model_status": registry.get("model_status"),
        "ml_signal_status": ml_signals.get("model_status"),
        "metrics": metrics,
        "order_enabled": False,
    }


def build_report(result: Dict[str, Any]) -> str:
    lines = [
        "# Backtest Report",
        "",
        f"- generated_at: {result.get('generated_at')}",
        "- 실제 주문 실행: 비활성화",
        "- order_enabled=false",
        f"- backtest_status: {result.get('backtest_status')}",
        f"- dataset_path: {result.get('dataset_path')}",
        "",
    ]
    if result.get("backtest_status") == "no_op":
        lines.append("## 백테스트 보류")
        lines.append(f"- 사유: {result.get('reason')}")
        lines.append(f"- labeled_rows: {result.get('labeled_rows', 0)}")
        return "\n".join(lines)

    lines.append("## 지표")
    for key, value in result.get("metrics", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend([
        "",
        "## 주의사항",
        "- 이 백테스트는 단순 top-k 검증이며 체결, 세금, 수수료, 슬리피지를 반영하지 않습니다.",
        "- 주문 실행 기능은 포함하지 않습니다.",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest rule/ML scores on labeled model_dataset rows.")
    parser.add_argument("--dataset-path")
    parser.add_argument("--use-historical", action="store_true")
    parser.add_argument("--score-column", default="final_score")
    parser.add_argument("--target-column", default="future_return_5d")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-labeled-rows", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_backtest(args)
    save_json(BACKTEST_METRICS_PATH, result)
    BACKTEST_REPORT_PATH.write_text(build_report(result), encoding="utf-8")
    print(f"Saved backtest metrics: {BACKTEST_METRICS_PATH}")
    print(f"Saved backtest report: {BACKTEST_REPORT_PATH}")
    print(f"backtest_status={result.get('backtest_status')}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
