#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from _bootstrap import add_src_to_path

add_src_to_path()

from policylink.dataset.label import build_price_index, find_base_index, future_max_drawdown, future_return
from policylink.paths import (
    BACKFILL_PRICES_DAILY_HISTORY_PATH,
    PAPER_TRADES_JSONL_PATH,
    PAPER_TRADING_REPORT_JSON_PATH,
    PAPER_TRADING_REPORT_MD_PATH,
    PRICES_DAILY_PATH,
    ensure_project_dirs,
)
from policylink.utils import load_json, load_jsonl, normalize_code, parse_number, save_json, save_jsonl


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def merge_price_indexes(*indexes: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    merged: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for index in indexes:
        for code, rows in index.items():
            normalized = normalize_code(code)
            merged.setdefault(normalized, {})
            for row in rows:
                date = str(row.get("date") or "")
                if date:
                    merged[normalized][date] = row
    return {
        code: sorted(rows_by_date.values(), key=lambda row: str(row.get("date")))
        for code, rows_by_date in merged.items()
    }


def side_adjusted_return(side: str, raw_return: Optional[float]) -> Optional[float]:
    if raw_return is None:
        return None
    if side == "sell":
        return -float(raw_return)
    if side == "watch":
        return 0.0
    return float(raw_return)


def net_return(side: str, raw_return: Optional[float], cost_return: float) -> Optional[float]:
    adjusted = side_adjusted_return(side, raw_return)
    if adjusted is None:
        return None
    if side == "watch":
        return 0.0
    return adjusted - cost_return


def label_trade(row: Dict[str, Any], price_index: Dict[str, List[Dict[str, Any]]], cost_return: float) -> Dict[str, Any]:
    code = normalize_code(row.get("stock_code"))
    prices = price_index.get(code, [])
    base_idx = find_base_index(prices, str(row.get("snapshot_date") or ""))
    updated = dict(row)
    if base_idx is None:
        updated["label_status"] = "pending_future_prices"
        return updated

    r1 = future_return(prices, base_idx, 1)
    r5 = future_return(prices, base_idx, 5)
    r20 = future_return(prices, base_idx, 20)
    dd5 = future_max_drawdown(prices, base_idx, 5)

    updated["future_return_1d"] = None if r1 is None else round(float(r1), 6)
    updated["future_return_5d"] = None if r5 is None else round(float(r5), 6)
    updated["future_return_20d"] = None if r20 is None else round(float(r20), 6)
    updated["max_drawdown_5d"] = None if dd5 is None else round(float(dd5), 6)
    adjusted = side_adjusted_return(str(row.get("side") or ""), r5)
    net = net_return(str(row.get("side") or ""), r5, cost_return)
    updated["paper_return_5d"] = None if adjusted is None else round(float(adjusted), 6)
    updated["net_return_5d"] = None if net is None else round(float(net), 6)

    if r5 is not None:
        updated["label_status"] = "labeled"
    elif r1 is not None:
        updated["label_status"] = "partially_labeled"
    else:
        updated["label_status"] = "pending_future_prices"
    updated["label_updated_at"] = utc_now()
    return updated


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def metrics_for(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labeled = [row for row in rows if row.get("label_status") == "labeled" and row.get("net_return_5d") is not None]
    raw_returns = [parse_number(row.get("paper_return_5d"), 0.0) for row in labeled]
    net_returns = [parse_number(row.get("net_return_5d"), 0.0) for row in labeled]
    drawdowns = [parse_number(row.get("max_drawdown_5d"), 0.0) for row in labeled if row.get("max_drawdown_5d") is not None]
    return {
        "total_trades": len(rows),
        "labeled_trades": len(labeled),
        "pending_trades": len([row for row in rows if row.get("label_status") != "labeled"]),
        "buy_candidate_count": len([row for row in rows if row.get("side") == "buy"]),
        "sell_candidate_count": len([row for row in rows if row.get("side") == "sell"]),
        "watch_candidate_count": len([row for row in rows if row.get("side") == "watch"]),
        "mean_return_5d": mean(raw_returns),
        "mean_net_return_5d": mean(net_returns),
        "hit_rate_5d": mean([1.0 if value > 0 else 0.0 for value in net_returns]),
        "avg_max_drawdown_5d": mean(drawdowns),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_project_dirs()
    trades = load_jsonl(PAPER_TRADES_JSONL_PATH)
    cost_return = (float(args.fee_bps) + float(args.slippage_bps)) / 10000.0
    live_index = build_price_index(load_json(PRICES_DAILY_PATH, {"prices": {}}))
    backfill_index = build_price_index(load_json(BACKFILL_PRICES_DAILY_HISTORY_PATH, {"prices": {}}))
    price_index = merge_price_indexes(backfill_index, live_index)
    updated = [label_trade(row, price_index, cost_return) for row in trades]
    save_jsonl(PAPER_TRADES_JSONL_PATH, updated)

    metrics = metrics_for(updated)
    approved = [row for row in updated if row.get("risk_check_status") == "approved_for_review"]
    rejected = [row for row in updated if row.get("risk_check_status") == "rejected"]
    approved_metrics = metrics_for(approved)
    rejected_metrics = metrics_for(rejected)
    status = "completed" if metrics["labeled_trades"] > 0 else "no_op_insufficient_labeled_paper_trades"
    return {
        "generated_at": utc_now(),
        "mode": "paper_trading_evaluation",
        "evaluation_status": status,
        "horizon_days": args.horizon_days,
        "fee_bps": float(args.fee_bps),
        "slippage_bps": float(args.slippage_bps),
        "round_trip_cost_return": round(cost_return, 6),
        "metrics": {
            **metrics,
            "approved_for_review_mean_return": approved_metrics.get("mean_net_return_5d"),
            "rejected_mean_return": rejected_metrics.get("mean_net_return_5d"),
            "rule_first_ml_modifier": metrics.get("mean_net_return_5d"),
        },
        "order_enabled": False,
        "execution_status": "paper_only",
    }


def build_report(result: Dict[str, Any]) -> str:
    metrics = result.get("metrics") or {}
    lines = [
        "# Paper Trading Report",
        "",
        f"- generated_at: {result.get('generated_at')}",
        "- order_enabled=false",
        "- execution_status=paper_only",
        f"- evaluation_status: {result.get('evaluation_status')}",
        f"- horizon_days: {result.get('horizon_days')}",
        f"- fee_bps: {result.get('fee_bps')}",
        f"- slippage_bps: {result.get('slippage_bps')}",
        "",
        "## Metrics",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    if result.get("evaluation_status") == "no_op_insufficient_labeled_paper_trades":
        lines.extend([
            "",
            "## No-op",
            "- 아직 평가 가능한 labeled paper trade가 부족합니다.",
        ])
    lines.extend([
        "",
        "Paper trading은 실제 주문 체결이 아니라 후보 추적 평가입니다.",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paper trade candidates against future price data.")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--fee-bps", type=float, default=15.0)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    save_json(PAPER_TRADING_REPORT_JSON_PATH, result)
    PAPER_TRADING_REPORT_MD_PATH.write_text(build_report(result), encoding="utf-8")
    print(f"Saved paper trading json: {PAPER_TRADING_REPORT_JSON_PATH}")
    print(f"Saved paper trading report: {PAPER_TRADING_REPORT_MD_PATH}")
    print(f"evaluation_status={result.get('evaluation_status')}")
    print("order_enabled=false")
    print("execution_status=paper_only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
