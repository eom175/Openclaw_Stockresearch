#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

from _bootstrap import add_src_to_path

add_src_to_path()

from policylink.paths import (
    MODEL_DATASET_CSV_PATH,
    ORDER_PROPOSALS_JSON_PATH,
    ORDER_RISK_CHECK_JSON_PATH,
    PAPER_TRADES_JSONL_PATH,
    PAPER_TRADES_SNAPSHOT_REPORT_PATH,
    PRICE_FEATURES_PATH,
    ensure_project_dirs,
)
from policylink.utils import load_json, load_jsonl, normalize_code, parse_number, save_jsonl


KST = timezone(timedelta(hours=9))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def today_yyyymmdd() -> str:
    return datetime.now(KST).strftime("%Y%m%d")


def load_latest_model_rows() -> Dict[str, Dict[str, Any]]:
    import csv

    if not MODEL_DATASET_CSV_PATH.exists():
        return {}
    try:
        with MODEL_DATASET_CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return {}
    if not rows:
        return {}
    latest = max(str(row.get("snapshot_date") or "") for row in rows)
    return {
        normalize_code(row.get("stock_code")): row
        for row in rows
        if str(row.get("snapshot_date") or "") == latest and normalize_code(row.get("stock_code"))
    }


def normalize_price_features() -> Dict[str, Dict[str, Any]]:
    payload = load_json(PRICE_FEATURES_PATH, {"features": {}})
    features = payload.get("features", {}) if isinstance(payload, dict) else {}
    if not isinstance(features, dict):
        return {}
    return {normalize_code(code): item for code, item in features.items() if isinstance(item, dict)}


def risk_status_by_id() -> Dict[str, Dict[str, Any]]:
    payload = load_json(ORDER_RISK_CHECK_JSON_PATH, {"checked_proposals": []})
    result = {}
    for item in payload.get("checked_proposals", []) or []:
        if isinstance(item, dict) and item.get("proposal_id"):
            result[str(item.get("proposal_id"))] = item
    return result


def existing_keys(rows: List[Dict[str, Any]]) -> set:
    return {
        (
            str(row.get("snapshot_date")),
            normalize_code(row.get("stock_code")),
            str(row.get("side")),
            str(row.get("proposal_type")),
            str(row.get("tracking_status")),
        )
        for row in rows
    }


def make_id(snapshot_date: str, sequence: int) -> str:
    return f"{snapshot_date}-PT-{sequence:03d}"


def should_keep(proposal: Dict[str, Any], args: argparse.Namespace, risk_item: Dict[str, Any]) -> bool:
    side = str(proposal.get("side") or "")
    risk_status = str(risk_item.get("proposal_status") or proposal.get("risk_check_status") or "")
    tracking_status = str(proposal.get("tracking_status") or risk_item.get("tracking_status") or "")
    paper_tracking_status = str(risk_item.get("paper_tracking_status") or proposal.get("paper_tracking_status") or "")

    if args.side != "both" and side != args.side:
        return False
    if side == "watch" and not (args.allow_watch or args.include_watch):
        return False
    if risk_status == "rejected" and not args.include_soft_rejected:
        return False

    if args.only_approved_for_review:
        return risk_status == "approved_for_review"

    if args.tracking_mode == "approved_only":
        return risk_status == "approved_for_review"
    if paper_tracking_status != "track":
        return False
    if args.tracking_mode == "paper_candidates":
        return tracking_status in {"execution_candidate", "paper_candidate"}
    if args.tracking_mode == "all_trackable":
        return True

    return True


def build_record(
    proposal: Dict[str, Any],
    risk_item: Dict[str, Any],
    price_features: Dict[str, Dict[str, Any]],
    model_rows: Dict[str, Dict[str, Any]],
    snapshot_date: str,
    sequence: int,
) -> Dict[str, Any]:
    code = normalize_code(proposal.get("stock_code"))
    price_feature = price_features.get(code, {})
    model_row = model_rows.get(code, {})
    scores = proposal.get("scores") if isinstance(proposal.get("scores"), dict) else {}
    entry_price = parse_number(price_feature.get("latest_close"), parse_number(proposal.get("estimated_price"), 0.0))
    return {
        "paper_trade_id": make_id(snapshot_date, sequence),
        "created_at": utc_now(),
        "snapshot_date": snapshot_date,
        "stock_code": code,
        "stock_name": proposal.get("stock_name"),
        "sector": proposal.get("sector"),
        "side": proposal.get("side"),
        "proposal_type": proposal.get("proposal_type"),
        "proposal_id": proposal.get("proposal_id"),
        "entry_reference_price": entry_price,
        "entry_price_source": "price_features.latest_close" if price_feature.get("latest_close") is not None else "proposal.estimated_price",
        "recommended_qty": int(parse_number(proposal.get("recommended_qty"), 0)),
        "estimated_amount": int(parse_number(proposal.get("estimated_amount"), 0)),
        "rule_final_score": parse_number(scores.get("final_score"), parse_number(model_row.get("final_score"), 0.0)),
        "ml_signal_score": parse_number(model_row.get("ml_signal_score"), 0.0) if model_row else parse_number(scores.get("ml_signal_score"), 0.0),
        "outperform_prob_5d": scores.get("ml_outperform_prob_5d"),
        "predicted_return_5d": scores.get("ml_predicted_return_5d"),
        "predicted_drawdown_5d": scores.get("ml_predicted_drawdown_5d"),
        "tracking_status": proposal.get("tracking_status") or risk_item.get("tracking_status"),
        "paper_tracking_reason": risk_item.get("paper_tracking_reason") or proposal.get("paper_tracking_reason") or [],
        "risk_check_status": risk_item.get("proposal_status") or proposal.get("risk_check_status"),
        "risk_guard_status": risk_item.get("proposal_status") or proposal.get("risk_check_status"),
        "paper_tracking_status": risk_item.get("paper_tracking_status") or proposal.get("paper_tracking_status"),
        "can_execute": False,
        "order_enabled": False,
        "execution_status": "paper_only",
        "label_status": "pending",
        "future_return_1d": None,
        "future_return_5d": None,
        "future_return_20d": None,
        "max_drawdown_5d": None,
        "net_return_5d": None,
    }


def log_paper_trades(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_project_dirs()
    proposals_payload = load_json(ORDER_PROPOSALS_JSON_PATH, {"proposals": []})
    proposals = [item for item in proposals_payload.get("proposals", []) or [] if isinstance(item, dict)]
    risk_map = risk_status_by_id()
    price_features = normalize_price_features()
    model_rows = load_latest_model_rows()
    existing = load_jsonl(PAPER_TRADES_JSONL_PATH)
    keys = existing_keys(existing)
    snapshot_date = today_yyyymmdd()
    new_records: List[Dict[str, Any]] = []
    duplicates = 0
    skipped = 0

    for proposal in proposals:
        if len(new_records) >= args.max_candidates:
            break
        risk_item = risk_map.get(str(proposal.get("proposal_id")), {})
        if not should_keep(proposal, args, risk_item):
            skipped += 1
            continue
        key = (
            snapshot_date,
            normalize_code(proposal.get("stock_code")),
            str(proposal.get("side")),
            str(proposal.get("proposal_type")),
            str(proposal.get("tracking_status")),
        )
        if key in keys:
            duplicates += 1
            continue
        new_records.append(build_record(proposal, risk_item, price_features, model_rows, snapshot_date, len(existing) + len(new_records) + 1))
        keys.add(key)

    all_rows = existing + new_records
    save_jsonl(PAPER_TRADES_JSONL_PATH, all_rows)
    return {
        "generated_at": utc_now(),
        "mode": "paper_trade_logging",
        "snapshot_date": snapshot_date,
        "proposal_count": len(proposals),
        "new_records": len(new_records),
        "duplicate_skipped": duplicates,
        "filter_skipped": skipped,
        "total_records": len(all_rows),
        "tracking_mode": args.tracking_mode,
        "only_approved_for_review": bool(args.only_approved_for_review),
        "include_watch": bool(args.include_watch or args.allow_watch),
        "include_soft_rejected": bool(args.include_soft_rejected),
        "order_enabled": False,
        "execution_status": "paper_only",
    }


def build_report(result: Dict[str, Any]) -> str:
    lines = [
        "# Paper Trades Snapshot",
        "",
        f"- generated_at: {result.get('generated_at')}",
        "- order_enabled=false",
        "- execution_status=paper_only",
        f"- snapshot_date: {result.get('snapshot_date')}",
        f"- proposal_count: {result.get('proposal_count')}",
        f"- new_records: {result.get('new_records')}",
        f"- duplicate_skipped: {result.get('duplicate_skipped')}",
        f"- filter_skipped: {result.get('filter_skipped')}",
        f"- total_records: {result.get('total_records')}",
        f"- tracking_mode: {result.get('tracking_mode')}",
        f"- include_watch: {result.get('include_watch')}",
        f"- include_soft_rejected: {result.get('include_soft_rejected')}",
        "",
        "Paper trading은 실제 주문이 아니라 후보 추적용 로그입니다.",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log proposal-only order candidates into paper trade tracking.")
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--side", choices=["both", "buy", "sell"], default="both")
    parser.add_argument("--tracking-mode", choices=["approved_only", "paper_candidates", "all_trackable"], default="paper_candidates")
    parser.add_argument("--include-watch", action="store_true")
    parser.add_argument("--include-soft-rejected", action="store_true")
    parser.add_argument("--only-approved-for-review", action="store_true")
    parser.add_argument("--allow-watch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = log_paper_trades(args)
    PAPER_TRADES_SNAPSHOT_REPORT_PATH.write_text(build_report(result), encoding="utf-8")
    print(f"Saved paper trades: {PAPER_TRADES_JSONL_PATH}")
    print(f"Saved paper trades snapshot: {PAPER_TRADES_SNAPSHOT_REPORT_PATH}")
    print(f"new_records={result.get('new_records')}")
    print("order_enabled=false")
    print("execution_status=paper_only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
