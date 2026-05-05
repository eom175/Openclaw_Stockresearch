#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _bootstrap import add_src_to_path

add_src_to_path()

from policylink.dart.disclosures import build_feature as build_dart_feature
from policylink.dataset.build import (
    CSV_COLUMNS,
    build_dart_columns,
    build_news_columns,
    build_yahoo_columns,
    normalize_yahoo_features,
)
from policylink.kiwoom.flows import build_flow_feature
from policylink.kiwoom.prices import build_features_for_stock
from policylink.news.naver import build_feature as build_news_feature, parse_pub_date
from policylink.paths import (
    BACKFILL_DART_EVENTS_HISTORY_PATH,
    BACKFILL_FLOWS_DAILY_HISTORY_PATH,
    BACKFILL_NAVER_NEWS_HISTORY_PATH,
    BACKFILL_PRICES_DAILY_HISTORY_PATH,
    BACKFILL_YAHOO_GLOBAL_HISTORY_PATH,
    HISTORICAL_DATASET_REPORT_PATH,
    HISTORICAL_MODEL_DATASET_CSV_PATH,
    HISTORICAL_MODEL_DATASET_JSONL_PATH,
    ensure_project_dirs,
)
from policylink.universe import KNOWN_STOCK_SECTOR, universe_for_dataset
from policylink.utils import load_json, normalize_code, parse_number, save_jsonl
from policylink.yahoo.finance import build_proxy_group_scores, build_sector_scores, build_ticker_feature, compute_price_features


KST = timezone(timedelta(hours=9))
HISTORICAL_COLUMNS = list(CSV_COLUMNS)
if "label_updated_at" not in HISTORICAL_COLUMNS:
    HISTORICAL_COLUMNS.append("label_updated_at")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ymd_to_dt(value: Any) -> Optional[datetime]:
    try:
        return datetime.strptime(str(value), "%Y%m%d").replace(tzinfo=KST)
    except Exception:
        return None


def ymd_to_iso(value: Any) -> str:
    text = str(value)
    if len(text) == 8:
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text[:10]


def iso_to_ymd(value: Any) -> Optional[str]:
    if not value:
        return None
    text = str(value)
    if len(text) >= 10 and "-" in text:
        return text[:10].replace("-", "")
    if len(text) >= 8:
        return text[:8]
    return None


def finite_round(value: Any, digits: int = 6) -> Optional[float]:
    try:
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return round(numeric, digits)
    except Exception:
        return None


def save_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def sorted_price_rows(prices_payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    prices = prices_payload.get("prices", {}) if isinstance(prices_payload, dict) else {}
    result: Dict[str, List[Dict[str, Any]]] = {}
    for code, item in prices.items():
        if not isinstance(item, dict):
            continue
        rows = [row for row in item.get("rows", []) if isinstance(row, dict) and row.get("date") and parse_number(row.get("close"), 0) > 0]
        rows.sort(key=lambda row: str(row.get("date")))
        result[normalize_code(code)] = rows
    return result


def sorted_flow_rows(flows_payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    flows = flows_payload.get("flows", {}) if isinstance(flows_payload, dict) else {}
    result: Dict[str, List[Dict[str, Any]]] = {}
    for code, item in flows.items():
        if not isinstance(item, dict):
            continue
        rows = [row for row in item.get("rows", []) if isinstance(row, dict) and row.get("date")]
        rows.sort(key=lambda row: str(row.get("date")))
        result[normalize_code(code)] = rows
    return result


def rows_until(rows: List[Dict[str, Any]], snapshot_date: str) -> List[Dict[str, Any]]:
    return [row for row in rows if str(row.get("date") or "") <= snapshot_date]


def events_until(events: List[Dict[str, Any]], snapshot_date: str) -> List[Dict[str, Any]]:
    return [event for event in events if str(event.get("rcept_dt") or "") <= snapshot_date]


def news_until(items: List[Dict[str, Any]], snapshot_end_utc: datetime) -> List[Dict[str, Any]]:
    result = []
    for item in items:
        parsed = parse_pub_date(item.get("pubDate"))
        if parsed is None or parsed <= snapshot_end_utc:
            result.append(item)
    return result


def snapshot_end_utc(snapshot_date: str) -> datetime:
    base = ymd_to_dt(snapshot_date) or datetime.now(KST)
    return (base + timedelta(hours=23, minutes=59, seconds=59)).astimezone(timezone.utc)


def price_score_from_feature(feature: Dict[str, Any]) -> float:
    score = 50.0
    ret_5d = parse_number(feature.get("return_5d"), 0.0)
    ret_20d = parse_number(feature.get("return_20d"), 0.0)
    volatility = parse_number(feature.get("volatility_20d"), 0.0)
    drawdown = parse_number(feature.get("drawdown_20d"), 0.0)
    trend = feature.get("trend_label")
    risk = feature.get("risk_label")

    score += ret_5d * 180.0
    score += ret_20d * 90.0
    if trend == "uptrend":
        score += 8.0
    elif trend == "downtrend":
        score -= 10.0
    if risk == "high_volatility" or volatility >= 0.04:
        score -= 10.0
    elif risk == "low_volatility":
        score += 2.0
    if drawdown < -0.10:
        score -= 5.0
    return round(max(0.0, min(100.0, score)), 4)


def combined_final_score(price_score: float, flow_score: float, dart_score: Optional[float], news_score: Optional[float], yahoo_score: Optional[float]) -> float:
    score = price_score * 0.35 + flow_score * 0.30
    score += parse_number(dart_score, 50.0) * 0.10
    score += parse_number(news_score, 50.0) * 0.10
    score += parse_number(yahoo_score, 50.0) * 0.15
    return round(max(0.0, min(100.0, score)), 4)


def risk_level_from_features(price_feature: Dict[str, Any], flow_feature: Optional[Dict[str, Any]], dart_columns: Dict[str, Any], news_columns: Dict[str, Any]) -> str:
    if price_feature.get("risk_label") == "high_volatility" or price_feature.get("trend_label") == "downtrend":
        return "high"
    if dart_columns.get("dart_label") == "disclosure_risk" or news_columns.get("naver_news_label") == "negative_news_flow":
        return "high"
    if flow_feature and parse_number(flow_feature.get("flow_score"), 50.0) < 40:
        return "medium"
    return "low"


def future_return(rows: List[Dict[str, Any]], idx: int, horizon: int) -> Optional[float]:
    if idx + horizon >= len(rows):
        return None
    current = parse_number(rows[idx].get("close"), 0.0)
    future = parse_number(rows[idx + horizon].get("close"), 0.0)
    if current <= 0:
        return None
    return finite_round(future / current - 1.0)


def future_drawdown(rows: List[Dict[str, Any]], idx: int, horizon: int) -> Optional[float]:
    current = parse_number(rows[idx].get("close"), 0.0)
    if current <= 0 or idx + 1 >= len(rows):
        return None
    future_rows = rows[idx + 1 : min(len(rows), idx + horizon + 1)]
    if not future_rows:
        return None
    min_close = min(parse_number(row.get("close"), current) for row in future_rows)
    return finite_round(min_close / current - 1.0)


def dataframe_from_yahoo_rows(rows: List[Dict[str, Any]], snapshot_date: str):
    import pandas as pd

    selected = [row for row in rows if iso_to_ymd(row.get("date")) and iso_to_ymd(row.get("date")) <= snapshot_date]
    if not selected:
        return pd.DataFrame()
    frame = pd.DataFrame(selected)
    frame["Date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date")
    frame = frame.set_index("Date")
    frame["Close"] = pd.to_numeric(frame.get("close"), errors="coerce")
    frame["Volume"] = pd.to_numeric(frame.get("volume"), errors="coerce").fillna(0)
    return frame


def yahoo_context_for_snapshot(yahoo_payload: Dict[str, Any], snapshot_date: str, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if snapshot_date in cache:
        return cache[snapshot_date]

    tickers = yahoo_payload.get("tickers", {}) if isinstance(yahoo_payload, dict) else {}
    features: Dict[str, Dict[str, Any]] = {}
    end_utc = snapshot_end_utc(snapshot_date)

    for ticker, item in tickers.items():
        if not isinstance(item, dict):
            continue
        history = dataframe_from_yahoo_rows(item.get("daily", []), snapshot_date)
        price_feature = compute_price_features(history)
        news_items = []
        for news in item.get("news", []):
            published = news.get("published_at")
            keep = True
            if published:
                try:
                    text = str(published)
                    if text.endswith("Z"):
                        text = text[:-1] + "+00:00"
                    keep = datetime.fromisoformat(text).astimezone(timezone.utc) <= end_utc
                except Exception:
                    keep = True
            if keep:
                news_items.append(news)
        features[ticker] = build_ticker_feature(ticker, {k: v for k, v in price_feature.items() if k != "daily"}, news_items)

    context = normalize_yahoo_features({
        "features": features,
        "sector_global_scores": build_sector_scores(features),
        "proxy_group_scores": build_proxy_group_scores(features),
    })
    cache[snapshot_date] = context
    return context


def candidate_snapshots(
    price_rows_by_code: Dict[str, List[Dict[str, Any]]],
    min_history_days: int,
    horizon_days: int,
    max_snapshot_days: int,
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[Tuple[str, str, int]]:
    candidates: List[Tuple[str, str, int]] = []
    for code, rows in price_rows_by_code.items():
        min_idx = max(0, min_history_days - 1)
        max_idx = len(rows) - horizon_days - 1
        for idx in range(min_idx, max_idx + 1):
            date = str(rows[idx].get("date"))
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue
            candidates.append((date, code, idx))

    dates = sorted({date for date, _, _ in candidates})
    if max_snapshot_days > 0 and len(dates) > max_snapshot_days:
        allowed = set(dates[-max_snapshot_days:])
        candidates = [item for item in candidates if item[0] in allowed]
    return sorted(candidates)


def create_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prices_payload = load_json(BACKFILL_PRICES_DAILY_HISTORY_PATH, {"prices": {}})
    flows_payload = load_json(BACKFILL_FLOWS_DAILY_HISTORY_PATH, {"flows": {}})
    dart_payload = load_json(BACKFILL_DART_EVENTS_HISTORY_PATH, {"events_by_stock": {}})
    naver_payload = load_json(BACKFILL_NAVER_NEWS_HISTORY_PATH, {"news_by_stock": {}})
    yahoo_payload = load_json(BACKFILL_YAHOO_GLOBAL_HISTORY_PATH, {"tickers": {}})

    price_rows_by_code = sorted_price_rows(prices_payload)
    flow_rows_by_code = sorted_flow_rows(flows_payload)
    dart_events_by_code = {
        normalize_code(code): item.get("events", [])
        for code, item in (dart_payload.get("events_by_stock", {}) or {}).items()
        if isinstance(item, dict)
    }
    naver_items_by_code = {
        normalize_code(code): item.get("items", [])
        for code, item in (naver_payload.get("news_by_stock", {}) or {}).items()
        if isinstance(item, dict)
    }
    universe = {normalize_code(item["stock_code"]): item for item in universe_for_dataset()}
    yahoo_cache: Dict[str, Dict[str, Any]] = {}
    snapshot_items = candidate_snapshots(
        price_rows_by_code,
        args.min_history_days,
        args.horizon_days,
        args.max_snapshot_days,
        args.start_date,
        args.end_date,
    )

    rows: List[Dict[str, Any]] = []
    generated_at = utc_now()
    for snapshot_date, code, idx in snapshot_items:
        stock_meta = universe.get(code, {
            "stock_code": code,
            "stock_name": code,
            "sector": KNOWN_STOCK_SECTOR.get(code, "unknown"),
        })
        sector = stock_meta.get("sector") or KNOWN_STOCK_SECTOR.get(code, "unknown")
        stock_name = stock_meta.get("stock_name") or code
        price_history = price_rows_by_code[code][: idx + 1]
        price_feature = build_features_for_stock(price_history) or {}
        flow_feature = build_flow_feature(rows_until(flow_rows_by_code.get(code, []), snapshot_date)) or {}

        dart_feature = build_dart_feature(
            {"code": code, "name": stock_name, "sector": sector},
            None,
            events_until(dart_events_by_code.get(code, []), snapshot_date),
            ymd_to_dt(snapshot_date) or datetime.now(KST),
        )
        dart_columns = build_dart_columns(dart_feature)

        news_feature = build_news_feature(
            {"code": code, "name": stock_name, "sector": sector},
            0,
            news_until(naver_items_by_code.get(code, []), snapshot_end_utc(snapshot_date)),
            snapshot_end_utc(snapshot_date),
        )
        news_columns = build_news_columns(news_feature)

        yahoo_context = yahoo_context_for_snapshot(yahoo_payload, snapshot_date, yahoo_cache)
        yahoo_columns = build_yahoo_columns(sector, yahoo_context)

        price_score = price_score_from_feature(price_feature)
        flow_score = parse_number(flow_feature.get("flow_score"), 50.0)
        final_score = combined_final_score(
            price_score,
            flow_score,
            dart_columns.get("dart_score"),
            news_columns.get("naver_sentiment_score"),
            yahoo_columns.get("yahoo_sector_global_signal_score"),
        )
        risk_level = risk_level_from_features(price_feature, flow_feature, dart_columns, news_columns)
        row = {
            "snapshot_date": snapshot_date,
            "generated_at": generated_at,
            "stock_code": code,
            "stock_name": stock_name,
            "sector": sector,
            "is_holding": 0,
            "holding_quantity": 0,
            "holding_eval_amount": 0,
            "holding_weight": 0,
            "holding_pnl": 0,
            "holding_return_rate": 0,
            "account_total_equity": None,
            "account_cash": None,
            "account_cash_weight": None,
            "account_invested_weight": None,
            "pending_or_reserved_amount": 0,
            "research_score": 0,
            "price_score": price_score,
            "flow_score": round(flow_score, 4),
            "final_score": final_score,
            "target_weight": 0,
            "recommendation_rank": 0,
            "latest_date": price_feature.get("latest_date"),
            "latest_close": price_feature.get("latest_close"),
            "latest_volume": price_feature.get("latest_volume"),
            "return_1d": price_feature.get("return_1d"),
            "return_5d": price_feature.get("return_5d"),
            "return_20d": price_feature.get("return_20d"),
            "volatility_20d": price_feature.get("volatility_20d"),
            "ma20_gap": price_feature.get("ma20_gap"),
            "ma60_gap": price_feature.get("ma60_gap"),
            "drawdown_20d": price_feature.get("drawdown_20d"),
            "volume_ratio_20": price_feature.get("volume_ratio_20"),
            "trend_label": price_feature.get("trend_label"),
            "price_risk_label": price_feature.get("risk_label"),
            "foreign_net_1d": flow_feature.get("foreign_net_1d"),
            "foreign_net_5d": flow_feature.get("foreign_net_5d"),
            "foreign_net_20d": flow_feature.get("foreign_net_20d"),
            "institution_net_1d": flow_feature.get("institution_net_1d"),
            "institution_net_5d": flow_feature.get("institution_net_5d"),
            "institution_net_20d": flow_feature.get("institution_net_20d"),
            "combined_net_1d": flow_feature.get("combined_net_1d"),
            "combined_net_5d": flow_feature.get("combined_net_5d"),
            "combined_net_20d": flow_feature.get("combined_net_20d"),
            "combined_net_5d_to_avg_volume_20": flow_feature.get("combined_net_5d_to_avg_volume_20"),
            "combined_net_20d_to_avg_volume_20": flow_feature.get("combined_net_20d_to_avg_volume_20"),
            "foreign_weight": flow_feature.get("foreign_weight"),
            "foreign_limit_exhaustion_rate": flow_feature.get("foreign_limit_exhaustion_rate"),
            "flow_label": flow_feature.get("flow_label"),
            **dart_columns,
            **news_columns,
            **yahoo_columns,
            "risk_level": risk_level,
            "risk_score": None,
            "opportunity_score": None,
            "macro_pressure_score": None,
            "future_return_1d": future_return(price_rows_by_code[code], idx, 1),
            "future_return_5d": future_return(price_rows_by_code[code], idx, args.horizon_days),
            "future_return_20d": future_return(price_rows_by_code[code], idx, 20),
            "future_outperform_5d": None,
            "future_max_drawdown_5d": future_drawdown(price_rows_by_code[code], idx, args.horizon_days),
            "label_status": "labeled",
            "label_updated_at": generated_at,
        }
        rows.append(row)

    for snapshot_date in sorted({row["snapshot_date"] for row in rows}):
        same_date = [row for row in rows if row["snapshot_date"] == snapshot_date and row.get("future_return_5d") is not None]
        if not same_date:
            continue
        returns = sorted(float(row["future_return_5d"]) for row in same_date)
        mid = len(returns) // 2
        median = returns[mid] if len(returns) % 2 else (returns[mid - 1] + returns[mid]) / 2
        for row in same_date:
            row["future_outperform_5d"] = 1 if float(row["future_return_5d"]) > median else 0

    rows.sort(key=lambda row: (str(row.get("snapshot_date")), str(row.get("stock_code"))))
    meta = {
        "generated_at": generated_at,
        "price_stock_count": len(price_rows_by_code),
        "flow_stock_count": len(flow_rows_by_code),
        "snapshot_date_count": len({row.get("snapshot_date") for row in rows}),
        "row_count": len(rows),
        "labeled_row_count": sum(1 for row in rows if row.get("future_outperform_5d") is not None),
        "point_in_time": True,
    }
    return rows, meta


def build_report(rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    lines = [
        "# Historical Model Dataset Report",
        "",
        f"- generated_at: {meta.get('generated_at')}",
        "- 실제 주문 실행: 비활성화",
        "- order_enabled=false",
        "- point_in_time=true",
        f"- row_count: {meta.get('row_count')}",
        f"- labeled_row_count: {meta.get('labeled_row_count')}",
        f"- snapshot_date_count: {meta.get('snapshot_date_count')}",
        f"- JSONL: `{HISTORICAL_MODEL_DATASET_JSONL_PATH}`",
        f"- CSV: `{HISTORICAL_MODEL_DATASET_CSV_PATH}`",
        "",
        "## Point-in-time 원칙",
        "- 각 snapshot_date의 feature는 해당 날짜 당일 또는 이전 데이터만 사용했습니다.",
        "- future_* 컬럼은 target 전용이며 feature 계산에는 사용하지 않았습니다.",
        "- 과거 계좌 상태는 없으므로 account/holding feature는 0 또는 null로 처리했습니다.",
        "- RSS 정책 리서치 과거 백필은 이번 범위에서 제외했고 research_score는 0으로 둡니다.",
        "- Naver News 백필은 검색 API 결과 기반 compact metadata이며 완전한 과거 뉴스 아카이브가 아닙니다.",
        "",
        "## Coverage",
    ]
    if rows:
        lines.append(f"- date_range: {rows[0].get('snapshot_date')} ~ {rows[-1].get('snapshot_date')}")
        by_sector: Dict[str, int] = {}
        for row in rows:
            by_sector[str(row.get("sector") or "unknown")] = by_sector.get(str(row.get("sector") or "unknown"), 0) + 1
        for sector, count in sorted(by_sector.items()):
            lines.append(f"- {sector}: {count}")
    else:
        lines.append("- 생성된 row가 없습니다. backfill prices row 수를 확인하세요.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build point-in-time historical model dataset from backfill data.")
    parser.add_argument("--min-history-days", type=int, default=80)
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--max-snapshot-days", type=int, default=180)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_project_dirs()
    rows, meta = create_rows(args)
    save_jsonl(HISTORICAL_MODEL_DATASET_JSONL_PATH, rows)
    save_csv(HISTORICAL_MODEL_DATASET_CSV_PATH, rows, HISTORICAL_COLUMNS)
    HISTORICAL_DATASET_REPORT_PATH.write_text(build_report(rows, meta), encoding="utf-8")
    print(f"Saved historical jsonl: {HISTORICAL_MODEL_DATASET_JSONL_PATH}")
    print(f"Saved historical csv: {HISTORICAL_MODEL_DATASET_CSV_PATH}")
    print(f"Saved report: {HISTORICAL_DATASET_REPORT_PATH}")
    print(f"row_count={meta.get('row_count')}")
    print(f"labeled_row_count={meta.get('labeled_row_count')}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
