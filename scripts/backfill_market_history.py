#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from _bootstrap import add_src_to_path

add_src_to_path()

from policylink.config import load_dart_api_key, load_naver_settings
from policylink.dart.disclosures import (
    build_feature as build_dart_feature,
    fetch_disclosures_for_corp,
    get_corp_codes,
    kst_today,
    safe_error_message as safe_dart_error,
    yyyymmdd,
)
from policylink.kiwoom.client import KiwoomRestClient
from policylink.kiwoom.flows import (
    extract_foreign_rows,
    extract_institution_rows,
    fetch_foreign_trading_trend,
    fetch_institutional_stock,
    merge_flow_rows,
)
from policylink.kiwoom.prices import extract_daily_rows, fetch_daily_chart, normalize_daily_row
from policylink.news.naver import (
    NAVER_NEWS_URL,
    QUERY_MAP,
    compact_news_item,
    dedupe_news,
    parse_pub_date,
    safe_error_message as safe_naver_error,
)
from policylink.paths import (
    BACKFILL_DART_EVENTS_HISTORY_PATH,
    BACKFILL_FLOWS_DAILY_HISTORY_PATH,
    BACKFILL_MARKET_HISTORY_REPORT_PATH,
    BACKFILL_NAVER_NEWS_HISTORY_PATH,
    BACKFILL_PRICES_DAILY_HISTORY_PATH,
    BACKFILL_YAHOO_GLOBAL_HISTORY_PATH,
    BACKFILL_STATUS_PATH,
    ensure_project_dirs,
)
from policylink.universe import universe_for_market_data
from policylink.utils import normalize_code, parse_number, save_json
from policylink.yahoo.finance import (
    PROXY_TICKERS,
    YAHOO_SOURCE_NOTE,
    build_proxy_group_scores,
    build_sector_scores,
    build_ticker_feature,
    collect_news as collect_yahoo_news,
    collect_price_history as collect_yahoo_price_history,
    compute_price_features,
)


KST = timezone(timedelta(hours=9))
SENSITIVE_RE = re.compile(
    r"(crtfc_key=|client[_-]?secret[\"'=:\s]+|client[_-]?id[\"'=:\s]+|app[_-]?key[\"'=:\s]+|secret[_-]?key[\"'=:\s]+|authorization[\"'=:\s]+|bearer\s+)[^&\s,}]+",
    re.IGNORECASE,
)
CREDENTIAL_NAME_RE = re.compile(
    r"\b(KIWOOM_APP_KEY|KIWOOM_SECRET_KEY|DART_API_KEY|NAVER_CLIENT_ID|NAVER_CLIENT_SECRET|access token|authorization header|account number)\b",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_error(exc: Exception) -> str:
    text = SENSITIVE_RE.sub(r"\1[REDACTED]", str(exc))
    text = CREDENTIAL_NAME_RE.sub("[CREDENTIAL_REDACTED]", text)
    return text[:500]


def parse_yyyymmdd(value: Any) -> Optional[datetime]:
    try:
        return datetime.strptime(str(value), "%Y%m%d").replace(tzinfo=KST)
    except Exception:
        return None


def start_yyyymmdd(days: int) -> str:
    return (datetime.now(KST) - timedelta(days=max(1, days))).strftime("%Y%m%d")


def filter_rows_by_days(rows: List[Dict[str, Any]], days: int) -> List[Dict[str, Any]]:
    start = start_yyyymmdd(days)
    return [row for row in rows if str(row.get("date") or "") >= start]


def collect_kiwoom_price_history(max_stocks: int, days: int, sleep_seconds: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    client = KiwoomRestClient()
    base_date = datetime.now(KST).strftime("%Y%m%d")
    prices: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}

    for stock in universe_for_market_data()[:max_stocks]:
        code = normalize_code(stock["code"])
        try:
            raw = fetch_daily_chart(client, code, base_date)
            raw_rows = extract_daily_rows(raw)
            rows = []
            for raw_row in raw_rows:
                normalized = normalize_daily_row(raw_row, code, stock["name"], stock["sector"])
                if normalized:
                    rows.append(normalized)
            rows = sorted(filter_rows_by_days(rows, days), key=lambda item: item["date"])
            prices[code] = {
                "stock_code": code,
                "stock_name": stock["name"],
                "sector": stock["sector"],
                "rows": rows,
            }
            diagnostics[code] = {
                "status": "ok" if rows else "warning",
                "raw_row_count": len(raw_rows),
                "row_count": len(rows),
            }
        except Exception as exc:
            prices[code] = {
                "stock_code": code,
                "stock_name": stock["name"],
                "sector": stock["sector"],
                "rows": [],
            }
            diagnostics[code] = {"status": "failed", "error": sanitize_error(exc), "row_count": 0}
        time.sleep(max(0.0, sleep_seconds))

    payload = {
        "generated_at": utc_now(),
        "source": "kiwoom_rest_api",
        "days": days,
        "base_date": base_date,
        "prices": prices,
    }
    return payload, diagnostics


def collect_flow_history(max_stocks: int, days: int, sleep_seconds: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    client = KiwoomRestClient()
    flows: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}

    for stock in universe_for_market_data()[:max_stocks]:
        code = normalize_code(stock["code"])
        try:
            foreign = fetch_foreign_trading_trend(client, code)
            time.sleep(max(0.0, sleep_seconds))
            institution = fetch_institutional_stock(client, code)
            foreign_rows = extract_foreign_rows(foreign)
            institution_rows = extract_institution_rows(institution)
            merged = merge_flow_rows(code, stock["name"], stock["sector"], foreign_rows, institution_rows)
            rows = sorted(filter_rows_by_days(merged, days), key=lambda item: item["date"])
            compact_rows = [
                {
                    "date": row.get("date"),
                    "stock_code": code,
                    "stock_name": stock["name"],
                    "sector": stock["sector"],
                    "close_price": row.get("close_price", 0),
                    "trading_volume": row.get("trading_volume", 0),
                    "foreign_net_qty": row.get("foreign_net_qty", 0),
                    "institution_net_qty": row.get("institution_net_qty", 0),
                    "combined_net_qty": int(row.get("foreign_net_qty", 0) or 0) + int(row.get("institution_net_qty", 0) or 0),
                    "foreign_weight": row.get("foreign_weight", 0.0),
                    "foreign_limit_exhaustion_rate": row.get("foreign_limit_exhaustion_rate", 0.0),
                }
                for row in rows
            ]
            flows[code] = {
                "stock_code": code,
                "stock_name": stock["name"],
                "sector": stock["sector"],
                "rows": compact_rows,
            }
            diagnostics[code] = {
                "status": "ok" if compact_rows else "warning",
                "foreign_raw_row_count": len(foreign_rows),
                "institution_raw_row_count": len(institution_rows),
                "row_count": len(compact_rows),
            }
        except Exception as exc:
            flows[code] = {
                "stock_code": code,
                "stock_name": stock["name"],
                "sector": stock["sector"],
                "rows": [],
            }
            diagnostics[code] = {"status": "failed", "error": sanitize_error(exc), "row_count": 0}
        time.sleep(max(0.0, sleep_seconds))

    payload = {
        "generated_at": utc_now(),
        "source": "kiwoom_rest_api",
        "days": days,
        "flows": flows,
    }
    return payload, diagnostics


def collect_dart_history(max_stocks: int, days: int, sleep_seconds: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    api_key = load_dart_api_key()
    corp_codes, refreshed = get_corp_codes(api_key, force_refresh=False)
    by_stock_code = corp_codes.get("by_stock_code", {}) if isinstance(corp_codes, dict) else {}
    end_dt = kst_today()
    bgn_de = yyyymmdd(end_dt - timedelta(days=max(1, days)))
    end_de = yyyymmdd(end_dt)
    events_by_stock: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {"corp_code_refreshed": refreshed, "stocks": {}}

    for stock in universe_for_market_data()[:max_stocks]:
        code = normalize_code(stock["code"])
        corp_info = by_stock_code.get(code) if isinstance(by_stock_code, dict) else None
        corp_code = corp_info.get("corp_code") if isinstance(corp_info, dict) else None
        if not corp_code:
            events_by_stock[code] = {"stock_code": code, "stock_name": stock["name"], "corp_code": None, "events": []}
            diagnostics["stocks"][code] = {"status": "missing_corp_code", "event_count": 0}
            continue
        try:
            events, diagnostic = fetch_disclosures_for_corp(
                api_key=api_key,
                corp_code=corp_code,
                stock_code=code,
                bgn_de=bgn_de,
                end_de=end_de,
                sleep_seconds=sleep_seconds,
            )
            events_by_stock[code] = {
                "stock_code": code,
                "stock_name": stock["name"],
                "sector": stock["sector"],
                "corp_code": corp_code,
                "events": events,
            }
            diagnostics["stocks"][code] = {
                "status": "rate_limited" if diagnostic.get("rate_limited") else "ok",
                "event_count": len(events),
                "rate_limited": bool(diagnostic.get("rate_limited")),
            }
        except Exception as exc:
            events_by_stock[code] = {
                "stock_code": code,
                "stock_name": stock["name"],
                "sector": stock["sector"],
                "corp_code": corp_code,
                "events": [],
            }
            diagnostics["stocks"][code] = {"status": "failed", "error": safe_dart_error(exc), "event_count": 0}

    payload = {
        "generated_at": utc_now(),
        "source": "opendart",
        "bgn_de": bgn_de,
        "end_de": end_de,
        "events_by_stock": events_by_stock,
    }
    return payload, diagnostics


def fetch_naver_query_paged(settings: Any, query: str, display: int, sleep_seconds: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import requests

    headers = {
        "X-Naver-Client-Id": settings.client_id,
        "X-Naver-Client-Secret": settings.client_secret,
    }
    display = max(1, min(100, display))
    items: List[Dict[str, Any]] = []
    page_diagnostics = []
    stop = False

    for start in range(1, 1001, display):
        params = {"query": query, "display": display, "start": start, "sort": "date"}
        diagnostic = {"query": query, "start": start, "status_code": None, "item_count": 0, "error": None}
        try:
            response = requests.get(NAVER_NEWS_URL, headers=headers, params=params, timeout=15)
            diagnostic["status_code"] = response.status_code
            if response.status_code in {401, 403}:
                diagnostic["error"] = "auth_error"
                stop = True
            elif response.status_code == 429:
                diagnostic["error"] = "rate_limited"
                stop = True
            elif response.status_code >= 400:
                diagnostic["error"] = f"http_{response.status_code}"
                stop = True
            else:
                data = response.json()
                page_items = data.get("items", [])
                if not isinstance(page_items, list):
                    page_items = []
                items.extend(item for item in page_items if isinstance(item, dict))
                diagnostic["item_count"] = len(page_items)
                if len(page_items) < display:
                    stop = True
        except Exception as exc:
            diagnostic["error"] = safe_naver_error(exc)
            stop = True
        page_diagnostics.append(diagnostic)
        time.sleep(max(0.0, sleep_seconds))
        if stop:
            break

    return items, {"query": query, "pages": page_diagnostics, "item_count": len(items)}


def collect_naver_history(max_stocks: int, days: int, sleep_seconds: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    settings = load_naver_settings()
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=max(1, days))
    news_by_stock: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {
        "note": "Naver News API backfill is search-result based and is not a complete historical news archive.",
        "stocks": {},
    }

    for stock in universe_for_market_data()[:max_stocks]:
        code = normalize_code(stock["code"])
        queries = QUERY_MAP.get(code, [stock["name"]])
        compact_items: List[Dict[str, Any]] = []
        query_diagnostics = []
        for query in queries:
            raw_items, diagnostic = fetch_naver_query_paged(settings, query, display=100, sleep_seconds=sleep_seconds)
            query_diagnostics.append(diagnostic)
            for item in raw_items:
                compact = compact_news_item(item, query, code)
                parsed = parse_pub_date(compact.get("pubDate"))
                if parsed is None or start_dt <= parsed <= end_dt:
                    compact_items.append(compact)
            if any((page.get("error") in {"auth_error", "rate_limited"}) for page in diagnostic.get("pages", [])):
                break

        deduped = dedupe_news(compact_items)
        news_by_stock[code] = {
            "stock_code": code,
            "stock_name": stock["name"],
            "sector": stock["sector"],
            "queries": queries,
            "items": deduped,
        }
        diagnostics["stocks"][code] = {
            "query_count": len(queries),
            "item_count": len(deduped),
            "query_diagnostics": query_diagnostics,
        }

    payload = {
        "generated_at": utc_now(),
        "source": "naver_news",
        "days": days,
        "archive_note": "Search API results are compact metadata only and not a complete historical archive.",
        "news_by_stock": news_by_stock,
    }
    return payload, diagnostics


def history_daily_rows(history: Any) -> List[Dict[str, Any]]:
    if history is None or getattr(history, "empty", True):
        return []

    rows = []
    for idx, row in history.iterrows():
        date = idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
        rows.append({
            "date": date,
            "open": parse_number(row.get("Open"), None),
            "high": parse_number(row.get("High"), None),
            "low": parse_number(row.get("Low"), None),
            "close": parse_number(row.get("Close"), None),
            "volume": int(parse_number(row.get("Volume"), 0)),
        })
    return [row for row in rows if row.get("close") not in {None, 0}]


def collect_yahoo_history(days: int, sleep_seconds: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Run: .venv/bin/pip install yfinance pandas") from exc

    tickers: Dict[str, Any] = {}
    features: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {
        "usage_note": YAHOO_SOURCE_NOTE,
        "tickers": {},
    }

    for meta in PROXY_TICKERS:
        ticker = meta["ticker"]
        try:
            history = collect_yahoo_price_history(yf, ticker, days)
            price_feature = compute_price_features(history)
            daily = history_daily_rows(history)
            news = collect_yahoo_news(yf, ticker, count=10)
            feature = build_ticker_feature(ticker, {k: v for k, v in price_feature.items() if k != "daily"}, news)
            tickers[ticker] = {
                "ticker": ticker,
                "label": meta["label"],
                "mapped_sectors": meta["mapped_sectors"],
                "daily": daily,
                "news": news,
            }
            features[ticker] = feature
            diagnostics["tickers"][ticker] = {"status": "ok", "daily_rows": len(daily), "news_count": len(news)}
        except Exception as exc:
            tickers[ticker] = {
                "ticker": ticker,
                "label": meta["label"],
                "mapped_sectors": meta["mapped_sectors"],
                "daily": [],
                "news": [],
            }
            diagnostics["tickers"][ticker] = {"status": "failed", "error": sanitize_error(exc), "daily_rows": 0}
        time.sleep(max(0.0, sleep_seconds))

    payload = {
        "generated_at": utc_now(),
        "source": "yfinance",
        "usage_note": YAHOO_SOURCE_NOTE,
        "tickers": tickers,
        "features": features,
        "sector_global_scores": build_sector_scores(features),
        "proxy_group_scores": build_proxy_group_scores(features),
    }
    return payload, diagnostics


def status_from_diagnostics(items: Dict[str, Any], row_count: int) -> str:
    statuses = []
    for item in items.values():
        if isinstance(item, dict):
            statuses.append(str(item.get("status") or "ok"))
    if statuses and all(status == "failed" for status in statuses):
        return "failed"
    if row_count <= 0 or any(status in {"failed", "warning", "rate_limited"} for status in statuses):
        return "warning"
    return "ok"


def build_report(status_payload: Dict[str, Any]) -> str:
    lines = [
        "# Historical Market Backfill Report",
        "",
        f"- generated_at: {status_payload.get('generated_at')}",
        "- 실제 주문 실행: 비활성화",
        "- order_enabled=false",
        f"- days: {status_payload.get('days')}",
        f"- universe_count: {status_payload.get('universe_count')}",
        "",
        "## Status",
    ]
    for key, value in status_payload.get("status", {}).items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Row Counts"])
    for key, value in status_payload.get("row_counts", {}).items():
        lines.append(f"- {key}: {value}")

    lines.extend([
        "",
        "## Notes",
        "- 이 백필은 주문을 실행하지 않으며 Kiwoom 주문 TR을 호출하지 않습니다.",
        "- DART/Naver/Yahoo 뉴스 원문 전체는 저장하지 않고 compact metadata만 저장합니다.",
        "- Naver News API 결과는 검색 API 기반이며 완전한 과거 뉴스 아카이브가 아닙니다.",
    ])
    for warning in status_payload.get("warnings", [])[:20]:
        lines.append(f"- warning: {warning}")
    return "\n".join(lines)


def run_backfill(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_project_dirs()
    max_stocks = max(1, min(args.max_stocks, len(universe_for_market_data())))
    warnings: List[str] = []

    prices_payload, price_diag = collect_kiwoom_price_history(max_stocks, args.days, args.sleep)
    save_json(BACKFILL_PRICES_DAILY_HISTORY_PATH, prices_payload)
    price_rows = sum(len(item.get("rows", [])) for item in prices_payload.get("prices", {}).values())

    flows_payload, flow_diag = collect_flow_history(max_stocks, args.days, args.sleep)
    save_json(BACKFILL_FLOWS_DAILY_HISTORY_PATH, flows_payload)
    flow_rows = sum(len(item.get("rows", [])) for item in flows_payload.get("flows", {}).values())

    dart_payload = {"generated_at": utc_now(), "source": "opendart", "events_by_stock": {}}
    dart_diag: Dict[str, Any] = {"stocks": {}}
    dart_status = "skipped"
    if args.price_only or args.skip_dart:
        warnings.append("DART backfill skipped.")
    else:
        try:
            dart_payload, dart_diag = collect_dart_history(max_stocks, args.days, args.sleep)
            dart_status = status_from_diagnostics(dart_diag.get("stocks", {}), sum(len(item.get("events", [])) for item in dart_payload.get("events_by_stock", {}).values()))
        except Exception as exc:
            dart_status = "failed"
            warnings.append(f"DART backfill failed: {sanitize_error(exc)}")
    save_json(BACKFILL_DART_EVENTS_HISTORY_PATH, dart_payload)
    dart_rows = sum(len(item.get("events", [])) for item in dart_payload.get("events_by_stock", {}).values())

    naver_payload = {"generated_at": utc_now(), "source": "naver_news", "news_by_stock": {}}
    naver_status = "skipped"
    if args.price_only or args.skip_naver:
        warnings.append("Naver backfill skipped.")
    else:
        try:
            naver_payload, naver_diag = collect_naver_history(max_stocks, args.days, args.sleep)
            naver_status = "warning" if not naver_payload.get("news_by_stock") else "ok"
        except Exception as exc:
            naver_status = "failed"
            warnings.append(f"Naver backfill failed: {sanitize_error(exc)}")
    save_json(BACKFILL_NAVER_NEWS_HISTORY_PATH, naver_payload)
    naver_rows = sum(len(item.get("items", [])) for item in naver_payload.get("news_by_stock", {}).values())

    yahoo_payload = {"generated_at": utc_now(), "source": "yfinance", "tickers": {}, "features": {}}
    yahoo_status = "skipped"
    if args.price_only or args.skip_yahoo:
        warnings.append("Yahoo backfill skipped.")
    else:
        try:
            yahoo_payload, yahoo_diag = collect_yahoo_history(args.days, args.sleep)
            yahoo_rows = sum(len(item.get("daily", [])) for item in yahoo_payload.get("tickers", {}).values())
            yahoo_status = status_from_diagnostics(yahoo_diag.get("tickers", {}), yahoo_rows)
        except Exception as exc:
            yahoo_status = "failed"
            warnings.append(f"Yahoo backfill failed: {sanitize_error(exc)}")
    save_json(BACKFILL_YAHOO_GLOBAL_HISTORY_PATH, yahoo_payload)
    yahoo_rows = sum(len(item.get("daily", [])) for item in yahoo_payload.get("tickers", {}).values())

    status_payload = {
        "generated_at": utc_now(),
        "days": args.days,
        "universe_count": max_stocks,
        "status": {
            "prices": status_from_diagnostics(price_diag, price_rows),
            "flows": status_from_diagnostics(flow_diag, flow_rows),
            "dart": dart_status,
            "naver": naver_status,
            "yahoo": yahoo_status,
        },
        "row_counts": {
            "prices": price_rows,
            "flows": flow_rows,
            "dart_events": dart_rows,
            "naver_news": naver_rows,
            "yahoo_proxy_rows": yahoo_rows,
        },
        "warnings": warnings,
        "order_enabled": False,
    }
    save_json(BACKFILL_STATUS_PATH, status_payload)
    BACKFILL_MARKET_HISTORY_REPORT_PATH.write_text(build_report(status_payload), encoding="utf-8")
    return status_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill compact historical market data for point-in-time datasets.")
    parser.add_argument("--days", type=int, default=252)
    parser.add_argument("--max-stocks", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.7)
    parser.add_argument("--include-dart", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-naver", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-yahoo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-dart", action="store_true")
    parser.add_argument("--skip-naver", action="store_true")
    parser.add_argument("--skip-yahoo", action="store_true")
    parser.add_argument("--price-only", action="store_true")
    args = parser.parse_args()
    args.skip_dart = args.skip_dart or not args.include_dart
    args.skip_naver = args.skip_naver or not args.include_naver
    args.skip_yahoo = args.skip_yahoo or not args.include_yahoo
    return args


def main() -> int:
    args = parse_args()
    result = run_backfill(args)
    print(f"Saved prices: {BACKFILL_PRICES_DAILY_HISTORY_PATH}")
    print(f"Saved flows: {BACKFILL_FLOWS_DAILY_HISTORY_PATH}")
    print(f"Saved status: {BACKFILL_STATUS_PATH}")
    print(f"Saved report: {BACKFILL_MARKET_HISTORY_REPORT_PATH}")
    print(f"status={result.get('status')}")
    print(f"row_counts={result.get('row_counts')}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
