from __future__ import annotations

import argparse
import html
import math
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

from policylink.paths import (
    YAHOO_FINANCE_SYNC_DIAGNOSTIC_PATH,
    YAHOO_GLOBAL_FEATURES_PATH,
    YAHOO_GLOBAL_FEATURES_REPORT_PATH,
    YAHOO_MARKET_DATA_PATH,
    YAHOO_NEWS_PATH,
    ensure_project_dirs,
)
from policylink.utils import parse_number, save_json


YAHOO_SOURCE_NOTE = (
    "yfinance is used for internal research/education only and is not an "
    "officially affiliated Yahoo Finance client."
)

PROXY_TICKERS: List[Dict[str, Any]] = [
    {"ticker": "QQQ", "label": "미국 성장주/나스닥 proxy", "mapped_sectors": ["core_market", "platform_internet"]},
    {"ticker": "SOXX", "label": "미국 반도체 proxy", "mapped_sectors": ["semiconductor_battery"]},
    {"ticker": "SMH", "label": "글로벌 반도체 proxy", "mapped_sectors": ["semiconductor_battery"]},
    {"ticker": "TLT", "label": "미국 장기채/금리 민감 proxy", "mapped_sectors": ["bond_cash_like", "financial_value"]},
    {"ticker": "IEF", "label": "미국 중기채 proxy", "mapped_sectors": ["bond_cash_like", "financial_value"]},
    {"ticker": "UUP", "label": "달러 강세 proxy", "mapped_sectors": ["korea_macro_policy", "core_market"]},
    {"ticker": "GLD", "label": "금/안전자산 proxy", "mapped_sectors": ["bond_cash_like", "geopolitical_risk"]},
    {"ticker": "USO", "label": "원유 proxy", "mapped_sectors": ["energy_infra"]},
    {"ticker": "XLE", "label": "에너지 proxy", "mapped_sectors": ["energy_infra"]},
    {"ticker": "XLF", "label": "금융 proxy", "mapped_sectors": ["financial_value"]},
    {"ticker": "KBE", "label": "미국 은행 proxy", "mapped_sectors": ["financial_value"]},
    {"ticker": "EWY", "label": "한국시장 ETF proxy", "mapped_sectors": ["core_market"]},
    {"ticker": "^VIX", "label": "변동성 proxy", "mapped_sectors": ["bond_cash_like", "geopolitical_risk"]},
    {"ticker": "KRW=X", "label": "USD/KRW proxy", "mapped_sectors": ["korea_macro_policy", "core_market"]},
]

PROXY_BY_TICKER = {item["ticker"]: item for item in PROXY_TICKERS}

PROXY_GROUPS: Dict[str, Dict[str, Any]] = {
    "semiconductor_proxy": {"label": "반도체 proxy", "tickers": ["SOXX", "SMH"]},
    "rate_proxy": {"label": "금리 proxy", "tickers": ["TLT", "IEF"]},
    "dollar_proxy": {"label": "달러/환율 proxy", "tickers": ["UUP", "KRW=X"]},
    "energy_proxy": {"label": "에너지 proxy", "tickers": ["USO", "XLE"]},
    "volatility_proxy": {"label": "변동성 proxy", "tickers": ["^VIX"]},
    "korea_proxy": {"label": "한국시장 proxy", "tickers": ["EWY"]},
}

POSITIVE_KEYWORDS = [
    "rally",
    "beat",
    "growth",
    "strong",
    "upgrade",
    "demand",
    "record",
    "profit",
    "earnings beat",
    "ai",
    "chip demand",
    "rate cut",
    "stimulus",
]

NEGATIVE_KEYWORDS = [
    "miss",
    "decline",
    "weak",
    "downgrade",
    "lawsuit",
    "investigation",
    "inflation",
    "rate hike",
    "recession",
    "war",
    "conflict",
    "tariff",
    "sanction",
    "oil spike",
    "volatility",
]

RISK_KEYWORDS = [
    "inflation",
    "rate",
    "fed",
    "yield",
    "dollar",
    "oil",
    "war",
    "conflict",
    "tariff",
    "sanction",
    "volatility",
    "vix",
]

RISK_RISE_IS_BAD = {"^VIX", "UUP", "KRW=X", "GLD"}
BOND_RATE_PROXIES = {"TLT", "IEF"}


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def clean_text(value: Any, max_len: Optional[int] = None) -> Optional[str]:
    if value is None:
        return None

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    if max_len and len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
    return text


def nested_get(obj: Any, path: List[Any], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif isinstance(cur, list) and isinstance(key, int) and 0 <= key < len(cur):
            cur = cur[key]
        else:
            return default
        if cur is None:
            return default
    return cur


def parse_publish_time(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
        except Exception:
            return None

    text = str(value).strip()
    if not text:
        return None

    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except Exception:
        pass

    try:
        parsed = parsedate_to_datetime(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except Exception:
        return text


def keyword_count(text: str, keywords: List[str]) -> int:
    lowered = text.lower()
    count = 0
    for keyword in keywords:
        if keyword.lower() in lowered:
            count += 1
    return count


def round_or_none(value: Any, digits: int = 6) -> Optional[float]:
    try:
        if value is None:
            return None
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return round(numeric, digits)
    except Exception:
        return None


def series_return(close, periods: int) -> Optional[float]:
    if close is None or len(close) <= periods:
        return None

    latest = float(close.iloc[-1])
    previous = float(close.iloc[-periods - 1])
    if previous == 0:
        return None
    return latest / previous - 1.0


def compute_price_features(history) -> Dict[str, Any]:
    if history is None or history.empty or "Close" not in history:
        return {
            "latest_date": None,
            "latest_close": None,
            "latest_volume": None,
            "return_1d": None,
            "return_5d": None,
            "return_20d": None,
            "volatility_20d": None,
            "ma20_gap": None,
            "ma60_gap": None,
            "drawdown_20d": None,
            "trend_label": "neutral",
            "risk_label": "normal",
            "daily": [],
        }

    history = history.dropna(subset=["Close"]).copy()
    if history.empty:
        return compute_price_features(None)

    close = history["Close"].astype(float)
    volume = history["Volume"].fillna(0) if "Volume" in history else None
    daily_return = close.pct_change()

    latest_date = history.index[-1]
    if hasattr(latest_date, "date"):
        latest_date_text = latest_date.date().isoformat()
    else:
        latest_date_text = str(latest_date)[:10]

    latest_close = float(close.iloc[-1])
    latest_volume = int(float(volume.iloc[-1])) if volume is not None and len(volume) else None

    ret_1d = series_return(close, 1)
    ret_5d = series_return(close, 5)
    ret_20d = series_return(close, 20)
    volatility_20d = float(daily_return.tail(20).std()) if len(daily_return.dropna()) >= 5 else None

    ma20 = float(close.tail(20).mean()) if len(close) >= 20 else None
    ma60 = float(close.tail(60).mean()) if len(close) >= 60 else None
    high_20 = float(close.tail(20).max()) if len(close) >= 1 else None

    ma20_gap = latest_close / ma20 - 1.0 if ma20 else None
    ma60_gap = latest_close / ma60 - 1.0 if ma60 else None
    drawdown_20d = latest_close / high_20 - 1.0 if high_20 else None

    if ma20_gap is not None and ret_5d is not None and ma20_gap > 0.03 and ret_5d > 0:
        trend_label = "uptrend"
    elif ma20_gap is not None and ret_5d is not None and ma20_gap < -0.03 and ret_5d < 0:
        trend_label = "downtrend"
    else:
        trend_label = "neutral"

    if volatility_20d is not None and volatility_20d >= 0.04:
        risk_label = "high_volatility"
    elif volatility_20d is not None and volatility_20d <= 0.015:
        risk_label = "low_volatility"
    else:
        risk_label = "normal"

    daily_rows = []
    for idx, row in history.tail(120).iterrows():
        if hasattr(idx, "date"):
            date_text = idx.date().isoformat()
        else:
            date_text = str(idx)[:10]
        daily_rows.append({
            "date": date_text,
            "close": round_or_none(row.get("Close"), 4),
            "volume": int(parse_number(row.get("Volume"), 0)) if "Volume" in row else None,
        })

    return {
        "latest_date": latest_date_text,
        "latest_close": round_or_none(latest_close, 4),
        "latest_volume": latest_volume,
        "return_1d": round_or_none(ret_1d),
        "return_5d": round_or_none(ret_5d),
        "return_20d": round_or_none(ret_20d),
        "volatility_20d": round_or_none(volatility_20d),
        "ma20_gap": round_or_none(ma20_gap),
        "ma60_gap": round_or_none(ma60_gap),
        "drawdown_20d": round_or_none(drawdown_20d),
        "trend_label": trend_label,
        "risk_label": risk_label,
        "daily": daily_rows,
    }


def extract_related_tickers(raw: Dict[str, Any], ticker: str) -> List[str]:
    related = raw.get("relatedTickers")
    if not related:
        related = nested_get(raw, ["content", "relatedTickers"])
    if not related:
        related = nested_get(raw, ["content", "finance", "stockTickers"])

    result = []
    if isinstance(related, list):
        for item in related:
            if isinstance(item, dict):
                symbol = item.get("symbol") or item.get("ticker") or item.get("name")
            else:
                symbol = item
            if symbol:
                result.append(str(symbol))

    if not result:
        result = [ticker, "ticker_proxy_news"]

    deduped = []
    for item in result:
        if item not in deduped:
            deduped.append(item)
    return deduped[:10]


def compact_news_item(raw: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    title = clean_text(raw.get("title") or nested_get(raw, ["content", "title"]), 220)
    summary = clean_text(
        raw.get("summary")
        or raw.get("description")
        or nested_get(raw, ["content", "summary"])
        or nested_get(raw, ["content", "description"]),
        360,
    )
    provider = clean_text(
        raw.get("publisher")
        or raw.get("provider")
        or nested_get(raw, ["content", "provider", "displayName"])
        or nested_get(raw, ["content", "provider", "name"]),
        120,
    )
    url = (
        raw.get("link")
        or raw.get("url")
        or nested_get(raw, ["clickThroughUrl", "url"])
        or nested_get(raw, ["content", "canonicalUrl", "url"])
        or nested_get(raw, ["content", "clickThroughUrl", "url"])
    )
    publish_time = parse_publish_time(
        raw.get("providerPublishTime")
        or raw.get("publishTime")
        or raw.get("pubDate")
        or nested_get(raw, ["content", "pubDate"])
        or nested_get(raw, ["content", "displayTime"])
    )

    return {
        "ticker": ticker,
        "title": title,
        "summary": summary,
        "provider": provider,
        "url": str(url).strip() if url else None,
        "published_at": publish_time,
        "related_tickers": extract_related_tickers(raw, ticker),
    }


def price_directional_score(ticker: str, price: Dict[str, Any]) -> float:
    ret_5d = parse_number(price.get("return_5d"), 0.0)
    ret_20d = parse_number(price.get("return_20d"), 0.0)
    trend_label = str(price.get("trend_label") or "neutral")
    risk_label = str(price.get("risk_label") or "normal")

    score = 50.0

    if ticker in RISK_RISE_IS_BAD:
        score -= ret_5d * 250.0
        score -= ret_20d * 120.0
        if trend_label == "uptrend":
            score -= 8.0
        elif trend_label == "downtrend":
            score += 6.0
    elif ticker in BOND_RATE_PROXIES:
        score += ret_5d * 180.0
        score += ret_20d * 90.0
        if trend_label == "uptrend":
            score += 8.0
        elif trend_label == "downtrend":
            score -= 10.0
    else:
        score += ret_5d * 180.0
        score += ret_20d * 90.0
        if trend_label == "uptrend":
            score += 8.0
        elif trend_label == "downtrend":
            score -= 8.0

    if risk_label == "high_volatility":
        score -= 6.0
    elif risk_label == "low_volatility":
        score += 2.0

    return clamp(score)


def proxy_risk_score(ticker: str, price: Dict[str, Any], risk_keyword_count: int) -> float:
    ret_5d = parse_number(price.get("return_5d"), 0.0)
    ret_20d = parse_number(price.get("return_20d"), 0.0)
    volatility = parse_number(price.get("volatility_20d"), 0.0)

    risk = 0.0

    if ticker == "^VIX":
        if ret_5d > 0.10:
            risk += 40.0
        if ret_20d > 0.20:
            risk += 20.0

    if ticker in {"UUP", "KRW=X"}:
        if ret_5d > 0.02:
            risk += 20.0
        if ret_20d > 0.04:
            risk += 20.0

    if ticker in BOND_RATE_PROXIES:
        if ret_5d < -0.03:
            risk += 25.0
        if ret_20d < -0.06:
            risk += 25.0

    if ticker in {"USO", "XLE"}:
        if ret_5d > 0.05:
            risk += 20.0
        if ret_20d > 0.10:
            risk += 20.0

    if ticker == "GLD" and ret_5d > 0.04:
        risk += 10.0

    if volatility >= 0.04:
        risk += 15.0

    risk += min(20.0, risk_keyword_count * 2.0)
    return round(clamp(risk), 2)


def score_news(news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    positive_count = 0
    negative_count = 0
    risk_count = 0

    for item in news_items:
        text = " ".join([
            str(item.get("title") or ""),
            str(item.get("summary") or ""),
        ])
        positive_count += keyword_count(text, POSITIVE_KEYWORDS)
        negative_count += keyword_count(text, NEGATIVE_KEYWORDS)
        risk_count += keyword_count(text, RISK_KEYWORDS)

    sentiment = 50.0 + positive_count * 2.0 - negative_count * 3.0 - risk_count * 1.0
    return {
        "positive_keyword_count": positive_count,
        "negative_keyword_count": negative_count,
        "risk_keyword_count": risk_count,
        "news_sentiment_score": round(clamp(sentiment), 2),
    }


def build_ticker_feature(ticker: str, price: Dict[str, Any], news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    meta = PROXY_BY_TICKER[ticker]
    news_score = score_news(news_items)
    price_score = price_directional_score(ticker, price)
    risk_score = proxy_risk_score(ticker, price, news_score["risk_keyword_count"])
    risk_adjustment = 100.0 - risk_score
    global_signal_score = (
        price_score * 0.60
        + news_score["news_sentiment_score"] * 0.25
        + risk_adjustment * 0.15
    )
    global_signal_score = round(clamp(global_signal_score), 2)

    if global_signal_score >= 70:
        global_signal_label = "supportive"
    elif global_signal_score >= 45:
        global_signal_label = "neutral"
    else:
        global_signal_label = "risk_off"

    latest_news = news_items[0] if news_items else {}

    return {
        "ticker": ticker,
        "label": meta["label"],
        "mapped_sectors": meta["mapped_sectors"],
        "latest_date": price.get("latest_date"),
        "latest_close": price.get("latest_close"),
        "return_1d": price.get("return_1d"),
        "return_5d": price.get("return_5d"),
        "return_20d": price.get("return_20d"),
        "volatility_20d": price.get("volatility_20d"),
        "ma20_gap": price.get("ma20_gap"),
        "ma60_gap": price.get("ma60_gap"),
        "drawdown_20d": price.get("drawdown_20d"),
        "trend_label": price.get("trend_label", "neutral"),
        "risk_label": price.get("risk_label", "normal"),
        "price_signal_score": round(price_score, 2),
        "news_count": len(news_items),
        **news_score,
        "proxy_risk_score": risk_score,
        "global_signal_score": global_signal_score,
        "global_signal_label": global_signal_label,
        "latest_news_title": latest_news.get("title"),
        "latest_news_url": latest_news.get("url"),
    }


def average(values: List[float], default: float = 50.0) -> float:
    values = [float(value) for value in values if value is not None]
    if not values:
        return default
    return sum(values) / len(values)


def build_sector_scores(features: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for feature in features.values():
        for sector in feature.get("mapped_sectors", []):
            grouped.setdefault(sector, []).append(feature)

    result = {}
    for sector, items in grouped.items():
        score = average([parse_number(item.get("global_signal_score"), 50.0) for item in items])
        risk_score = average([parse_number(item.get("proxy_risk_score"), 0.0) for item in items], default=0.0)
        result[sector] = {
            "global_signal_score": round(clamp(score), 2),
            "risk_score": round(clamp(risk_score), 2),
            "proxy_count": len(items),
            "related_proxies": [item["ticker"] for item in items],
            "related_proxy_labels": [item["label"] for item in items],
        }

    return result


def build_proxy_group_scores(features: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result = {}
    for group_key, group in PROXY_GROUPS.items():
        items = [features[ticker] for ticker in group["tickers"] if ticker in features]
        score = average([parse_number(item.get("global_signal_score"), 50.0) for item in items])
        risk_score = average([parse_number(item.get("proxy_risk_score"), 0.0) for item in items], default=0.0)
        result[group_key] = {
            "label": group["label"],
            "tickers": group["tickers"],
            "global_signal_score": round(clamp(score), 2),
            "risk_score": round(clamp(risk_score), 2),
            "proxy_count": len(items),
            "proxies": [
                {
                    "ticker": item["ticker"],
                    "label": item["label"],
                    "global_signal_score": item.get("global_signal_score"),
                    "global_signal_label": item.get("global_signal_label"),
                    "return_5d": item.get("return_5d"),
                    "trend_label": item.get("trend_label"),
                    "news_sentiment_score": item.get("news_sentiment_score"),
                    "proxy_risk_score": item.get("proxy_risk_score"),
                }
                for item in items
            ],
        }
    return result


def build_risk_warnings(features: Dict[str, Dict[str, Any]]) -> List[str]:
    warnings = []
    vix = features.get("^VIX")
    uup = features.get("UUP")
    krw = features.get("KRW=X")
    tlt = features.get("TLT")
    uso = features.get("USO")

    if vix and parse_number(vix.get("return_5d"), 0.0) > 0.10:
        warnings.append("^VIX 5일 상승률이 10%를 넘어 변동성 리스크를 점검해야 합니다.")
    if uup and parse_number(uup.get("return_5d"), 0.0) > 0.02:
        warnings.append("UUP 상승으로 달러 강세 압력이 감지됩니다.")
    if krw and parse_number(krw.get("return_5d"), 0.0) > 0.02:
        warnings.append("USD/KRW 상승으로 원화 약세 리스크가 감지됩니다.")
    if tlt and parse_number(tlt.get("return_5d"), 0.0) < -0.03:
        warnings.append("TLT 약세로 금리 상승 압력을 점검해야 합니다.")
    if uso and parse_number(uso.get("return_5d"), 0.0) > 0.05:
        warnings.append("USO 급등으로 유가 리스크를 점검해야 합니다.")

    return warnings


def build_markdown_report(features: Dict[str, Dict[str, Any]], group_scores: Dict[str, Dict[str, Any]], diagnostics: Dict[str, Any]) -> str:
    lines = [
        "# Yahoo Finance 글로벌 Proxy 피처",
        "",
        f"- 생성 시각 UTC: {datetime.now(timezone.utc).isoformat()}",
        "- source: yfinance",
        f"- 사용 원칙: {YAHOO_SOURCE_NOTE}",
        "- 뉴스 원문 전체는 저장하지 않고 제목/요약/URL/발행시각 중심의 compact metadata만 사용합니다.",
        "",
        "## 1. Proxy 그룹 요약",
    ]

    for group in group_scores.values():
        lines.append(
            f"- {group['label']} ({'/'.join(group['tickers'])}): "
            f"score={group['global_signal_score']} / risk={group['risk_score']} / proxies={group['proxy_count']}"
        )

    lines.extend(["", "## 2. Ticker별 피처"])
    for ticker, feature in features.items():
        latest_news_title = feature.get("latest_news_title") or "최신 뉴스 없음"
        lines.append(
            f"- {ticker} / {feature.get('label')}: "
            f"global={feature.get('global_signal_score')}({feature.get('global_signal_label')}) "
            f"/ risk={feature.get('proxy_risk_score')} "
            f"/ 5D={feature.get('return_5d')} "
            f"/ trend={feature.get('trend_label')} "
            f"/ news_sentiment={feature.get('news_sentiment_score')} "
            f"/ latest_news={latest_news_title}"
        )

    warning_count = len(diagnostics.get("warnings", []))
    error_count = len(diagnostics.get("errors", []))
    lines.extend([
        "",
        "## 3. 진단",
        f"- warnings: {warning_count}",
        f"- errors: {error_count}",
    ])

    return "\n".join(lines)


def collect_price_history(yf_module, ticker: str, days: int):
    fetch_days = max(days, 90)
    proxy = yf_module.Ticker(ticker)
    return proxy.history(period=f"{fetch_days}d", interval="1d", auto_adjust=False)


def collect_news(yf_module, ticker: str, count: int) -> List[Dict[str, Any]]:
    proxy = yf_module.Ticker(ticker)
    raw_items = []

    if hasattr(proxy, "get_news"):
        try:
            raw_items = proxy.get_news(count=count, tab="news") or []
        except TypeError:
            raw_items = proxy.get_news(count=count) or []
    elif hasattr(proxy, "news"):
        raw_items = proxy.news or []

    if not isinstance(raw_items, list):
        return []

    compact_items = []
    seen = set()
    for raw in raw_items[:count]:
        if not isinstance(raw, dict):
            continue
        item = compact_news_item(raw, ticker)
        key = item.get("url") or item.get("title")
        if not key or key in seen:
            continue
        seen.add(key)
        compact_items.append(item)

    return compact_items


def collect_yahoo_finance(days: int, news_count: int, max_tickers: int, sleep_seconds: float, skip_news: bool, skip_prices: bool) -> Dict[str, Any]:
    ensure_project_dirs()

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Run: .venv/bin/pip install yfinance pandas") from exc

    generated_at = datetime.now(timezone.utc).isoformat()
    selected = PROXY_TICKERS[:max_tickers]
    market_data: Dict[str, Any] = {
        "generated_at": generated_at,
        "source": "yfinance",
        "usage_note": YAHOO_SOURCE_NOTE,
        "tickers": {},
    }
    yahoo_news: Dict[str, Any] = {
        "generated_at": generated_at,
        "source": "yfinance",
        "usage_note": YAHOO_SOURCE_NOTE,
        "tickers": {},
    }
    diagnostics: Dict[str, Any] = {
        "generated_at": generated_at,
        "source": "yfinance",
        "requested_days": days,
        "news_count": news_count,
        "max_tickers": max_tickers,
        "skip_news": skip_news,
        "skip_prices": skip_prices,
        "warnings": [],
        "errors": [],
        "ticker_status": {},
    }

    features: Dict[str, Dict[str, Any]] = {}

    for meta in selected:
        ticker = meta["ticker"]
        status = {
            "price_ok": False,
            "news_ok": False,
            "feature_created": False,
        }

        price_feature = compute_price_features(None)
        if not skip_prices:
            try:
                history = collect_price_history(yf, ticker, days)
                price_feature = compute_price_features(history)
                market_data["tickers"][ticker] = {
                    "ticker": ticker,
                    "label": meta["label"],
                    "mapped_sectors": meta["mapped_sectors"],
                    "daily": price_feature.pop("daily", []),
                    "price_features": dict(price_feature),
                }
                status["price_ok"] = bool(price_feature.get("latest_close") is not None)
                if not status["price_ok"]:
                    diagnostics["warnings"].append({"ticker": ticker, "stage": "price", "message": "no usable close data"})
            except Exception as exc:
                diagnostics["errors"].append({"ticker": ticker, "stage": "price", "message": str(exc)[:300]})
                market_data["tickers"][ticker] = {
                    "ticker": ticker,
                    "label": meta["label"],
                    "mapped_sectors": meta["mapped_sectors"],
                    "daily": [],
                    "price_features": dict(price_feature),
                }
        else:
            market_data["tickers"][ticker] = {
                "ticker": ticker,
                "label": meta["label"],
                "mapped_sectors": meta["mapped_sectors"],
                "daily": [],
                "price_features": dict(price_feature),
            }

        news_items: List[Dict[str, Any]] = []
        if not skip_news:
            try:
                news_items = collect_news(yf, ticker, news_count)
                status["news_ok"] = True
            except Exception as exc:
                diagnostics["errors"].append({"ticker": ticker, "stage": "news", "message": str(exc)[:300]})
        yahoo_news["tickers"][ticker] = {
            "ticker": ticker,
            "label": meta["label"],
            "news": news_items,
        }

        feature = build_ticker_feature(ticker, price_feature, news_items)
        features[ticker] = feature
        status["feature_created"] = True
        diagnostics["ticker_status"][ticker] = status

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    sector_scores = build_sector_scores(features)
    proxy_group_scores = build_proxy_group_scores(features)
    diagnostics["warnings"].extend(build_risk_warnings(features))
    diagnostics["feature_count"] = len(features)

    global_features = {
        "generated_at": generated_at,
        "source": "yfinance",
        "usage_note": YAHOO_SOURCE_NOTE,
        "features": features,
        "sector_global_scores": sector_scores,
        "proxy_group_scores": proxy_group_scores,
        "risk_warnings": diagnostics["warnings"],
    }

    save_json(YAHOO_MARKET_DATA_PATH, market_data)
    save_json(YAHOO_NEWS_PATH, yahoo_news)
    save_json(YAHOO_GLOBAL_FEATURES_PATH, global_features)
    save_json(YAHOO_FINANCE_SYNC_DIAGNOSTIC_PATH, diagnostics)
    YAHOO_GLOBAL_FEATURES_REPORT_PATH.write_text(
        build_markdown_report(features, proxy_group_scores, diagnostics),
        encoding="utf-8",
    )

    return {
        "market_data": market_data,
        "news": yahoo_news,
        "features": global_features,
        "diagnostics": diagnostics,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Yahoo Finance global proxy data via yfinance.")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--news-count", type=int, default=10)
    parser.add_argument("--max-tickers", type=int, default=14)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument("--skip-prices", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    max_tickers = max(1, min(args.max_tickers, len(PROXY_TICKERS)))

    try:
        result = collect_yahoo_finance(
            days=max(1, args.days),
            news_count=max(0, args.news_count),
            max_tickers=max_tickers,
            sleep_seconds=max(0.0, args.sleep),
            skip_news=args.skip_news,
            skip_prices=args.skip_prices,
        )
    except Exception as exc:
        print(f"Yahoo Finance collection failed: {exc}")
        return 1

    features = result["features"].get("features", {})
    print(f"Saved market data: {YAHOO_MARKET_DATA_PATH}")
    print(f"Saved news: {YAHOO_NEWS_PATH}")
    print(f"Saved features: {YAHOO_GLOBAL_FEATURES_PATH}")
    print(f"Saved diagnostic: {YAHOO_FINANCE_SYNC_DIAGNOSTIC_PATH}")
    print(f"Saved report: {YAHOO_GLOBAL_FEATURES_REPORT_PATH}")
    print(f"Feature rows: {len(features)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
