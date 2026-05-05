from __future__ import annotations

import argparse
import html
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from policylink.config import load_naver_settings
from policylink.paths import (
    DATA_DIR,
    NAVER_NEWS_FEATURES_REPORT_PATH,
    NAVER_NEWS_PATH,
    NAVER_NEWS_SYNC_DIAGNOSTIC_PATH,
    NEWS_EVENT_FEATURES_PATH,
    REPORTS_DIR,
)
from policylink.universe import universe_for_market_data
from policylink.utils import normalize_code, save_json


NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"

QUERY_MAP = {
    "005930": ["삼성전자 반도체 HBM", "삼성전자 실적 투자"],
    "000660": ["SK하이닉스 HBM 반도체", "SK하이닉스 실적 투자"],
    "005380": ["현대차 전기차 수출", "현대차 실적 투자"],
    "000270": ["기아 전기차 수출", "기아 실적 투자"],
    "105560": ["KB금융 밸류업 배당", "KB금융 실적 투자"],
    "055550": ["신한지주 밸류업 배당", "신한지주 실적 투자"],
    "035420": ["NAVER AI 플랫폼", "네이버 실적 투자"],
    "035720": ["카카오 플랫폼 규제", "카카오 실적 투자"],
    "012450": ["한화에어로스페이스 방산 수출", "한화에어로스페이스 수주"],
    "034020": ["두산에너빌리티 원전 수주", "두산에너빌리티 실적 투자"],
}

POSITIVE_KEYWORDS = [
    "실적",
    "영업이익",
    "매출",
    "수주",
    "계약",
    "공급",
    "투자",
    "증설",
    "HBM",
    "AI",
    "배당",
    "자사주",
    "밸류업",
    "수출",
    "흑자",
    "목표가 상향",
    "호실적",
]

NEGATIVE_KEYWORDS = [
    "적자",
    "손실",
    "감익",
    "하락",
    "급락",
    "소송",
    "제재",
    "규제",
    "과징금",
    "리콜",
    "파업",
    "감산",
    "목표가 하향",
    "부진",
    "불확실",
    "중단",
    "취소",
]

RISK_KEYWORDS = [
    "중동",
    "환율",
    "금리",
    "유가",
    "관세",
    "공급망",
    "중국",
    "미국",
    "수출통제",
    "공매도",
]

TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
SENSITIVE_QUERY_RE = re.compile(r"(X-Naver-Client-(?:Id|Secret)\s*[:=]\s*)[^\s,}]+", re.IGNORECASE)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = TAG_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def normalize_title(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    return text


def parse_pub_date(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(str(value))
    except Exception:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def safe_error_message(exc: Exception) -> str:
    text = SENSITIVE_QUERY_RE.sub(r"\1[REDACTED]", str(exc))
    return text[:500]


def keyword_count(text: str, keywords: Iterable[str]) -> int:
    lowered = text.lower()
    count = 0
    for keyword in keywords:
        if keyword.lower() in lowered:
            count += 1
    return count


def compact_news_item(raw: Dict[str, Any], query: str, stock_code: str) -> Dict[str, Any]:
    title = clean_text(raw.get("title"))
    description = clean_text(raw.get("description"))
    originallink = str(raw.get("originallink") or "").strip()
    link = str(raw.get("link") or "").strip()
    pub_date = str(raw.get("pubDate") or "").strip()

    text = f"{title} {description}"
    return {
        "title": title,
        "description": description,
        "originallink": originallink,
        "link": link,
        "pubDate": pub_date,
        "pubDate_utc": parse_pub_date(pub_date).isoformat() if parse_pub_date(pub_date) else None,
        "query": query,
        "stock_code": stock_code,
        "positive_keyword_count": keyword_count(text, POSITIVE_KEYWORDS),
        "negative_keyword_count": keyword_count(text, NEGATIVE_KEYWORDS),
        "risk_keyword_count": keyword_count(text, RISK_KEYWORDS),
    }


def dedupe_news(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_links = set()
    seen_titles = set()
    deduped = []

    for item in items:
        link_key = item.get("originallink") or item.get("link")
        title_key = normalize_title(item.get("title"))

        if link_key and link_key in seen_links:
            continue
        if title_key and title_key in seen_titles:
            continue

        if link_key:
            seen_links.add(link_key)
        if title_key:
            seen_titles.add(title_key)

        deduped.append(item)

    return deduped


def fetch_query(settings, query: str, display: int, sort: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    headers = {
        "X-Naver-Client-Id": settings.client_id,
        "X-Naver-Client-Secret": settings.client_secret,
    }
    params = {
        "query": query,
        "display": max(1, min(100, display)),
        "start": 1,
        "sort": sort,
    }

    response = requests.get(NAVER_NEWS_URL, headers=headers, params=params, timeout=15)
    status_code = response.status_code

    diagnostic = {
        "query": query,
        "status_code": status_code,
        "item_count": 0,
        "error": None,
        "stop_collection": False,
    }

    if status_code in {401, 403}:
        diagnostic["error"] = "auth_error"
        diagnostic["stop_collection"] = True
        return [], diagnostic

    if status_code == 429:
        diagnostic["error"] = "rate_limited"
        diagnostic["stop_collection"] = True
        return [], diagnostic

    if status_code >= 400:
        diagnostic["error"] = f"http_{status_code}"
        return [], diagnostic

    try:
        data = response.json()
    except Exception as exc:
        diagnostic["error"] = safe_error_message(exc)
        return [], diagnostic

    items = data.get("items", [])
    if not isinstance(items, list):
        items = []

    diagnostic["item_count"] = len(items)
    return [item for item in items if isinstance(item, dict)], diagnostic


def filter_by_date(
    items: List[Dict[str, Any]],
    end_dt: datetime,
    days: int,
    diagnostics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    start_dt = end_dt - timedelta(days=days)
    kept = []

    for item in items:
        parsed = parse_pub_date(item.get("pubDate"))
        if parsed is None:
            diagnostics["pub_date_parse_failures"] = diagnostics.get("pub_date_parse_failures", 0) + 1
            kept.append(item)
            continue

        if start_dt <= parsed <= end_dt:
            kept.append(item)
        else:
            diagnostics["out_of_range_count"] = diagnostics.get("out_of_range_count", 0) + 1

    return kept


def count_since(items: List[Dict[str, Any]], end_dt: datetime, days: int) -> int:
    start_dt = end_dt - timedelta(days=days)
    count = 0
    for item in items:
        parsed = parse_pub_date(item.get("pubDate"))
        if parsed and start_dt <= parsed <= end_dt:
            count += 1
    return count


def items_since(items: List[Dict[str, Any]], end_dt: datetime, days: int) -> List[Dict[str, Any]]:
    start_dt = end_dt - timedelta(days=days)
    result = []
    for item in items:
        parsed = parse_pub_date(item.get("pubDate"))
        if parsed and start_dt <= parsed <= end_dt:
            result.append(item)
    return result


def build_feature(stock: Dict[str, str], query_count: int, news_items: List[Dict[str, Any]], end_dt: datetime) -> Dict[str, Any]:
    sorted_items = sorted(
        news_items,
        key=lambda item: parse_pub_date(item.get("pubDate")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    items_7d = items_since(sorted_items, end_dt, 7)

    positive_count = sum(int(item.get("positive_keyword_count") or 0) for item in items_7d)
    negative_count = sum(int(item.get("negative_keyword_count") or 0) for item in items_7d)
    risk_count = sum(int(item.get("risk_keyword_count") or 0) for item in items_7d)

    sentiment_score = 50.0
    sentiment_score += positive_count * 2
    sentiment_score -= negative_count * 3
    sentiment_score -= risk_count * 1
    sentiment_score = round(clamp(sentiment_score, 0.0, 100.0), 1)

    if sentiment_score >= 65:
        label = "positive_news_flow"
    elif sentiment_score >= 45:
        label = "neutral"
    else:
        label = "negative_news_flow"

    news_count_7d = count_since(sorted_items, end_dt, 7)
    attention_score = min(100.0, news_count_7d * 5 + positive_count * 2 + negative_count * 2 + risk_count)

    latest = sorted_items[0] if sorted_items else {}

    return {
        "stock_code": stock["code"],
        "stock_name": stock["name"],
        "sector": stock["sector"],
        "query_count": query_count,
        "news_count": len(sorted_items),
        "news_count_1d": count_since(sorted_items, end_dt, 1),
        "news_count_7d": news_count_7d,
        "positive_keyword_count_7d": positive_count,
        "negative_keyword_count_7d": negative_count,
        "risk_keyword_count_7d": risk_count,
        "attention_score": round(attention_score, 1),
        "sentiment_score": sentiment_score,
        "news_label": label,
        "latest_news_date": latest.get("pubDate"),
        "latest_news_title": latest.get("title"),
        "top_news": [
            {
                "title": item.get("title"),
                "description": item.get("description"),
                "originallink": item.get("originallink"),
                "link": item.get("link"),
                "pubDate": item.get("pubDate"),
                "query": item.get("query"),
            }
            for item in sorted_items[:10]
        ],
    }


def build_sector_scores(features: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for feature in features.values():
        sector = feature.get("sector") or "unknown"
        grouped.setdefault(sector, []).append(feature)

    result = {}
    for sector, items in grouped.items():
        if not items:
            continue
        result[sector] = {
            "news_count_7d": sum(int(item.get("news_count_7d") or 0) for item in items),
            "sentiment_score": round(sum(float(item.get("sentiment_score") or 50.0) for item in items) / len(items), 2),
            "attention_score": round(sum(float(item.get("attention_score") or 0.0) for item in items) / len(items), 2),
        }

    return result


def build_markdown(features: Dict[str, Dict[str, Any]], sector_scores: Dict[str, Dict[str, Any]], diagnostics: Dict[str, Any]) -> str:
    lines = [
        "# 네이버 뉴스 이벤트 피처",
        "",
        f"- 생성 시각 UTC: {diagnostics.get('generated_at')}",
        "- source: naver_news",
        f"- 요청 종목 수: {diagnostics.get('requested_stock_count')}",
        f"- feature 종목 수: {diagnostics.get('feature_stock_count')}",
        "",
        "## 종목별 뉴스 feature",
        "",
    ]

    for code, item in sorted(features.items()):
        lines.append(
            f"- {item.get('stock_name')}({code}) "
            f"/ 7D뉴스={item.get('news_count_7d')} "
            f"/ sentiment={item.get('sentiment_score')} "
            f"/ attention={item.get('attention_score')} "
            f"/ label={item.get('news_label')}"
        )
        if item.get("latest_news_title"):
            lines.append(f"  - latest: {item.get('latest_news_title')}")

    lines.extend(["", "## 섹터별 뉴스 점수", ""])
    for sector, score in sorted(sector_scores.items()):
        lines.append(
            f"- {sector}: 7D뉴스={score.get('news_count_7d')} "
            f"/ sentiment={score.get('sentiment_score')} "
            f"/ attention={score.get('attention_score')}"
        )

    lines.extend([
        "",
        "## 주의",
        "",
        "- 기사 본문 scraping은 하지 않습니다.",
        "- 네이버 검색 API 응답의 제목, 요약, URL, 날짜만 compact metadata로 저장합니다.",
        "- 동일 링크 또는 제목으로 보이는 항목은 중복 제거합니다.",
    ])
    return "\n".join(lines)


def collect_naver_news(days: int, max_stocks: int, display: int, sort: str, sleep_seconds: float) -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    settings = load_naver_settings()
    end_dt = datetime.now(timezone.utc)
    stocks = universe_for_market_data()[:max_stocks]

    news_by_stock: Dict[str, Any] = {}
    features: Dict[str, Dict[str, Any]] = {}
    stock_diagnostics: Dict[str, Any] = {}
    stop_all = False

    for stock in stocks:
        code = normalize_code(stock["code"])
        queries = QUERY_MAP.get(code, [stock["name"]])
        raw_items = []
        query_diagnostics = []
        stock_diag = {
            "stock_name": stock["name"],
            "query_count": len(queries),
            "pub_date_parse_failures": 0,
            "out_of_range_count": 0,
            "status": "ok",
        }

        if stop_all:
            stock_diag["status"] = "skipped_after_rate_limit_or_auth_error"
        else:
            for query in queries:
                try:
                    items, diagnostic = fetch_query(settings, query, display=display, sort=sort)
                except Exception as exc:
                    items = []
                    diagnostic = {
                        "query": query,
                        "status_code": None,
                        "item_count": 0,
                        "error": safe_error_message(exc),
                        "stop_collection": False,
                    }

                query_diagnostics.append(diagnostic)

                if diagnostic.get("stop_collection"):
                    stop_all = True
                    stock_diag["status"] = diagnostic.get("error") or "stopped"
                    break

                raw_items.extend(compact_news_item(item, query, code) for item in items)
                time.sleep(max(0.0, sleep_seconds))

        filtered_items = filter_by_date(raw_items, end_dt, days, stock_diag)
        deduped_items = dedupe_news(filtered_items)

        news_by_stock[code] = {
            "stock_code": code,
            "stock_name": stock["name"],
            "sector": stock["sector"],
            "queries": queries,
            "items": deduped_items,
        }
        features[code] = build_feature(stock, len(queries), deduped_items, end_dt)
        stock_diag["raw_item_count"] = len(raw_items)
        stock_diag["deduped_item_count"] = len(deduped_items)
        stock_diag["query_diagnostics"] = query_diagnostics
        stock_diagnostics[code] = stock_diag

        if stop_all:
            break

    sector_scores = build_sector_scores(features)
    generated_at = datetime.now(timezone.utc).isoformat()

    news_payload = {
        "generated_at": generated_at,
        "source": "naver_news",
        "days": days,
        "news": news_by_stock,
    }
    features_payload = {
        "generated_at": generated_at,
        "source": "naver_news",
        "features": features,
        "sector_scores": sector_scores,
    }
    diagnostic_payload = {
        "generated_at": generated_at,
        "source": "naver_news",
        "days": days,
        "display": display,
        "sort": sort,
        "requested_stock_count": len(stocks),
        "feature_stock_count": len(features),
        "stopped_early": stop_all,
        "stock_diagnostics": stock_diagnostics,
    }

    save_json(NAVER_NEWS_PATH, news_payload)
    save_json(NEWS_EVENT_FEATURES_PATH, features_payload)
    save_json(NAVER_NEWS_SYNC_DIAGNOSTIC_PATH, diagnostic_payload)
    NAVER_NEWS_FEATURES_REPORT_PATH.write_text(build_markdown(features, sector_scores, diagnostic_payload), encoding="utf-8")

    return {
        "news": news_payload,
        "features": features_payload,
        "diagnostic": diagnostic_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect compact Naver News Search API features.")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--max-stocks", type=int, default=10)
    parser.add_argument("--display", type=int, default=20)
    parser.add_argument("--sort", choices=["date", "sim"], default="date")
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    try:
        result = collect_naver_news(
            days=args.days,
            max_stocks=args.max_stocks,
            display=args.display,
            sort=args.sort,
            sleep_seconds=args.sleep,
        )
    except Exception as exc:
        print(f"Naver News collection failed: {safe_error_message(exc)}")
        raise SystemExit(1)

    features = result["features"].get("features", {})
    print(f"Saved news: {NAVER_NEWS_PATH}")
    print(f"Saved features: {NEWS_EVENT_FEATURES_PATH}")
    print(f"Saved diagnostic: {NAVER_NEWS_SYNC_DIAGNOSTIC_PATH}")
    print(f"Saved report: {NAVER_NEWS_FEATURES_REPORT_PATH}")
    print(f"Feature rows: {len(features)}")


if __name__ == "__main__":
    main()
