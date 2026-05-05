from __future__ import annotations

import argparse
import json
import re
import time
import zipfile
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree

import requests

from policylink.config import load_dart_api_key
from policylink.paths import (
    DART_CORP_CODES_PATH,
    DART_DISCLOSURES_PATH,
    DART_DISCLOSURES_REPORT_PATH,
    DART_EVENT_FEATURES_PATH,
    DART_SYNC_DIAGNOSTIC_PATH,
    DATA_DIR,
    REPORTS_DIR,
)
from policylink.universe import universe_for_market_data
from policylink.utils import normalize_code, save_json


CORP_CODE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"
DISCLOSURE_LIST_URL = "https://opendart.fss.or.kr/api/list.json"
DART_VIEW_URL = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"
SENSITIVE_QUERY_RE = re.compile(r"(crtfc_key=)[^&\s]+", re.IGNORECASE)

POSITIVE_FLAGS = {
    "contract_supply": ["단일판매", "공급계약", "수주", "계약체결"],
    "new_investment": ["신규시설투자", "시설투자"],
    "treasury_stock": ["자기주식취득", "자사주"],
    "dividend": ["배당", "현금배당"],
    "bonus_issue": ["무상증자"],
}

POSITIVE_GENERAL_KEYWORDS = ["영업실적", "매출액", "영업이익"]

NEGATIVE_FLAGS = {
    "paid_in_capital_increase": ["유상증자"],
    "convertible_bond": ["전환사채", "신주인수권부사채", "사채"],
    "ownership_change": ["최대주주변경"],
    "trading_halt": ["거래정지"],
    "delisting_risk": ["불성실공시", "상장폐지", "관리종목"],
    "lawsuit": ["소송"],
    "embezzlement_breach": ["횡령", "배임"],
    "audit_opinion_risk": ["감사의견"],
    "correction_delay": ["정정", "지연공시"],
}

FEATURE_FLAG_KEYS = [
    "contract_supply",
    "new_investment",
    "treasury_stock",
    "dividend",
    "bonus_issue",
    "paid_in_capital_increase",
    "convertible_bond",
    "ownership_change",
    "trading_halt",
    "delisting_risk",
    "lawsuit",
    "embezzlement_breach",
    "audit_opinion_risk",
    "correction_delay",
]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_error_message(exc: Exception) -> str:
    text = str(exc)
    text = SENSITIVE_QUERY_RE.sub(r"\1[REDACTED]", text)
    return text[:500]


def kst_today() -> datetime:
    return datetime.now(timezone(timedelta(hours=9)))


def yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def parse_yyyymmdd(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y%m%d").replace(tzinfo=timezone(timedelta(hours=9)))
    except ValueError:
        return None


def load_cached_corp_codes() -> Optional[Dict[str, Any]]:
    if not DART_CORP_CODES_PATH.exists():
        return None
    try:
        data = json.loads(DART_CORP_CODES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def parse_corp_code_zip(content: bytes) -> List[Dict[str, str]]:
    with zipfile.ZipFile(BytesIO(content)) as archive:
        xml_names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        if not xml_names:
            raise RuntimeError("OpenDART corpCode response did not contain XML.")
        xml_bytes = archive.read(xml_names[0])

    root = ElementTree.fromstring(xml_bytes)
    rows: List[Dict[str, str]] = []

    for item in root.findall(".//list"):
        corp_code = (item.findtext("corp_code") or "").strip()
        corp_name = (item.findtext("corp_name") or "").strip()
        stock_code = normalize_code(item.findtext("stock_code"))
        modify_date = (item.findtext("modify_date") or "").strip()

        if not corp_code:
            continue

        rows.append({
            "corp_code": corp_code,
            "corp_name": corp_name,
            "stock_code": stock_code,
            "modify_date": modify_date,
        })

    return rows


def download_corp_codes(api_key: str) -> Dict[str, Any]:
    response = requests.get(CORP_CODE_URL, params={"crtfc_key": api_key}, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(f"OpenDART corpCode request failed with HTTP {response.status_code}.")
    rows = parse_corp_code_zip(response.content)

    by_stock_code = {
        row["stock_code"]: row
        for row in rows
        if row.get("stock_code")
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "opendart",
        "row_count": len(rows),
        "listed_stock_count": len(by_stock_code),
        "by_stock_code": by_stock_code,
    }
    save_json(DART_CORP_CODES_PATH, payload)
    return payload


def get_corp_codes(api_key: str, force_refresh: bool) -> Tuple[Dict[str, Any], bool]:
    if not force_refresh:
        cached = load_cached_corp_codes()
        if cached and isinstance(cached.get("by_stock_code"), dict):
            return cached, False

    return download_corp_codes(api_key), True


def classify_title(title: str) -> Tuple[Dict[str, int], bool, bool, bool]:
    flags = {key: 0 for key in FEATURE_FLAG_KEYS}
    positive = False
    negative = False

    for flag, keywords in POSITIVE_FLAGS.items():
        if any(keyword in title for keyword in keywords):
            flags[flag] = 1
            positive = True

    if any(keyword in title for keyword in POSITIVE_GENERAL_KEYWORDS):
        positive = True

    for flag, keywords in NEGATIVE_FLAGS.items():
        if any(keyword in title for keyword in keywords):
            flags[flag] = 1
            negative = True

    return flags, positive, negative, positive or negative


def merge_flags(events: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    merged = {key: 0 for key in FEATURE_FLAG_KEYS}
    for event in events:
        flags = event.get("feature_flags", {})
        if not isinstance(flags, dict):
            continue
        for key in FEATURE_FLAG_KEYS:
            if flags.get(key):
                merged[key] = 1
    return merged


def compact_event(raw: Dict[str, Any], stock_code: str) -> Dict[str, Any]:
    report_nm = str(raw.get("report_nm") or "")
    flags, positive, negative, major = classify_title(report_nm)
    rcept_no = str(raw.get("rcept_no") or "")

    return {
        "rcept_no": rcept_no,
        "rcept_dt": str(raw.get("rcept_dt") or ""),
        "report_nm": report_nm,
        "corp_name": str(raw.get("corp_name") or ""),
        "stock_code": stock_code,
        "rm": str(raw.get("rm") or ""),
        "url": DART_VIEW_URL.format(rcept_no=rcept_no) if rcept_no else None,
        "feature_flags": flags,
        "is_positive": 1 if positive else 0,
        "is_negative": 1 if negative else 0,
        "is_major": 1 if major else 0,
    }


def fetch_disclosures_for_corp(
    api_key: str,
    corp_code: str,
    stock_code: str,
    bgn_de: str,
    end_de: str,
    sleep_seconds: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    page_no = 1
    page_count = 100
    events: List[Dict[str, Any]] = []
    statuses: List[Dict[str, Any]] = []
    stopped_by_rate_limit = False

    while True:
        params = {
            "crtfc_key": api_key,
            "corp_code": corp_code,
            "bgn_de": bgn_de,
            "end_de": end_de,
            "page_no": page_no,
            "page_count": page_count,
        }
        response = requests.get(DISCLOSURE_LIST_URL, params=params, timeout=20)
        if response.status_code >= 400:
            raise RuntimeError(f"OpenDART list request failed with HTTP {response.status_code}.")
        data = response.json()

        status = str(data.get("status") or "")
        message = str(data.get("message") or "")
        statuses.append({
            "page_no": page_no,
            "status": status,
            "message": message,
        })

        if status == "013":
            break

        if status == "020":
            stopped_by_rate_limit = True
            break

        if status != "000":
            break

        raw_items = data.get("list", [])
        if isinstance(raw_items, list):
            events.extend(compact_event(item, stock_code) for item in raw_items if isinstance(item, dict))

        total_page = int(data.get("total_page") or 1)
        if page_no >= total_page:
            break

        page_no += 1
        time.sleep(max(0.0, sleep_seconds))

    time.sleep(max(0.0, sleep_seconds))

    diagnostic = {
        "corp_code": corp_code,
        "status_history": statuses,
        "event_count": len(events),
        "rate_limited": stopped_by_rate_limit,
    }
    return events, diagnostic


def count_recent(events: List[Dict[str, Any]], end_dt: datetime, days: int) -> int:
    start_dt = end_dt - timedelta(days=days)
    count = 0
    for event in events:
        event_dt = parse_yyyymmdd(event.get("rcept_dt"))
        if event_dt and start_dt <= event_dt <= end_dt:
            count += 1
    return count


def events_in_window(events: List[Dict[str, Any]], end_dt: datetime, days: int) -> List[Dict[str, Any]]:
    start_dt = end_dt - timedelta(days=days)
    result = []
    for event in events:
        event_dt = parse_yyyymmdd(event.get("rcept_dt"))
        if event_dt and start_dt <= event_dt <= end_dt:
            result.append(event)
    return result


def build_feature(stock: Dict[str, str], corp_code: Optional[str], events: List[Dict[str, Any]], end_dt: datetime) -> Dict[str, Any]:
    events_sorted = sorted(events, key=lambda item: str(item.get("rcept_dt") or ""), reverse=True)
    events_30d = events_in_window(events_sorted, end_dt, 30)
    flags = merge_flags(events_30d)

    positive_count = sum(1 for event in events_30d if event.get("is_positive"))
    negative_count = sum(1 for event in events_30d if event.get("is_negative"))
    major_count = sum(1 for event in events_30d if event.get("is_major"))

    score = 50.0
    score += positive_count * 8
    score += major_count * 4
    score -= negative_count * 10

    if flags.get("delisting_risk"):
        score -= 30
    if flags.get("trading_halt"):
        score -= 25
    if flags.get("embezzlement_breach"):
        score -= 30
    if flags.get("audit_opinion_risk"):
        score -= 25

    score = round(clamp(score, 0, 100), 1)
    if score >= 70:
        label = "positive_disclosure"
    elif score >= 45:
        label = "neutral"
    else:
        label = "disclosure_risk"

    latest = events_sorted[0] if events_sorted else {}

    return {
        "stock_code": stock["code"],
        "stock_name": stock["name"],
        "corp_code": corp_code,
        "event_count_7d": count_recent(events_sorted, end_dt, 7),
        "event_count_30d": count_recent(events_sorted, end_dt, 30),
        "event_count_90d": count_recent(events_sorted, end_dt, 90),
        "positive_event_count_30d": positive_count,
        "negative_event_count_30d": negative_count,
        "major_event_count_30d": major_count,
        "latest_event_date": latest.get("rcept_dt"),
        "latest_event_title": latest.get("report_nm"),
        "dart_score": score,
        "dart_label": label,
        "feature_flags": flags,
        "events": [
            {
                "rcept_no": event.get("rcept_no"),
                "rcept_dt": event.get("rcept_dt"),
                "report_nm": event.get("report_nm"),
                "corp_name": event.get("corp_name"),
                "stock_code": event.get("stock_code"),
                "rm": event.get("rm"),
                "url": event.get("url"),
            }
            for event in events_sorted[:20]
        ],
    }


def build_markdown(features: Dict[str, Dict[str, Any]], diagnostics: Dict[str, Any]) -> str:
    lines = [
        "# DART 공시 이벤트 요약",
        "",
        f"- 생성 시각 UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- source: opendart",
        f"- 요청 종목 수: {diagnostics.get('requested_stock_count')}",
        f"- 수집 성공 종목 수: {diagnostics.get('feature_stock_count')}",
        f"- corp_code refresh: {diagnostics.get('corp_code_refreshed')}",
        "",
        "## 종목별 공시 feature",
        "",
    ]

    for code, item in sorted(features.items()):
        lines.append(
            f"- {item.get('stock_name')}({code}) "
            f"/ 30D={item.get('event_count_30d')} "
            f"/ positive={item.get('positive_event_count_30d')} "
            f"/ negative={item.get('negative_event_count_30d')} "
            f"/ score={item.get('dart_score')} "
            f"/ label={item.get('dart_label')}"
        )
        latest_title = item.get("latest_event_title")
        latest_date = item.get("latest_event_date")
        if latest_title:
            lines.append(f"  - latest: {latest_date} {latest_title}")

    lines.extend([
        "",
        "## 주의",
        "",
        "- 공시 원문은 저장하지 않고 접수번호, 제목, 날짜, 회사명, 링크용 URL 등 compact metadata만 저장합니다.",
        "- OpenDART 응답 status 013은 조회 결과 없음으로 처리합니다.",
        "- OpenDART 응답 status 020은 요청 제한으로 보고 해당 종목 수집을 중단합니다.",
    ])
    return "\n".join(lines)


def collect_dart(days: int, max_stocks: int, sleep_seconds: float, force_corp_code_refresh: bool = False) -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    api_key = load_dart_api_key()
    corp_codes, refreshed = get_corp_codes(api_key, force_corp_code_refresh)
    by_stock_code = corp_codes.get("by_stock_code", {})
    if not isinstance(by_stock_code, dict):
        by_stock_code = {}

    end_dt = kst_today()
    start_dt = end_dt - timedelta(days=days)
    bgn_de = yyyymmdd(start_dt)
    end_de = yyyymmdd(end_dt)

    stocks = universe_for_market_data()[:max_stocks]
    disclosures: Dict[str, Any] = {}
    features: Dict[str, Dict[str, Any]] = {}
    stock_diagnostics: Dict[str, Any] = {}

    for stock in stocks:
        code = normalize_code(stock["code"])
        corp_info = by_stock_code.get(code)
        corp_code = corp_info.get("corp_code") if isinstance(corp_info, dict) else None

        if not corp_code:
            disclosures[code] = {
                "stock_code": code,
                "stock_name": stock["name"],
                "corp_code": None,
                "events": [],
            }
            features[code] = build_feature(stock, None, [], end_dt)
            stock_diagnostics[code] = {
                "stock_name": stock["name"],
                "corp_code_found": False,
                "event_count": 0,
                "status": "missing_corp_code",
            }
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
            status = "rate_limited" if diagnostic.get("rate_limited") else "ok"
        except Exception as exc:
            events = []
            diagnostic = {
                "corp_code": corp_code,
                "event_count": 0,
                "rate_limited": False,
                "error": safe_error_message(exc),
            }
            status = "error"

        disclosures[code] = {
            "stock_code": code,
            "stock_name": stock["name"],
            "corp_code": corp_code,
            "events": events,
        }
        features[code] = build_feature(stock, corp_code, events, end_dt)
        stock_diagnostics[code] = {
            "stock_name": stock["name"],
            "corp_code_found": True,
            "event_count": len(events),
            "status": status,
            **diagnostic,
        }

    generated_at = datetime.now(timezone.utc).isoformat()
    disclosures_payload = {
        "generated_at": generated_at,
        "source": "opendart",
        "bgn_de": bgn_de,
        "end_de": end_de,
        "disclosures": disclosures,
    }
    features_payload = {
        "generated_at": generated_at,
        "source": "opendart",
        "features": features,
    }
    diagnostic_payload = {
        "generated_at": generated_at,
        "source": "opendart",
        "bgn_de": bgn_de,
        "end_de": end_de,
        "days": days,
        "requested_stock_count": len(stocks),
        "feature_stock_count": len(features),
        "corp_code_refreshed": refreshed,
        "stock_diagnostics": stock_diagnostics,
    }

    save_json(DART_DISCLOSURES_PATH, disclosures_payload)
    save_json(DART_EVENT_FEATURES_PATH, features_payload)
    save_json(DART_SYNC_DIAGNOSTIC_PATH, diagnostic_payload)
    DART_DISCLOSURES_REPORT_PATH.write_text(build_markdown(features, diagnostic_payload), encoding="utf-8")

    return {
        "disclosures": disclosures_payload,
        "features": features_payload,
        "diagnostic": diagnostic_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect compact OpenDART disclosure event features.")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--max-stocks", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--force-corp-code-refresh", action="store_true")
    args = parser.parse_args()

    try:
        result = collect_dart(
            days=args.days,
            max_stocks=args.max_stocks,
            sleep_seconds=args.sleep,
            force_corp_code_refresh=args.force_corp_code_refresh,
        )
    except Exception as exc:
        print(f"DART collection failed: {safe_error_message(exc)}")
        raise SystemExit(1)

    features = result["features"]["features"]
    print(f"Saved corp codes: {DART_CORP_CODES_PATH}")
    print(f"Saved disclosures: {DART_DISCLOSURES_PATH}")
    print(f"Saved features: {DART_EVENT_FEATURES_PATH}")
    print(f"Saved diagnostic: {DART_SYNC_DIAGNOSTIC_PATH}")
    print(f"Saved report: {DART_DISCLOSURES_REPORT_PATH}")
    print(f"Feature rows: {len(features)}")


if __name__ == "__main__":
    main()
