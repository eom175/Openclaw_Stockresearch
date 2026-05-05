import argparse
import json
import re
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from typing import Optional

import feedparser

from policylink.paths import CANDIDATES_PATH, DATA_DIR

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. RSS SOURCES
# ============================================================
# 국내투자 중심:
# - 한국 정책/산업/규제/수출입/공시성 보도자료를 우선
# - Fed는 국내 시장에도 금리/환율/외국인 수급 영향을 주므로 유지
# ============================================================

FEEDS = [
    {
        "name": "Federal Reserve Press Releases",
        "url": "https://www.federalreserve.gov/feeds/press_all.xml",
        "country": "US",
        "source_type": "official_global_macro",
        "base_score": 3,
    },
    {
        "name": "Federal Reserve Speeches",
        "url": "https://www.federalreserve.gov/feeds/speeches.xml",
        "country": "US",
        "source_type": "official_global_macro",
        "base_score": 2,
    },
    {
        "name": "Bank of Korea Press Releases",
        "url": "https://www.bok.or.kr/portal/bbs/P0000559/news.rss",
        "country": "KR",
        "source_type": "official_macro",
        "base_score": 5,
    },
    {
        "name": "Financial Services Commission Press Releases",
        "url": "http://www.fsc.go.kr/about/fsc_bbs_rss/?fid=0111",
        "country": "KR",
        "source_type": "official_finance",
        "base_score": 5,
    },
    {
        "name": "Korea Policy Briefing - Press Releases",
        "url": "https://www.korea.kr/rss/pressrelease.xml",
        "country": "KR",
        "source_type": "official_policy_portal",
        "base_score": 2,
    },
    {
        "name": "Korea Policy Briefing - Policy News",
        "url": "https://www.korea.kr/rss/policy.xml",
        "country": "KR",
        "source_type": "official_policy_news",
        "base_score": 1,
    },
    {
        "name": "MOEF - Economy and Finance",
        "url": "https://www.korea.kr/rss/dept_moef.xml",
        "country": "KR",
        "source_type": "official_macro_policy",
        "base_score": 5,
    },
    {
        "name": "MOTIR - Industry and Trade",
        "url": "https://www.korea.kr/rss/dept_motir.xml",
        "country": "KR",
        "source_type": "official_industry_policy",
        "base_score": 5,
    },
    {
        "name": "MSIT - Science and ICT",
        "url": "https://www.korea.kr/rss/dept_msit.xml",
        "country": "KR",
        "source_type": "official_tech_policy",
        "base_score": 4,
    },
    {
        "name": "MOLIT - Land and Transport",
        "url": "https://www.korea.kr/rss/dept_molit.xml",
        "country": "KR",
        "source_type": "official_real_estate_transport",
        "base_score": 4,
    },
    {
        "name": "MSS - SMEs and Startups",
        "url": "https://www.korea.kr/rss/dept_mss.xml",
        "country": "KR",
        "source_type": "official_sme_policy",
        "base_score": 3,
    },
    {
        "name": "MCEE - Climate Energy Environment",
        "url": "https://www.korea.kr/rss/dept_mcee.xml",
        "country": "KR",
        "source_type": "official_energy_environment",
        "base_score": 4,
    },
    {
        "name": "MOF - Oceans and Fisheries",
        "url": "https://www.korea.kr/rss/dept_mof.xml",
        "country": "KR",
        "source_type": "official_shipbuilding_logistics",
        "base_score": 3,
    },
    {
        "name": "Korea Customs Service",
        "url": "https://www.korea.kr/rss/dept_customs.xml",
        "country": "KR",
        "source_type": "official_trade_data",
        "base_score": 4,
    },
    {
        "name": "Defense Acquisition Program Administration",
        "url": "https://www.korea.kr/rss/dept_dapa.xml",
        "country": "KR",
        "source_type": "official_defense",
        "base_score": 4,
    },
    {
        "name": "Korea AeroSpace Administration",
        "url": "https://www.korea.kr/rss/dept_kasa.xml",
        "country": "KR",
        "source_type": "official_aerospace",
        "base_score": 3,
    },
    {
        "name": "Fair Trade Commission",
        "url": "https://www.korea.kr/rss/dept_ftc.xml",
        "country": "KR",
        "source_type": "official_regulation",
        "base_score": 4,
    },
    {
        "name": "Financial Services Commission - Policy Briefing",
        "url": "https://www.korea.kr/rss/dept_fsc.xml",
        "country": "KR",
        "source_type": "official_finance_policy",
        "base_score": 4,
    },
]


# ============================================================
# 2. KEYWORDS
# ============================================================

KEYWORDS = {
    "monetary_policy": [
        "fed", "fomc", "federal reserve", "interest rate", "rate cut",
        "rate hike", "inflation", "cpi", "pce", "employment", "jobs",
        "금리", "기준금리", "금통위", "통화정책", "인플레이션", "물가",
        "소비자물가", "생산자물가", "고용", "환율", "원달러", "원/달러",
        "외환", "국고채", "채권", "한국은행",
    ],
    "korea_macro_policy": [
        "재정", "세제", "세금", "감세", "증세", "추경", "예산",
        "국채", "경제성장률", "경기", "소비", "투자", "수출", "수입",
        "무역수지", "경상수지", "기획재정부", "재정경제부",
    ],
    "trade_policy": [
        "tariff", "export control", "sanction", "trade", "customs",
        "관세", "수출통제", "제재", "무역", "통상", "수출입",
        "FTA", "공급망", "리쇼어링", "관세청",
    ],
    "korea_market_policy": [
        "밸류업", "기업가치", "자본시장", "거래소", "상장", "상장폐지",
        "공매도", "증권시장", "증시", "주식시장", "ETF", "ETN",
        "증권거래세", "배당", "자사주", "주주환원", "스튜어드십",
        "금융투자", "금융위원회", "금융감독원",
    ],
    "financial_regulation": [
        "bank", "liquidity", "capital", "financial regulation",
        "은행", "보험", "증권", "카드", "금융지주", "유동성",
        "자본규제", "충당금", "연체율", "부실채권", "PF",
        "대손충당금", "금융안정", "가계대출", "부동산PF",
    ],
    "semiconductor_battery": [
        "반도체", "HBM", "D램", "DRAM", "낸드", "NAND", "파운드리",
        "소부장", "AI 반도체", "첨단산업", "국가전략기술",
        "배터리", "이차전지", "2차전지", "양극재", "음극재",
        "분리막", "전해질", "전고체", "ESS",
    ],
    "auto_ev": [
        "자동차", "전기차", "EV", "하이브리드", "수소차", "자율주행",
        "완성차", "자동차부품", "모빌리티", "충전", "배터리 리스",
    ],
    "shipbuilding_defense": [
        "조선", "LNG선", "해양플랜트", "선박", "해운", "항만",
        "방산", "무기", "수출계약", "국방", "항공우주", "우주항공",
        "위성", "발사체", "K-방산",
    ],
    "energy_policy": [
        "oil", "gas", "energy", "opec", "crude",
        "원유", "유가", "에너지", "전력", "전기요금", "가스요금",
        "LNG", "원전", "재생에너지", "태양광", "풍력", "수소",
        "탄소", "배출권", "기후", "환경",
    ],
    "platform_regulation": [
        "플랫폼", "온라인", "공정거래", "과징금", "독과점",
        "카카오", "네이버", "쿠팡", "배달", "이커머스", "온라인 플랫폼",
        "개인정보", "광고", "수수료",
    ],
    "real_estate_construction": [
        "부동산", "주택", "건설", "SOC", "철도", "공항", "도로",
        "교통", "전세", "분양", "재건축", "재개발", "시멘트",
        "건설사", "프로젝트파이낸싱",
    ],
    "bio_healthcare": [
        "바이오", "제약", "의약품", "신약", "임상", "의료기기",
        "헬스케어", "건강보험", "약가", "식약처", "백신",
    ],
    "consumer_tourism_content": [
        "소비", "내수", "관광", "면세", "화장품", "유통", "편의점",
        "콘텐츠", "게임", "K-콘텐츠", "영화", "음악", "공연",
    ],
    "geopolitical_risk": [
        "war", "conflict", "iran", "china", "taiwan", "russia",
        "전쟁", "분쟁", "중동", "이란", "중국", "대만", "러시아",
        "북한", "안보", "제재", "지정학",
    ],
}


# ============================================================
# 3. ASSET HINTS
# ============================================================

ASSET_HINTS = {
    "monetary_policy": [
        "KOSPI", "KOSDAQ", "KRW", "USD/KRW", "국고채", "은행주",
        "성장주", "배당주",
    ],
    "korea_macro_policy": [
        "KOSPI", "KOSDAQ", "KRW", "국고채", "내수주", "은행주",
    ],
    "trade_policy": [
        "KOSPI", "KOSDAQ", "KRW", "수출주", "반도체", "자동차",
        "조선", "화학",
    ],
    "korea_market_policy": [
        "KOSPI", "KOSDAQ", "증권주", "고배당주", "밸류업", "ETF",
    ],
    "financial_regulation": [
        "은행주", "보험주", "증권주", "금융지주", "건설주", "리츠",
    ],
    "semiconductor_battery": [
        "삼성전자", "SK하이닉스", "반도체ETF", "소부장", "2차전지ETF",
        "배터리소재",
    ],
    "auto_ev": [
        "현대차", "기아", "자동차부품", "2차전지", "전기차충전",
    ],
    "shipbuilding_defense": [
        "조선주", "방산주", "항공우주", "해운주",
    ],
    "energy_policy": [
        "정유주", "가스주", "전력주", "원전주", "태양광", "풍력",
        "배출권", "화학주",
    ],
    "platform_regulation": [
        "NAVER", "카카오", "플랫폼주", "유통주", "게임주",
    ],
    "real_estate_construction": [
        "건설주", "은행주", "리츠", "시멘트", "철강",
    ],
    "bio_healthcare": [
        "제약바이오", "의료기기", "헬스케어ETF",
    ],
    "consumer_tourism_content": [
        "화장품", "면세점", "유통주", "여행주", "엔터주", "게임주",
    ],
    "geopolitical_risk": [
        "KRW", "USD/KRW", "방산주", "정유주", "금", "VIX", "KOSPI",
    ],
}


IMPORTANT_TERMS = [
    "기준금리", "금통위", "통화정책", "물가", "인플레이션", "환율",
    "원달러", "원/달러", "국고채", "fomc", "fed", "rate", "inflation",
    "밸류업", "공매도", "상장폐지", "자사주", "배당", "자본시장",
    "반도체", "HBM", "배터리", "이차전지", "조선", "방산", "원전",
    "전기차", "수출", "관세", "공급망",
    "전쟁", "분쟁", "제재", "북한", "중동", "대만",
]


# ============================================================
# 4. UTILS
# ============================================================

def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_title_key(title: str) -> str:
    return re.sub(r"[^가-힣a-zA-Z0-9]+", "", (title or "").lower())[:90]


def parse_date(entry) -> datetime:
    for key in ["published", "updated", "created"]:
        value = entry.get(key)
        if value:
            try:
                dt = parsedate_to_datetime(value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass

    for key in ["published_parsed", "updated_parsed", "created_parsed"]:
        value = entry.get(key)
        if value:
            try:
                return datetime(*value[:6], tzinfo=timezone.utc)
            except Exception:
                pass

    return datetime.now(timezone.utc)


def recency_score(published_at: datetime) -> int:
    age_hours = (datetime.now(timezone.utc) - published_at).total_seconds() / 3600

    if age_hours <= 24:
        return 6
    if age_hours <= 72:
        return 4
    if age_hours <= 168:
        return 2
    if age_hours <= 720:
        return 1
    return 0


def classify_and_score(
    title: str,
    snippet: str,
    feed: Optional[dict] = None,
    published_at: Optional[datetime] = None
):
    text = f"{title} {snippet}".lower()
    matched_types = []
    matched_keywords = []
    score = 0

    feed = feed or {}
    base_score = int(feed.get("base_score", 0))
    score += base_score

    for policy_type, words in KEYWORDS.items():
        local_hits = []

        for word in words:
            if word.lower() in text:
                local_hits.append(word)

        if local_hits:
            matched_types.append(policy_type)
            matched_keywords.extend(local_hits)
            score += len(local_hits) * 2

    for term in IMPORTANT_TERMS:
        if term.lower() in text:
            score += 3

    source_type = feed.get("source_type", "")
    country = feed.get("country", "")

    if country == "KR" and source_type.startswith("official"):
        score += 2

    if source_type in [
        "official_macro",
        "official_macro_policy",
        "official_finance",
        "official_finance_policy",
        "official_industry_policy",
    ]:
        score += 2

    if any(x in text for x in ["fed", "fomc", "federal reserve", "한국은행", "금통위", "통화정책"]):
        score += 4

    if published_at:
        score += recency_score(published_at)

    if not matched_types:
        matched_types = ["official_update" if country == "KR" else "general_macro"]

    asset_hints = []
    for policy_type in matched_types:
        asset_hints.extend(ASSET_HINTS.get(policy_type, []))

    return {
        "score": score,
        "policy_types": sorted(set(matched_types)),
        "matched_keywords": sorted(set(matched_keywords)),
        "asset_hints": sorted(set(asset_hints)),
    }


def should_keep_item(features: dict, feed: dict) -> bool:
    score = features.get("score", 0)
    country = feed.get("country", "")
    source_type = feed.get("source_type", "")

    if score >= 5:
        return True

    if country == "KR" and source_type.startswith("official"):
        return True

    return False


def infer_priority_bucket(item: dict) -> str:
    policy_types = set(item.get("policy_types", []))

    if {"monetary_policy", "korea_macro_policy"} & policy_types:
        return "macro_policy"
    if {"korea_market_policy", "financial_regulation"} & policy_types:
        return "market_finance"
    if {
        "semiconductor_battery",
        "auto_ev",
        "shipbuilding_defense",
        "energy_policy",
    } & policy_types:
        return "industry_sector"
    if {"geopolitical_risk"} & policy_types:
        return "risk"
    return "official_update"


# ============================================================
# 5. COLLECT
# ============================================================

def collect(hours: int, max_items: int):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    seen_urls = set()
    seen_titles = set()
    items = []
    feed_errors = []

    for feed in FEEDS:
        try:
            parsed = feedparser.parse(feed["url"])
        except Exception as e:
            feed_errors.append({
                "source": feed["name"],
                "url": feed["url"],
                "error": str(e),
            })
            continue

        if getattr(parsed, "bozo", False):
            feed_errors.append({
                "source": feed["name"],
                "url": feed["url"],
                "error": str(getattr(parsed, "bozo_exception", "feed parsing warning")),
            })

        for entry in parsed.entries:
            title = clean_text(entry.get("title", ""))
            url = entry.get("link", "")
            snippet = clean_text(entry.get("summary", entry.get("description", "")))
            published_at = parse_date(entry)

            if not title or not url:
                continue

            if published_at < cutoff:
                continue

            if url in seen_urls:
                continue

            title_key = normalize_title_key(title)
            if title_key in seen_titles:
                continue

            features = classify_and_score(
                title=title,
                snippet=snippet,
                feed=feed,
                published_at=published_at,
            )

            if not should_keep_item(features, feed):
                continue

            seen_urls.add(url)
            seen_titles.add(title_key)

            item = {
                "title": title[:180],
                "source": feed["name"],
                "source_type": feed["source_type"],
                "country": feed["country"],
                "url": url,
                "published_at": published_at.isoformat(),
                "snippet": snippet[:450],
                "score": features["score"],
                "recency_score": recency_score(published_at),
                "priority_bucket": "",
                "policy_types": features["policy_types"],
                "matched_keywords": features["matched_keywords"][:12],
                "asset_hints": features["asset_hints"],
            }
            item["priority_bucket"] = infer_priority_bucket(item)

            items.append(item)

    items.sort(
        key=lambda x: (
            int(x.get("score", 0)),
            x.get("published_at", ""),
        ),
        reverse=True,
    )

    return items[:max_items], feed_errors


# ============================================================
# 6. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hours",
        type=int,
        default=168,
        help="How many recent hours to collect. Default: 168 hours = 7 days.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=50,
        help="Maximum number of compact candidates to save.",
    )
    args = parser.parse_args()

    items, feed_errors = collect(args.hours, args.max_items)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hours": args.hours,
        "count": len(items),
        "description": (
            "Domestic-investing-oriented compact research candidates. "
            "Prioritize KR official policy, finance, industry, trade, "
            "macro, sector, and regulation sources. "
            "OpenClaw should read this compact JSON only, not raw web pages by default."
        ),
        "items": items,
        "feed_errors": feed_errors[:20],
    }

    out_path = CANDIDATES_PATH
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(items)} items to {out_path}")

    if feed_errors:
        print(f"Feed warnings/errors: {len(feed_errors)}")
        for err in feed_errors[:5]:
            print(f"- {err['source']}: {err['error']}")


if __name__ == "__main__":
    main()
