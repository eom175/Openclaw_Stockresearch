import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from policylink.paths import (
    CANDIDATES_PATH,
    DAILY_BRIEF_PATH,
    DAILY_FEATURES_PATH,
    DATA_DIR,
    FLOW_FEATURES_PATH,
    KIWOOM_ACCOUNT_SUMMARY_PATH,
    NEWS_EVENT_FEATURES_PATH,
    PORTFOLIO_RECOMMENDATION_JSON_PATH,
    PORTFOLIO_RECOMMENDATION_MD_PATH,
    PRICE_FEATURES_PATH,
    REPORTS_DIR,
)
from policylink.universe import KNOWN_STOCK_SECTOR
from policylink.utils import load_json, load_text, normalize_code, parse_number

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ACCOUNT_SUMMARY_PATH = KIWOOM_ACCOUNT_SUMMARY_PATH

OUTPUT_JSON_PATH = PORTFOLIO_RECOMMENDATION_JSON_PATH
OUTPUT_MD_PATH = PORTFOLIO_RECOMMENDATION_MD_PATH


DOMESTIC_SECTOR_MAP = {
    "core_market": {
        "label": "국내 대형주 / KOSPI200",
        "description": "국내 전체 시장에 분산 노출하는 기본 코어 자산군",
        "candidate_assets": [
            {"name": "KOSPI200 ETF", "code": "", "note": "국내 상장 KOSPI200 추종 ETF"},
            {"name": "대형 우량주 바스켓", "code": "", "note": "삼성전자, 현대차, 금융지주 등 대형주 중심"},
        ],
    },
    "semiconductor_battery": {
        "label": "반도체 / 2차전지",
        "description": "AI, HBM, 수출, 배터리 정책에 민감한 성장 섹터",
        "candidate_assets": [
            {"name": "삼성전자", "code": "005930", "note": "반도체 대표 대형주"},
            {"name": "SK하이닉스", "code": "000660", "note": "HBM/메모리 민감주"},
            {"name": "국내 반도체 ETF", "code": "", "note": "반도체 섹터 분산 노출"},
            {"name": "국내 2차전지 ETF", "code": "", "note": "배터리/소재 섹터 분산 노출"},
        ],
    },
    "financial_value": {
        "label": "금융 / 밸류업 / 배당",
        "description": "금리, 금융규제, 주주환원 정책에 민감한 가치 섹터",
        "candidate_assets": [
            {"name": "KB금융", "code": "105560", "note": "금융지주 대표주"},
            {"name": "신한지주", "code": "055550", "note": "금융지주 대표주"},
            {"name": "증권/은행 ETF", "code": "", "note": "금융 섹터 분산 노출"},
            {"name": "고배당 ETF", "code": "", "note": "밸류업/주주환원 정책 수혜 후보"},
        ],
    },
    "auto_ev": {
        "label": "자동차 / 전기차",
        "description": "수출, 환율, 관세, 전기차 정책에 민감한 섹터",
        "candidate_assets": [
            {"name": "현대차", "code": "005380", "note": "완성차 대표주"},
            {"name": "기아", "code": "000270", "note": "완성차 대표주"},
            {"name": "자동차부품주 바스켓", "code": "", "note": "부품/모빌리티 관련 분산 후보"},
        ],
    },
    "defense_shipbuilding": {
        "label": "방산 / 조선 / 항공우주",
        "description": "수출계약, 지정학 리스크, 국방정책에 민감한 섹터",
        "candidate_assets": [
            {"name": "한화에어로스페이스", "code": "012450", "note": "방산/항공우주 대표주"},
            {"name": "조선주 바스켓", "code": "", "note": "LNG선/해양플랜트/수주 사이클 후보"},
            {"name": "방산 ETF", "code": "", "note": "방산 섹터 분산 후보"},
        ],
    },
    "energy_infra": {
        "label": "에너지 / 원전 / 인프라",
        "description": "유가, 전력정책, 원전, 인프라 정책에 민감한 섹터",
        "candidate_assets": [
            {"name": "두산에너빌리티", "code": "034020", "note": "원전/에너지 설비 대표주"},
            {"name": "정유/가스주 바스켓", "code": "", "note": "유가/가스요금 민감 후보"},
            {"name": "전력/인프라 관련주", "code": "", "note": "전력망/인프라 정책 후보"},
        ],
    },
    "platform_internet": {
        "label": "플랫폼 / 인터넷 / 콘텐츠",
        "description": "공정거래, 플랫폼 규제, AI/콘텐츠 정책에 민감한 섹터",
        "candidate_assets": [
            {"name": "NAVER", "code": "035420", "note": "플랫폼/AI/광고 민감주"},
            {"name": "카카오", "code": "035720", "note": "플랫폼/콘텐츠/규제 민감주"},
            {"name": "게임/콘텐츠주 바스켓", "code": "", "note": "K-콘텐츠 관련 후보"},
        ],
    },
    "bond_cash_like": {
        "label": "국내 채권 / 현금성",
        "description": "금리 변동성과 주식 리스크를 낮추기 위한 방어 자산군",
        "candidate_assets": [
            {"name": "국고채 ETF", "code": "", "note": "금리 하락 기대 시 후보"},
            {"name": "단기채 ETF", "code": "", "note": "대기자금/현금성 대체 후보"},
            {"name": "현금", "code": "CASH", "note": "불확실성 확대 시 유지"},
        ],
    },
}

POLICY_PRIORITY = [
    "monetary_policy",
    "korea_macro_policy",
    "korea_market_policy",
    "financial_regulation",
    "trade_policy",
    "semiconductor_battery",
    "auto_ev",
    "shipbuilding_defense",
    "energy_policy",
    "platform_regulation",
    "real_estate_construction",
    "bio_healthcare",
    "consumer_tourism_content",
    "geopolitical_risk",
    "official_update",
    "general_macro",
]

POLICY_TO_SECTOR = {
    "monetary_policy": ["bond_cash_like", "core_market", "financial_value"],
    "korea_macro_policy": ["core_market", "bond_cash_like", "financial_value"],
    "korea_market_policy": ["core_market", "financial_value"],
    "financial_regulation": ["financial_value"],
    "trade_policy": ["semiconductor_battery", "auto_ev", "defense_shipbuilding"],
    "semiconductor_battery": ["semiconductor_battery"],
    "auto_ev": ["auto_ev"],
    "shipbuilding_defense": ["defense_shipbuilding"],
    "energy_policy": ["energy_infra", "bond_cash_like"],
    "platform_regulation": ["platform_internet"],
    "real_estate_construction": ["financial_value", "energy_infra"],
    "bio_healthcare": ["core_market"],
    "consumer_tourism_content": ["platform_internet", "core_market"],
    "geopolitical_risk": ["defense_shipbuilding", "energy_infra", "bond_cash_like"],
    "official_update": ["core_market"],
    "general_macro": ["core_market"],
}

RISK_POLICY_TYPES = {
    "monetary_policy",
    "geopolitical_risk",
    "energy_policy",
    "financial_regulation",
    "real_estate_construction",
}

POSITIVE_OPPORTUNITY_POLICY_TYPES = {
    "korea_market_policy",
    "semiconductor_battery",
    "auto_ev",
    "shipbuilding_defense",
    "trade_policy",
}

def fmt_krw(value) -> str:
    return f"{int(parse_number(value)):,}원"


def fmt_pct(value) -> str:
    return f"{float(value) * 100:.1f}%"


def fmt_ratio(value) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def safe_list(value):
    return value if isinstance(value, list) else []


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def log_cap_score(raw_score: float) -> float:
    raw_score = max(0.0, raw_score)
    return clamp(math.log1p(raw_score) * 1.4, 0.0, 8.0)


def recency_multiplier(recency_score: float) -> float:
    if recency_score >= 6:
        return 1.25
    if recency_score >= 4:
        return 1.10
    if recency_score >= 2:
        return 0.90
    if recency_score >= 1:
        return 0.65
    return 0.40


def source_multiplier(source_type: str) -> float:
    source_type = str(source_type or "")
    if source_type in {
        "official_macro",
        "official_macro_policy",
        "official_finance",
        "official_finance_policy",
        "official_industry_policy",
    }:
        return 1.15
    if source_type.startswith("official"):
        return 1.0
    return 0.75


def select_primary_policy_types(policy_types: List[str], max_count: int = 3) -> List[str]:
    unique = []
    for policy_type in policy_types:
        if policy_type not in unique:
            unique.append(policy_type)

    def priority(policy_type: str) -> int:
        try:
            return POLICY_PRIORITY.index(policy_type)
        except ValueError:
            return len(POLICY_PRIORITY)

    unique.sort(key=priority)
    return unique[:max_count]


def normalize_scores_0_to_100(raw_scores: Dict[str, float]) -> Dict[str, float]:
    if not raw_scores:
        return {}

    max_score = max(raw_scores.values()) if raw_scores else 0.0
    if max_score <= 0:
        return {key: 0.0 for key in raw_scores}

    return {
        key: round(clamp((value / max_score) * 100.0, 0.0, 100.0), 2)
        for key, value in raw_scores.items()
    }


def load_price_features() -> Dict[str, Any]:
    raw = load_json(PRICE_FEATURES_PATH, {"features": {}})
    features = raw.get("features", {})
    if not isinstance(features, dict):
        features = {}

    normalized = {}
    for code, item in features.items():
        normalized[normalize_code(code)] = item

    return {
        "generated_at": raw.get("generated_at"),
        "base_date": raw.get("base_date"),
        "features": normalized,
    }


def load_flow_features() -> Dict[str, Any]:
    raw = load_json(FLOW_FEATURES_PATH, {"features": {}, "sector_flow_scores": {}})
    features = raw.get("features", {})
    sector_flow_scores = raw.get("sector_flow_scores", {})

    if not isinstance(features, dict):
        features = {}

    if not isinstance(sector_flow_scores, dict):
        sector_flow_scores = {}

    normalized = {}
    for code, item in features.items():
        normalized[normalize_code(code)] = item

    return {
        "generated_at": raw.get("generated_at"),
        "features": normalized,
        "sector_flow_scores": sector_flow_scores,
    }


def load_news_features() -> Dict[str, Any]:
    raw = load_json(NEWS_EVENT_FEATURES_PATH, {"features": {}, "sector_scores": {}})
    features = raw.get("features", {})
    sector_scores = raw.get("sector_scores", {})

    if not isinstance(features, dict):
        features = {}
    if not isinstance(sector_scores, dict):
        sector_scores = {}

    normalized = {}
    for code, item in features.items():
        normalized[normalize_code(code)] = item

    return {
        "generated_at": raw.get("generated_at"),
        "features": normalized,
        "sector_scores": sector_scores,
    }


def price_feature_score(feature: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not feature:
        return {
            "score": 50.0,
            "label": "no_price_data",
            "action_hint": "가격 데이터 없음 / 중립 처리",
            "reasons": ["price_features.json에 해당 종목 데이터가 없어 중립값 50으로 처리했습니다."],
            "risk_penalty": 0.0,
            "opportunity_bonus": 0.0,
            "feature": None,
        }

    return_5d = parse_number(feature.get("return_5d"), 0.0)
    return_20d = parse_number(feature.get("return_20d"), 0.0)
    volatility_20d = parse_number(feature.get("volatility_20d"), 0.0)
    ma20_gap = parse_number(feature.get("ma20_gap"), 0.0)
    ma60_gap = parse_number(feature.get("ma60_gap"), 0.0)
    drawdown_20d = parse_number(feature.get("drawdown_20d"), 0.0)
    volume_ratio_20 = parse_number(feature.get("volume_ratio_20"), 1.0)
    trend_label = str(feature.get("trend_label", "neutral"))
    risk_label = str(feature.get("risk_label", "normal"))

    score = 50.0
    reasons = []

    if trend_label == "uptrend":
        score += 15
        reasons.append("상승 추세")
    elif trend_label == "downtrend":
        score -= 18
        reasons.append("하락 추세")
    else:
        reasons.append("중립 추세")

    if return_5d > 0.03:
        score += 8
        reasons.append("최근 5일 수익률 양호")
    elif return_5d < -0.03:
        score -= 8
        reasons.append("최근 5일 수익률 약세")

    if return_20d > 0.15:
        score -= 12
        reasons.append("20일 급등 후 추격매수 주의")
    elif return_20d > 0.05:
        score += 5
        reasons.append("20일 중기 추세 양호")
    elif return_20d < -0.10:
        score -= 8
        reasons.append("20일 하락폭 큼")

    if ma20_gap > 0.10:
        score -= 8
        reasons.append("20일선 대비 과열 가능성")
    elif ma20_gap > 0.02:
        score += 6
        reasons.append("20일선 위 안정적 위치")
    elif ma20_gap < -0.05:
        score -= 10
        reasons.append("20일선 아래 약세")

    if ma60_gap > 0.15:
        score -= 5
        reasons.append("60일선 대비 단기 부담")
    elif ma60_gap > 0.03:
        score += 4
        reasons.append("60일선 위 중기 추세 양호")
    elif ma60_gap < -0.08:
        score -= 8
        reasons.append("60일선 아래 중기 약세")

    if drawdown_20d < -0.12:
        score -= 8
        reasons.append("20일 고점 대비 낙폭 큼")
    elif -0.08 <= drawdown_20d <= -0.03:
        score += 4
        reasons.append("단기 눌림 구간 가능성")

    if volatility_20d >= 0.04:
        score -= 12
        reasons.append("20일 변동성 높음")
    elif 0.0 < volatility_20d <= 0.018:
        score += 4
        reasons.append("변동성 안정")

    if volume_ratio_20 >= 1.8:
        score += 4
        reasons.append("거래량 증가")
    elif volume_ratio_20 <= 0.5:
        score -= 3
        reasons.append("거래량 감소")

    if risk_label == "high_volatility":
        score -= 8
    elif risk_label == "low_volatility":
        score += 3

    score = clamp(score, 0.0, 100.0)

    if score >= 70:
        label = "favorable"
        action_hint = "가격 조건 양호 / 소액 분할진입 후보"
    elif score >= 55:
        label = "neutral_positive"
        action_hint = "관심 유지 / 눌림 시 분할 접근"
    elif score >= 40:
        label = "neutral_caution"
        action_hint = "관망 / 가격 안정 확인 필요"
    else:
        label = "unfavorable"
        action_hint = "추격매수 금지 / 비중 확대 보류"

    risk_penalty = 0.0
    opportunity_bonus = 0.0

    if volatility_20d >= 0.04:
        risk_penalty += 10.0
    if return_20d > 0.15:
        risk_penalty += 8.0
    if trend_label == "downtrend":
        risk_penalty += 8.0
    if score >= 65:
        opportunity_bonus += 8.0
    if trend_label == "uptrend" and volatility_20d < 0.04:
        opportunity_bonus += 6.0

    return {
        "score": round(score, 2),
        "label": label,
        "action_hint": action_hint,
        "reasons": reasons[:8],
        "risk_penalty": round(risk_penalty, 2),
        "opportunity_bonus": round(opportunity_bonus, 2),
        "feature": feature,
    }


def flow_feature_score(feature: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not feature:
        return {
            "score": 50.0,
            "label": "no_flow_data",
            "action_hint": "수급 데이터 없음 / 중립 처리",
            "reasons": ["flow_features.json에 해당 종목 데이터가 없어 중립값 50으로 처리했습니다."],
            "risk_penalty": 0.0,
            "opportunity_bonus": 0.0,
            "feature": None,
        }

    score = parse_number(feature.get("flow_score"), 50.0)
    label = str(feature.get("flow_label", "neutral"))
    action_hint = str(feature.get("action_hint", "수급 중립"))
    reasons = safe_list(feature.get("reasons"))

    foreign_5d = parse_number(feature.get("foreign_net_5d"), 0.0)
    institution_5d = parse_number(feature.get("institution_net_5d"), 0.0)
    combined_5d = parse_number(feature.get("combined_net_5d"), 0.0)

    risk_penalty = 0.0
    opportunity_bonus = 0.0

    if foreign_5d < 0 and institution_5d < 0:
        risk_penalty += 10.0
    elif foreign_5d > 0 and institution_5d > 0:
        opportunity_bonus += 10.0

    if combined_5d < 0:
        risk_penalty += 4.0
    elif combined_5d > 0:
        opportunity_bonus += 4.0

    if score >= 70:
        opportunity_bonus += 5.0
    elif score < 40:
        risk_penalty += 5.0

    return {
        "score": round(clamp(score, 0.0, 100.0), 2),
        "label": label,
        "action_hint": action_hint,
        "reasons": reasons[:8],
        "risk_penalty": round(risk_penalty, 2),
        "opportunity_bonus": round(opportunity_bonus, 2),
        "feature": feature,
    }


def news_feature_score(feature: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not feature:
        return {
            "score": 50.0,
            "label": "no_news_data",
            "action_hint": "뉴스 데이터 없음 / 중립 처리",
            "reasons": ["news_event_features.json에 해당 종목 데이터가 없어 중립값 50으로 처리했습니다."],
            "risk_penalty": 0.0,
            "opportunity_bonus": 0.0,
            "feature": None,
        }

    score = parse_number(feature.get("sentiment_score"), 50.0)
    label = str(feature.get("news_label", "neutral"))
    attention = parse_number(feature.get("attention_score"), 0.0)
    positive_count = parse_number(feature.get("positive_keyword_count_7d"), 0)
    negative_count = parse_number(feature.get("negative_keyword_count_7d"), 0)
    risk_count = parse_number(feature.get("risk_keyword_count_7d"), 0)

    reasons = [
        f"뉴스 라벨 {label}",
        f"관심도 {attention}",
        f"긍정키워드 {int(positive_count)}",
        f"부정키워드 {int(negative_count)}",
        f"리스크키워드 {int(risk_count)}",
    ]

    risk_penalty = 0.0
    opportunity_bonus = 0.0
    if score < 40:
        risk_penalty += 12.0
    if negative_count >= 4:
        risk_penalty += 8.0
    if risk_count >= 5:
        risk_penalty += 5.0
    if score >= 65:
        opportunity_bonus += 8.0
    if positive_count >= 4 and negative_count <= 2:
        opportunity_bonus += 5.0

    if score >= 65:
        action_hint = "뉴스 흐름 우호 / 가격·수급 확인 후 분할 접근 가능"
    elif score >= 45:
        action_hint = "뉴스 흐름 중립 / 기존 조건 우선"
    else:
        action_hint = "뉴스 흐름 약세 / 추가매수 금지"

    return {
        "score": round(clamp(score, 0.0, 100.0), 2),
        "label": label,
        "action_hint": action_hint,
        "reasons": reasons,
        "risk_penalty": round(risk_penalty, 2),
        "opportunity_bonus": round(opportunity_bonus, 2),
        "feature": feature,
    }


def build_sector_price_scores(price_features: Dict[str, Any]) -> Dict[str, Any]:
    features = price_features.get("features", {})
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for code, feature in features.items():
        sector = feature.get("sector") or KNOWN_STOCK_SECTOR.get(normalize_code(code), "unknown")
        grouped.setdefault(sector, [])

        scored = price_feature_score(feature)
        scored["stock_code"] = normalize_code(code)
        scored["stock_name"] = feature.get("stock_name", code)
        grouped[sector].append(scored)

    result = {}

    for sector, items in grouped.items():
        if not items:
            continue

        result[sector] = {
            "price_score": round(sum(item["score"] for item in items) / len(items), 2),
            "risk_penalty": round(sum(item["risk_penalty"] for item in items) / len(items), 2),
            "opportunity_bonus": round(sum(item["opportunity_bonus"] for item in items) / len(items), 2),
            "items": items,
        }

    return result


def build_sector_flow_scores(flow_features: Dict[str, Any]) -> Dict[str, Any]:
    features = flow_features.get("features", {})
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for code, feature in features.items():
        sector = feature.get("sector") or KNOWN_STOCK_SECTOR.get(normalize_code(code), "unknown")
        grouped.setdefault(sector, [])

        scored = flow_feature_score(feature)
        scored["stock_code"] = normalize_code(code)
        scored["stock_name"] = feature.get("stock_name", code)
        grouped[sector].append(scored)

    result = {}

    for sector, items in grouped.items():
        if not items:
            continue

        result[sector] = {
            "flow_score": round(sum(item["score"] for item in items) / len(items), 2),
            "risk_penalty": round(sum(item["risk_penalty"] for item in items) / len(items), 2),
            "opportunity_bonus": round(sum(item["opportunity_bonus"] for item in items) / len(items), 2),
            "items": items,
        }

    return result


def build_sector_news_scores(news_features: Dict[str, Any]) -> Dict[str, Any]:
    features = news_features.get("features", {})
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for code, feature in features.items():
        sector = feature.get("sector") or KNOWN_STOCK_SECTOR.get(normalize_code(code), "unknown")
        grouped.setdefault(sector, [])

        scored = news_feature_score(feature)
        scored["stock_code"] = normalize_code(code)
        scored["stock_name"] = feature.get("stock_name", code)
        grouped[sector].append(scored)

    result = {}

    for sector, items in grouped.items():
        if not items:
            continue

        result[sector] = {
            "news_score": round(sum(item["score"] for item in items) / len(items), 2),
            "attention_score": round(
                sum(parse_number((item.get("feature") or {}).get("attention_score"), 0.0) for item in items) / len(items),
                2,
            ),
            "risk_penalty": round(sum(item["risk_penalty"] for item in items) / len(items), 2),
            "opportunity_bonus": round(sum(item["opportunity_bonus"] for item in items) / len(items), 2),
            "items": items,
        }

    return result


def analyze_research(candidates: Dict[str, Any], daily_features: Dict[str, Any]) -> Dict[str, Any]:
    items = safe_list(candidates.get("items"))

    raw_sector_scores = {key: 0.0 for key in DOMESTIC_SECTOR_MAP.keys()}
    risk_points = []
    opportunity_points = []
    macro_points = []
    top_events = []

    for item in items:
        title = str(item.get("title", ""))
        source = str(item.get("source", ""))
        source_type = str(item.get("source_type", ""))
        score = parse_number(item.get("score"), 0.0)
        recency = parse_number(item.get("recency_score"), 0.0)

        all_policy_types = safe_list(item.get("policy_types"))
        primary_policy_types = select_primary_policy_types(all_policy_types, max_count=3)
        matched_keywords = safe_list(item.get("matched_keywords"))
        asset_hints = safe_list(item.get("asset_hints"))

        event_strength = (
            log_cap_score(score)
            * recency_multiplier(recency)
            * source_multiplier(source_type)
        )
        event_strength = clamp(event_strength, 0.0, 8.0)

        event_risk = 0.0
        event_opportunity = 0.0
        event_macro = 0.0

        for policy_type in primary_policy_types:
            for sector in POLICY_TO_SECTOR.get(policy_type, []):
                raw_sector_scores[sector] = raw_sector_scores.get(sector, 0.0) + event_strength

            if policy_type in RISK_POLICY_TYPES:
                event_risk += min(2.5, event_strength * 0.45)

            if policy_type in POSITIVE_OPPORTUNITY_POLICY_TYPES:
                event_opportunity += min(2.5, event_strength * 0.45)

            if policy_type in {"monetary_policy", "korea_macro_policy"}:
                event_macro += min(2.0, event_strength * 0.35)

        joined_text = " ".join(
            [title] + [str(x) for x in matched_keywords] + [str(x) for x in asset_hints]
        ).lower()

        if any(term in joined_text for term in ["금리", "물가", "환율", "fomc", "fed"]):
            event_risk += 0.8
            event_macro += 0.8

        if any(term in joined_text for term in ["전쟁", "분쟁", "제재", "중동", "유가"]):
            event_risk += 1.0

        risk_points.append(clamp(event_risk, 0.0, 4.0))
        opportunity_points.append(clamp(event_opportunity, 0.0, 4.0))
        macro_points.append(clamp(event_macro, 0.0, 3.0))

        top_events.append({
            "title": title,
            "source": source,
            "source_type": source_type,
            "raw_score": round(score, 2),
            "event_strength": round(event_strength, 2),
            "recency_score": recency,
            "policy_types_all": all_policy_types,
            "policy_types_used": primary_policy_types,
            "matched_keywords": matched_keywords[:8],
            "asset_hints": asset_hints[:8],
        })

    for event in safe_list(daily_features.get("events")):
        policy_type = event.get("policy_type")
        importance = clamp(parse_number(event.get("importance_score_0_to_1"), 0.0), 0.0, 1.0)

        if policy_type in POLICY_TO_SECTOR:
            for sector in POLICY_TO_SECTOR.get(policy_type, []):
                raw_sector_scores[sector] += importance * 2.0

        if policy_type in RISK_POLICY_TYPES:
            risk_points.append(importance * 2.0)

        if policy_type in POSITIVE_OPPORTUNITY_POLICY_TYPES:
            opportunity_points.append(importance * 2.0)

    if risk_points:
        top_risk = sorted(risk_points, reverse=True)[:10]
        risk_score = clamp((sum(top_risk) / len(top_risk)) * 25.0, 0.0, 100.0)
    else:
        risk_score = 0.0

    if opportunity_points:
        top_opp = sorted(opportunity_points, reverse=True)[:10]
        opportunity_score = clamp((sum(top_opp) / len(top_opp)) * 25.0, 0.0, 100.0)
    else:
        opportunity_score = 0.0

    if macro_points:
        top_macro = sorted(macro_points, reverse=True)[:10]
        macro_pressure_score = clamp((sum(top_macro) / len(top_macro)) * 33.3, 0.0, 100.0)
    else:
        macro_pressure_score = 0.0

    if risk_score >= 70:
        risk_level = "high"
    elif risk_score >= 40:
        risk_level = "medium"
    else:
        risk_level = "low"

    normalized_sector_scores = normalize_scores_0_to_100(raw_sector_scores)

    ranked_sectors = sorted(
        [
            {
                "sector": sector,
                "label": DOMESTIC_SECTOR_MAP[sector]["label"],
                "score": score,
                "raw_score": round(raw_sector_scores.get(sector, 0.0), 2),
                "candidate_assets": DOMESTIC_SECTOR_MAP[sector]["candidate_assets"],
                "description": DOMESTIC_SECTOR_MAP[sector]["description"],
            }
            for sector, score in normalized_sector_scores.items()
            if score > 0
        ],
        key=lambda x: x["score"],
        reverse=True,
    )

    top_events.sort(key=lambda x: (x.get("event_strength", 0.0), x.get("raw_score", 0.0)), reverse=True)

    return {
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "opportunity_score": round(opportunity_score, 2),
        "macro_pressure_score": round(macro_pressure_score, 2),
        "sector_scores": normalized_sector_scores,
        "raw_sector_scores": {key: round(value, 2) for key, value in raw_sector_scores.items()},
        "ranked_sectors": ranked_sectors,
        "top_events": top_events[:7],
        "scoring_note": "리서치 점수는 이벤트별 cap과 0~100 정규화를 적용한 값입니다.",
    }


def normalize_holding(item: Dict[str, Any]) -> Dict[str, Any]:
    code = normalize_code(item.get("stock_code") or item.get("stk_cd") or item.get("code") or "-")
    name = str(item.get("stock_name") or item.get("stk_nm") or item.get("name") or "이름 확인 필요")
    quantity = parse_number(item.get("quantity") or item.get("rmnd_qty") or item.get("qty"), 0)
    evaluation_amount = parse_number(item.get("evaluation_amount") or item.get("eval_amount") or item.get("evlt_amt"), 0)
    purchase_amount = parse_number(item.get("purchase_amount") or item.get("pur_amt"), 0)
    pnl = parse_number(item.get("pnl") or item.get("evltv_prft"), 0)
    return_rate = parse_number(item.get("return_rate") or item.get("prft_rt"), 0)
    sector = KNOWN_STOCK_SECTOR.get(code, "unknown")

    return {
        "stock_code": code,
        "stock_name": name,
        "quantity": int(quantity),
        "evaluation_amount": int(evaluation_amount),
        "purchase_amount": int(purchase_amount),
        "pnl": int(pnl),
        "return_rate": return_rate,
        "sector": sector,
    }


def analyze_account(account_summary: Dict[str, Any]) -> Dict[str, Any]:
    cash = parse_number(account_summary.get("available_cash"), 0)
    if cash <= 0:
        cash = parse_number(account_summary.get("orderable_100_amount"), 0)
    if cash <= 0:
        cash = parse_number(account_summary.get("orderable_amount"), 0)

    total_equity = parse_number(account_summary.get("estimated_total_equity"), 0)

    raw_holdings = safe_list(account_summary.get("holdings") or account_summary.get("top_holdings"))
    holdings = [normalize_holding(item) for item in raw_holdings]
    invested_value = sum(item["evaluation_amount"] for item in holdings)

    if total_equity <= 0:
        total_equity = cash + invested_value
    if total_equity <= 0:
        total_equity = 1

    pending_or_reserved_amount = parse_number(
        account_summary.get("pending_or_reserved_amount") or account_summary.get("pending_or_reserved"),
        0,
    )

    sector_exposure = {}
    for item in holdings:
        sector = item["sector"]
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + item["evaluation_amount"] / total_equity

    return {
        "cash": int(cash),
        "total_equity": int(total_equity),
        "invested_value": int(invested_value),
        "cash_weight": round(cash / total_equity, 4),
        "invested_weight": round(invested_value / total_equity, 4),
        "holding_count": len(holdings),
        "holdings": holdings,
        "sector_exposure": {key: round(value, 4) for key, value in sector_exposure.items()},
        "pending_or_reserved_amount": int(pending_or_reserved_amount),
    }


def combine_sector_scores(
    research: Dict[str, Any],
    sector_price_scores: Dict[str, Any],
    sector_flow_scores: Dict[str, Any],
    sector_news_scores: Optional[Dict[str, Any]] = None,
    has_news_features: bool = False,
) -> List[Dict[str, Any]]:
    combined = []
    sector_news_scores = sector_news_scores or {}

    if has_news_features:
        weights = {"research": 0.40, "price": 0.28, "flow": 0.22, "news": 0.10}
    else:
        weights = {"research": 0.45, "price": 0.30, "flow": 0.25, "news": 0.0}

    for sector_item in research["ranked_sectors"]:
        sector = sector_item["sector"]

        research_score = parse_number(sector_item["score"], 0.0)

        price_info = sector_price_scores.get(sector, {})
        price_score = parse_number(price_info.get("price_score"), 50.0)
        price_risk_penalty = parse_number(price_info.get("risk_penalty"), 0.0)
        price_opportunity_bonus = parse_number(price_info.get("opportunity_bonus"), 0.0)

        flow_info = sector_flow_scores.get(sector, {})
        flow_score = parse_number(flow_info.get("flow_score"), 50.0)
        flow_risk_penalty = parse_number(flow_info.get("risk_penalty"), 0.0)
        flow_opportunity_bonus = parse_number(flow_info.get("opportunity_bonus"), 0.0)

        news_info = sector_news_scores.get(sector, {})
        news_score = parse_number(news_info.get("news_score"), 50.0)
        news_risk_penalty = parse_number(news_info.get("risk_penalty"), 0.0)
        news_opportunity_bonus = parse_number(news_info.get("opportunity_bonus"), 0.0)

        final_score = (
            research_score * weights["research"]
            + price_score * weights["price"]
            + flow_score * weights["flow"]
            + news_score * weights["news"]
            + price_opportunity_bonus * 0.12
            + flow_opportunity_bonus * 0.15
            + news_opportunity_bonus * 0.10
            - price_risk_penalty * 0.15
            - flow_risk_penalty * 0.18
            - news_risk_penalty * 0.10
        )

        final_score = clamp(final_score, 0.0, 100.0)

        combined.append({
            "sector": sector,
            "label": sector_item["label"],
            "description": sector_item["description"],
            "candidate_assets": sector_item["candidate_assets"],
            "research_score": round(research_score, 2),
            "price_score": round(price_score, 2),
            "flow_score": round(flow_score, 2),
            "news_score": round(news_score, 2) if has_news_features else None,
            "news_label": (
                news_info.get("items", [{}])[0].get("label")
                if news_info.get("items")
                else None
            ),
            "news_attention_score": round(parse_number(news_info.get("attention_score"), 0.0), 2) if has_news_features else None,
            "price_risk_penalty": round(price_risk_penalty, 2),
            "flow_risk_penalty": round(flow_risk_penalty, 2),
            "news_risk_penalty": round(news_risk_penalty, 2) if has_news_features else 0.0,
            "price_opportunity_bonus": round(price_opportunity_bonus, 2),
            "flow_opportunity_bonus": round(flow_opportunity_bonus, 2),
            "news_opportunity_bonus": round(news_opportunity_bonus, 2) if has_news_features else 0.0,
            "final_score": round(final_score, 2),
            "scoring_weights": weights,
            "price_items": price_info.get("items", []),
            "flow_items": flow_info.get("items", []),
            "news_items": news_info.get("items", []),
        })

    combined.sort(key=lambda x: x["final_score"], reverse=True)
    return combined


def build_target_allocation(risk_level: str, combined_sectors: List[Dict[str, Any]]) -> Dict[str, float]:
    if risk_level == "high":
        base = {"cash": 0.70, "core_market": 0.12, "bond_cash_like": 0.10, "sector_rotation": 0.08}
    elif risk_level == "medium":
        base = {"cash": 0.55, "core_market": 0.22, "bond_cash_like": 0.08, "sector_rotation": 0.15}
    else:
        base = {"cash": 0.40, "core_market": 0.30, "bond_cash_like": 0.05, "sector_rotation": 0.25}

    top_sector_candidates = [
        item for item in combined_sectors
        if item["sector"] not in {"core_market", "bond_cash_like"}
    ][:3]

    allocation = {
        "cash": base["cash"],
        "core_market": base["core_market"],
        "bond_cash_like": base["bond_cash_like"],
    }

    if top_sector_candidates:
        total_score = sum(max(1.0, item["final_score"]) for item in top_sector_candidates)

        for item in top_sector_candidates:
            allocation[item["sector"]] = round(
                base["sector_rotation"] * max(1.0, item["final_score"]) / total_score,
                4,
            )
    else:
        allocation["core_market"] += base["sector_rotation"]

    total = sum(allocation.values())
    if total > 0:
        allocation = {key: round(value / total, 4) for key, value in allocation.items()}

    return allocation


def make_holding_recommendations(
    account: Dict[str, Any],
    research: Dict[str, Any],
    price_features: Dict[str, Any],
    flow_features: Dict[str, Any],
    news_features: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    recommendations = []

    sector_scores = research["sector_scores"]
    price_by_code = price_features.get("features", {})
    flow_by_code = flow_features.get("features", {})
    news_by_code = (news_features or {}).get("features", {})
    has_news_features = bool(news_by_code)
    risk_level = research["risk_level"]

    for item in account["holdings"]:
        code = item["stock_code"]
        sector = item["sector"]

        sector_score = sector_scores.get(sector, 0.0)

        price_result = price_feature_score(price_by_code.get(code))
        flow_result = flow_feature_score(flow_by_code.get(code))
        news_result = news_feature_score(news_by_code.get(code)) if has_news_features else news_feature_score(None)

        if has_news_features:
            combined_quality = (
                sector_score * 0.40
                + price_result["score"] * 0.28
                + flow_result["score"] * 0.22
                + news_result["score"] * 0.10
            )
        else:
            combined_quality = (
                sector_score * 0.45
                + price_result["score"] * 0.30
                + flow_result["score"] * 0.25
            )

        negative_news_count = parse_number((news_result.get("feature") or {}).get("negative_keyword_count_7d"), 0)

        action = "hold"
        reason = "보유 유지가 기본입니다. 단, 가격·수급 조건을 함께 보며 추가매수 여부를 판단합니다."

        if item["quantity"] <= 1 and item["evaluation_amount"] < account["total_equity"] * 0.01:
            action = "test_position"
            reason = "현재 비중이 매우 작아 매매 기능 검증용 포지션으로 분류합니다."

        if price_result["score"] < 40 or flow_result["score"] < 40:
            action = "hold_no_add"
            reason = "가격 또는 수급 피처가 약해 추가매수는 보류하는 편이 적절합니다."

        if risk_level == "high" and combined_quality < 55:
            action = "consider_reduce_or_wait"
            reason = "리스크 레벨이 높고 종합 점수가 낮아 신규 확대보다 관망이 적절합니다."

        if combined_quality >= 70:
            action = "hold_or_gradual_add"
            reason = "리서치·가격·수급 조건이 모두 비교적 양호합니다. 단, 모의투자 단계에서는 소액 분할 접근이 적절합니다."

        if has_news_features and news_result["score"] < 40:
            action = "hold_no_add"
            reason = "뉴스 심리가 약해 추가매수는 금지하고 가격·수급 안정 확인이 필요합니다."

        if has_news_features and negative_news_count >= 4:
            action = "consider_reduce_or_wait"
            reason = "최근 뉴스의 부정 키워드가 높아 비중 확대보다 관망 또는 축소 검토가 적절합니다."

        if (
            has_news_features
            and news_result["score"] >= 65
            and price_result["score"] >= 55
            and flow_result["score"] >= 55
            and combined_quality >= 65
        ):
            action = "hold_or_gradual_add"
            reason = "뉴스·가격·수급 조건이 함께 양호해 보유 유지 또는 소액 분할 확대 후보입니다."

        recommendations.append({
            "stock_code": code,
            "stock_name": item["stock_name"],
            "sector": sector,
            "sector_score": sector_score,
            "price_score": price_result["score"],
            "flow_score": flow_result["score"],
            "news_score": news_result["score"] if has_news_features else None,
            "news_label": news_result["label"] if has_news_features else None,
            "news_attention_score": parse_number((news_result.get("feature") or {}).get("attention_score"), 0) if has_news_features else None,
            "combined_quality": round(combined_quality, 2),
            "quantity": item["quantity"],
            "evaluation_amount": item["evaluation_amount"],
            "current_weight": round(item["evaluation_amount"] / account["total_equity"], 4),
            "pnl": item["pnl"],
            "return_rate": item["return_rate"],
            "action": action,
            "reason": reason,
            "price_hint": price_result["action_hint"],
            "flow_hint": flow_result["action_hint"],
            "news_hint": news_result["action_hint"] if has_news_features else "뉴스 데이터 없음",
            "price_feature": price_result.get("feature"),
            "flow_feature": flow_result.get("feature"),
            "news_feature": news_result.get("feature") if has_news_features else None,
            "price_reasons": price_result["reasons"],
            "flow_reasons": flow_result["reasons"],
            "news_reasons": news_result["reasons"] if has_news_features else [],
        })

    return recommendations


def make_new_position_plan(
    account: Dict[str, Any],
    research: Dict[str, Any],
    combined_sectors: List[Dict[str, Any]],
    target_allocation: Dict[str, float],
) -> Dict[str, Any]:
    risk_level = research["risk_level"]

    if risk_level == "high":
        deploy_ratio = 0.01
        stance = "관망 우선 / 테스트 금액만 허용"
    elif risk_level == "medium":
        deploy_ratio = 0.03
        stance = "소액 분할진입 가능"
    else:
        deploy_ratio = 0.05
        stance = "분할진입 가능"

    if account["pending_or_reserved_amount"] > 0:
        deploy_ratio = min(deploy_ratio, 0.005)
        stance += " / 단, 미체결·주문대기 금액 확인 필요"

    raw_limit = account["cash"] * deploy_ratio
    today_new_investment_limit = int(min(raw_limit, 3_000_000))

    if account["cash_weight"] < target_allocation.get("cash", 0.5):
        today_new_investment_limit = 0
        stance = "현금 비중이 목표보다 낮아 신규 진입보다 보유 점검 우선"

    watchlist = []

    for item in combined_sectors[:4]:
        sector = item["sector"]
        target_weight = target_allocation.get(sector, 0.0)

        if item["final_score"] >= 75:
            timing_hint = "리서치·가격·수급 모두 양호 / 소액 분할진입 후보"
        elif item["final_score"] >= 60:
            timing_hint = "관심 유지 / 가격 또는 수급 추가 확인 후 접근"
        elif item["final_score"] >= 45:
            timing_hint = "중립 / 추격 진입보다 관망 우선"
        else:
            timing_hint = "후순위 / 신규 진입 보류"

        if item.get("news_score") is not None:
            news_score = parse_number(item.get("news_score"), 50.0)
            if news_score < 45:
                timing_hint += " / 뉴스 흐름 약세로 보수적 접근"
            elif news_score >= 65:
                timing_hint += " / 뉴스 흐름 우호"

        watchlist.append({
            "sector": sector,
            "label": item["label"],
            "research_score": item["research_score"],
            "price_score": item["price_score"],
            "flow_score": item["flow_score"],
            "news_score": item.get("news_score"),
            "news_label": item.get("news_label"),
            "news_attention_score": item.get("news_attention_score"),
            "final_score": item["final_score"],
            "target_weight": target_weight,
            "candidate_assets": item["candidate_assets"],
            "reason": item["description"],
            "timing_hint": timing_hint,
            "price_items": item.get("price_items", []),
            "flow_items": item.get("flow_items", []),
            "news_items": item.get("news_items", []),
        })

    return {
        "stance": stance,
        "today_new_investment_limit": today_new_investment_limit,
        "deploy_ratio": deploy_ratio,
        "watchlist": watchlist,
        "rule": "실제 주문은 생성하지 않습니다. 추천 리포트만 생성합니다.",
    }


def build_recommendation():
    candidates = load_json(CANDIDATES_PATH, {"items": []})
    daily_features = load_json(DAILY_FEATURES_PATH, {"events": []})
    account_summary = load_json(ACCOUNT_SUMMARY_PATH, {})
    daily_brief = load_text(DAILY_BRIEF_PATH, "")
    price_features = load_price_features()
    flow_features = load_flow_features()
    news_features = load_news_features()

    research = analyze_research(candidates, daily_features)
    account = analyze_account(account_summary)

    sector_price_scores = build_sector_price_scores(price_features)
    sector_flow_scores = build_sector_flow_scores(flow_features)
    sector_news_scores = build_sector_news_scores(news_features)
    has_news_features = bool(news_features.get("features"))

    combined_sectors = combine_sector_scores(
        research=research,
        sector_price_scores=sector_price_scores,
        sector_flow_scores=sector_flow_scores,
        sector_news_scores=sector_news_scores,
        has_news_features=has_news_features,
    )

    target_allocation = build_target_allocation(
        risk_level=research["risk_level"],
        combined_sectors=combined_sectors,
    )

    holding_recommendations = make_holding_recommendations(
        account=account,
        research=research,
        price_features=price_features,
        flow_features=flow_features,
        news_features=news_features,
    )

    new_position_plan = make_new_position_plan(
        account=account,
        research=research,
        combined_sectors=combined_sectors,
        target_allocation=target_allocation,
    )

    warnings = [
        "이 결과는 모의투자/개발용 추천 리포트이며 투자 자문이 아닙니다.",
        "실제 주문은 생성하지 않습니다.",
        "이번 버전은 리서치·가격·수급을 함께 반영하지만, 아직 재무/실적/밸류에이션/백테스트 검증은 포함하지 않았습니다.",
    ]

    if account["pending_or_reserved_amount"] > 0:
        warnings.append("주문대기/미체결로 추정되는 금액이 있으므로 신규 주문 전 주문체결현황을 확인해야 합니다.")

    if not daily_brief:
        warnings.append("daily_brief.md가 없어 리서치 요약 신뢰도가 낮습니다.")

    if not price_features.get("features"):
        warnings.append("price_features.json이 비어 있어 가격 기반 추천 신뢰도가 낮습니다.")

    if not flow_features.get("features"):
        warnings.append("flow_features.json이 비어 있어 수급 기반 추천 신뢰도가 낮습니다.")

    if not news_features.get("features"):
        warnings.append("news_event_features.json이 없어 뉴스 기반 추천 보정은 적용하지 않았습니다.")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "mock_recommendation_only",
        "order_enabled": False,
        "account": account,
        "research": research,
        "price_features_meta": {
            "generated_at": price_features.get("generated_at"),
            "base_date": price_features.get("base_date"),
            "count": len(price_features.get("features", {})),
        },
        "flow_features_meta": {
            "generated_at": flow_features.get("generated_at"),
            "count": len(flow_features.get("features", {})),
        },
        "news_features_meta": {
            "generated_at": news_features.get("generated_at"),
            "count": len(news_features.get("features", {})),
            "enabled": has_news_features,
        },
        "sector_price_scores": sector_price_scores,
        "sector_flow_scores": sector_flow_scores,
        "sector_news_scores": sector_news_scores,
        "combined_sectors": combined_sectors,
        "target_allocation": target_allocation,
        "holding_recommendations": holding_recommendations,
        "new_position_plan": new_position_plan,
        "warnings": warnings,
        "next_step": "build_dataset.py로 리서치·가격·수급·뉴스 피처를 날짜/종목 단위 학습 데이터셋으로 결합합니다.",
    }


def allocation_label(key: str) -> str:
    if key == "cash":
        return "현금"
    return DOMESTIC_SECTOR_MAP.get(key, {}).get("label", key)


def build_price_feature_line(feature: Optional[Dict[str, Any]]) -> str:
    if not feature:
        return "가격 데이터 없음"

    return (
        f"종가 {feature.get('latest_close'):,}원"
        f" / 5일 {fmt_ratio(feature.get('return_5d'))}"
        f" / 20일 {fmt_ratio(feature.get('return_20d'))}"
        f" / 변동성20D {fmt_ratio(feature.get('volatility_20d'))}"
        f" / MA20 괴리 {fmt_ratio(feature.get('ma20_gap'))}"
        f" / 추세 {feature.get('trend_label')}"
        f" / 리스크 {feature.get('risk_label')}"
    )


def build_flow_feature_line(feature: Optional[Dict[str, Any]]) -> str:
    if not feature:
        return "수급 데이터 없음"

    return (
        f"외국인5D {int(parse_number(feature.get('foreign_net_5d'))):,}주"
        f" / 기관5D {int(parse_number(feature.get('institution_net_5d'))):,}주"
        f" / 합산5D {int(parse_number(feature.get('combined_net_5d'))):,}주"
        f" / flow_score {feature.get('flow_score')}"
        f" / label {feature.get('flow_label')}"
    )


def build_news_feature_line(feature: Optional[Dict[str, Any]]) -> str:
    if not feature:
        return "뉴스 데이터 없음"

    latest_title = feature.get("latest_news_title") or "최신 뉴스 없음"
    return (
        f"뉴스7D {feature.get('news_count_7d')}"
        f" / 긍정키워드 {feature.get('positive_keyword_count_7d')}"
        f" / 부정키워드 {feature.get('negative_keyword_count_7d')}"
        f" / 리스크키워드 {feature.get('risk_keyword_count_7d')}"
        f" / 뉴스점수 {feature.get('sentiment_score')}"
        f" / 관심도 {feature.get('attention_score')}"
        f" / label {feature.get('news_label')}"
        f" / 최신 뉴스 제목: {latest_title}"
    )


def build_markdown(result: Dict[str, Any]) -> str:
    account = result["account"]
    research = result["research"]
    new_plan = result["new_position_plan"]

    lines = []

    lines.append("# 국내투자 포트폴리오 추천 리포트")
    lines.append("")
    lines.append(f"- 생성 시각 UTC: {result['generated_at']}")
    lines.append("- 모드: 모의투자 / 추천 전용")
    lines.append("- 실제 주문 실행: 비활성화")
    lines.append(f"- 가격 데이터 기준일: {result['price_features_meta'].get('base_date')}")
    lines.append(f"- 수급 데이터 생성 시각: {result['flow_features_meta'].get('generated_at')}")
    if result.get("news_features_meta", {}).get("enabled"):
        lines.append(f"- 뉴스 데이터 생성 시각: {result['news_features_meta'].get('generated_at')}")
    else:
        lines.append("- 뉴스 데이터: 없음 / 기존 리서치·가격·수급 비중 유지")
        lines.append("- 뉴스 점수: N/A / 뉴스 라벨: N/A / 최신 뉴스 제목: N/A")
    lines.append("")

    lines.append("## 1. 오늘의 판단 요약")
    lines.append(f"- 리스크 레벨: {research['risk_level']} / risk_score={research['risk_score']}")
    lines.append(f"- 기회 점수: {research['opportunity_score']}")
    lines.append(f"- 매크로 압력 점수: {research['macro_pressure_score']}")
    lines.append(f"- 추천 스탠스: {new_plan['stance']}")
    lines.append(f"- 오늘 신규 진입 한도 제안: {fmt_krw(new_plan['today_new_investment_limit'])}")
    lines.append(f"- 현재 총자산 후보: {fmt_krw(account['total_equity'])}")
    lines.append(f"- 사용 가능 현금: {fmt_krw(account['cash'])}")
    lines.append(f"- 현금 비중: {fmt_pct(account['cash_weight'])}")
    lines.append(f"- 투자 비중: {fmt_pct(account['invested_weight'])}")
    lines.append("")

    lines.append("## 2. 목표 비중 제안")
    for key, value in result["target_allocation"].items():
        lines.append(f"- {allocation_label(key)}: {value * 100:.1f}%")
    lines.append("")

    lines.append("## 3. 관심 섹터 / 후보 자산")
    if new_plan["watchlist"]:
        for i, item in enumerate(new_plan["watchlist"], 1):
            lines.append(f"### {i}. {item['label']}")
            lines.append(f"- 최종 점수: {item['final_score']}")
            lines.append(f"- 리서치 점수: {item['research_score']}")
            lines.append(f"- 가격 점수: {item['price_score']}")
            lines.append(f"- 수급 점수: {item['flow_score']}")
            if item.get("news_score") is not None:
                lines.append(f"- 뉴스 점수: {item.get('news_score')}")
                lines.append(f"- 뉴스 라벨: {item.get('news_label')}")
                lines.append(f"- 뉴스 관심도: {item.get('news_attention_score')}")
            lines.append(f"- 목표 비중 후보: {item['target_weight'] * 100:.1f}%")
            lines.append(f"- 타이밍 힌트: {item['timing_hint']}")
            lines.append(f"- 이유: {item['reason']}")
            lines.append("- 후보:")
            for asset in item["candidate_assets"]:
                code = asset.get("code")
                code_text = f"({code})" if code else ""
                lines.append(f"  - {asset['name']}{code_text}: {asset['note']}")

            if item.get("price_items"):
                lines.append("- 가격 피처 참고:")
                for price_item in item["price_items"][:3]:
                    lines.append(
                        f"  - {price_item.get('stock_name')}({price_item.get('stock_code')}): "
                        f"가격점수 {price_item.get('score')} / {price_item.get('action_hint')}"
                    )
                    lines.append(f"    - {build_price_feature_line(price_item.get('feature'))}")

            if item.get("flow_items"):
                lines.append("- 수급 피처 참고:")
                for flow_item in item["flow_items"][:3]:
                    lines.append(
                        f"  - {flow_item.get('stock_name')}({flow_item.get('stock_code')}): "
                        f"수급점수 {flow_item.get('score')} / {flow_item.get('action_hint')}"
                    )
                    lines.append(f"    - {build_flow_feature_line(flow_item.get('feature'))}")

            if item.get("news_items"):
                lines.append("- 뉴스 피처 참고:")
                for news_item in item["news_items"][:3]:
                    feature = news_item.get("feature") or {}
                    lines.append(
                        f"  - {news_item.get('stock_name')}({news_item.get('stock_code')}): "
                        f"뉴스점수 {news_item.get('score')} / {news_item.get('label')}"
                    )
                    lines.append(f"    - {build_news_feature_line(feature)}")
            lines.append("")
    else:
        lines.append("- 오늘 리서치에서 뚜렷한 신규 섹터 후보가 잡히지 않았습니다.")
        lines.append("")

    lines.append("## 4. 현재 보유 종목별 제안")
    if result["holding_recommendations"]:
        for item in result["holding_recommendations"]:
            lines.append(
                f"- {item['stock_name']}({item['stock_code']}) "
                f"/ 비중 {item['current_weight'] * 100:.2f}% "
                f"/ 수량 {item['quantity']}주 "
                f"/ 섹터점수 {item['sector_score']} "
                f"/ 가격점수 {item['price_score']} "
                f"/ 수급점수 {item['flow_score']} "
                f"/ 뉴스점수 {item.get('news_score') if item.get('news_score') is not None else 'N/A'} "
                f"/ 종합품질 {item['combined_quality']} "
                f"/ 액션: {item['action']}"
            )
            lines.append(f"  - 이유: {item['reason']}")
            lines.append(f"  - 가격 힌트: {item['price_hint']}")
            lines.append(f"  - 수급 힌트: {item['flow_hint']}")
            lines.append(f"  - 뉴스 힌트: {item.get('news_hint', '뉴스 데이터 없음')}")
            if item.get("price_feature"):
                lines.append(f"  - 가격 피처: {build_price_feature_line(item.get('price_feature'))}")
            if item.get("flow_feature"):
                lines.append(f"  - 수급 피처: {build_flow_feature_line(item.get('flow_feature'))}")
            if item.get("news_feature"):
                lines.append(f"  - 뉴스 피처: {build_news_feature_line(item.get('news_feature'))}")
            if item.get("price_reasons"):
                lines.append(f"  - 가격 근거: {', '.join(item.get('price_reasons', [])[:6])}")
            if item.get("flow_reasons"):
                lines.append(f"  - 수급 근거: {', '.join(item.get('flow_reasons', [])[:6])}")
            if item.get("news_reasons"):
                lines.append(f"  - 뉴스 근거: {', '.join(item.get('news_reasons', [])[:6])}")
    else:
        lines.append("- 현재 보유 종목이 없거나 파싱된 보유 종목이 없습니다.")
    lines.append("")

    lines.append("## 5. 오늘 리서치 상위 이벤트")
    if research["top_events"]:
        for i, event in enumerate(research["top_events"][:5], 1):
            types_used = ", ".join(event.get("policy_types_used", []))
            hints = ", ".join([str(x) for x in event.get("asset_hints", [])[:5]])
            lines.append(
                f"{i}. {event['title']} "
                f"/ source={event['source']} "
                f"/ raw_score={event['raw_score']} "
                f"/ event_strength={event['event_strength']} "
                f"/ types_used={types_used} "
                f"/ hints={hints}"
            )
    else:
        lines.append("- 리서치 이벤트가 없습니다.")
    lines.append("")

    lines.append("## 6. 점수 산정 메모")
    if result.get("news_features_meta", {}).get("enabled"):
        lines.append("- 최종 섹터 점수는 리서치 40%, 가격 28%, 수급 22%, 뉴스 10% 비중으로 계산합니다.")
    else:
        lines.append("- 최종 섹터 점수는 리서치 45%, 가격 30%, 수급 25% 비중으로 계산합니다.")
    lines.append("- 가격점수는 5일/20일 수익률, 20일 변동성, 이동평균 괴리율, 낙폭, 거래량 비율을 반영합니다.")
    lines.append("- 수급점수는 외국인/기관 5일·20일 순매수와 동반 순매수/순매도 여부를 반영합니다.")
    lines.append("- 뉴스점수는 네이버 뉴스 검색 API의 제목/요약 키워드 기반 sentiment와 attention을 반영합니다.")
    lines.append("- 아직 재무, 실적, 밸류에이션, 체결강도, 백테스트 검증은 포함하지 않았습니다.")
    lines.append("")

    lines.append("## 7. 주의사항")
    for warning in result["warnings"]:
        lines.append(f"- {warning}")

    lines.append("")
    lines.append("## 8. 다음 단계")
    lines.append("- 이 리포트는 주문을 만들지 않습니다.")
    lines.append("- 다음 개발 단계는 `build_dataset.py`로 리서치·가격·수급·뉴스 피처를 날짜/종목 단위 학습 데이터셋으로 결합하는 것입니다.")
    lines.append("- 이후 `train_model.py`에서 XGBoost 또는 룰 기반+모델 혼합 방식으로 신호를 검증합니다.")

    return "\n".join(lines)


def main():
    result = build_recommendation()

    OUTPUT_JSON_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown = build_markdown(result)

    OUTPUT_MD_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print(f"Saved recommendation json: {OUTPUT_JSON_PATH}")
    print(f"Saved recommendation report: {OUTPUT_MD_PATH}")
    print("")
    print(markdown)


if __name__ == "__main__":
    main()
