from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from policylink.paths import (
    CANDIDATES_PATH,
    DART_EVENT_FEATURES_PATH,
    DATA_DIR,
    FLOW_FEATURES_PATH,
    KIWOOM_ACCOUNT_SUMMARY_PATH,
    MODEL_DATASET_CSV_PATH,
    MODEL_DATASET_JSONL_PATH,
    MODEL_DATASET_SNAPSHOT_PATH,
    NEWS_EVENT_FEATURES_PATH,
    PORTFOLIO_RECOMMENDATION_JSON_PATH,
    PRICE_FEATURES_PATH,
    REPORTS_DIR,
    YAHOO_GLOBAL_FEATURES_PATH,
)
from policylink.universe import KNOWN_STOCK_SECTOR, universe_for_dataset
from policylink.utils import load_json, load_jsonl, normalize_code, parse_number, save_csv as write_csv, save_jsonl


DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


ACCOUNT_SUMMARY_PATH = KIWOOM_ACCOUNT_SUMMARY_PATH
PORTFOLIO_RECOMMENDATION_PATH = PORTFOLIO_RECOMMENDATION_JSON_PATH

DATASET_JSONL_PATH = MODEL_DATASET_JSONL_PATH
DATASET_CSV_PATH = MODEL_DATASET_CSV_PATH
SNAPSHOT_REPORT_PATH = MODEL_DATASET_SNAPSHOT_PATH


DEFAULT_UNIVERSE = universe_for_dataset()


CSV_COLUMNS = [
    "snapshot_date",
    "generated_at",
    "stock_code",
    "stock_name",
    "sector",

    "is_holding",
    "holding_quantity",
    "holding_eval_amount",
    "holding_weight",
    "holding_pnl",
    "holding_return_rate",

    "account_total_equity",
    "account_cash",
    "account_cash_weight",
    "account_invested_weight",
    "pending_or_reserved_amount",

    "research_score",
    "price_score",
    "flow_score",
    "final_score",
    "target_weight",
    "recommendation_rank",

    "latest_date",
    "latest_close",
    "latest_volume",
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "ma20_gap",
    "ma60_gap",
    "drawdown_20d",
    "volume_ratio_20",
    "trend_label",
    "price_risk_label",

    "foreign_net_1d",
    "foreign_net_5d",
    "foreign_net_20d",
    "institution_net_1d",
    "institution_net_5d",
    "institution_net_20d",
    "combined_net_1d",
    "combined_net_5d",
    "combined_net_20d",
    "combined_net_5d_to_avg_volume_20",
    "combined_net_20d_to_avg_volume_20",
    "foreign_weight",
    "foreign_limit_exhaustion_rate",
    "flow_label",

    "dart_event_count_7d",
    "dart_event_count_30d",
    "dart_event_count_90d",
    "dart_positive_event_count_30d",
    "dart_negative_event_count_30d",
    "dart_major_event_count_30d",
    "dart_score",
    "dart_label",
    "dart_contract_supply",
    "dart_new_investment",
    "dart_treasury_stock",
    "dart_dividend",
    "dart_bonus_issue",
    "dart_paid_in_capital_increase",
    "dart_convertible_bond",
    "dart_ownership_change",
    "dart_trading_halt",
    "dart_delisting_risk",
    "dart_lawsuit",
    "dart_embezzlement_breach",
    "dart_audit_opinion_risk",
    "dart_correction_delay",

    "naver_news_count",
    "naver_news_count_1d",
    "naver_news_count_7d",
    "naver_positive_keyword_count_7d",
    "naver_negative_keyword_count_7d",
    "naver_risk_keyword_count_7d",
    "naver_attention_score",
    "naver_sentiment_score",
    "naver_news_label",

    "yahoo_sector_global_signal_score",
    "yahoo_sector_global_risk_score",
    "yahoo_sector_proxy_count",
    "yahoo_related_proxy_labels",
    "yahoo_semiconductor_proxy_score",
    "yahoo_rate_proxy_score",
    "yahoo_dollar_proxy_score",
    "yahoo_energy_proxy_score",
    "yahoo_volatility_proxy_score",
    "yahoo_korea_proxy_score",

    "risk_level",
    "risk_score",
    "opportunity_score",
    "macro_pressure_score",

    "future_return_1d",
    "future_return_5d",
    "future_return_20d",
    "future_outperform_5d",
    "future_max_drawdown_5d",
    "label_status",
]

DART_FLAG_COLUMNS = {
    "contract_supply": "dart_contract_supply",
    "new_investment": "dart_new_investment",
    "treasury_stock": "dart_treasury_stock",
    "dividend": "dart_dividend",
    "bonus_issue": "dart_bonus_issue",
    "paid_in_capital_increase": "dart_paid_in_capital_increase",
    "convertible_bond": "dart_convertible_bond",
    "ownership_change": "dart_ownership_change",
    "trading_halt": "dart_trading_halt",
    "delisting_risk": "dart_delisting_risk",
    "lawsuit": "dart_lawsuit",
    "embezzlement_breach": "dart_embezzlement_breach",
    "audit_opinion_risk": "dart_audit_opinion_risk",
    "correction_delay": "dart_correction_delay",
}

YAHOO_PROXY_GROUPS = {
    "semiconductor_proxy": {
        "column": "yahoo_semiconductor_proxy_score",
        "tickers": ["SOXX", "SMH"],
    },
    "rate_proxy": {
        "column": "yahoo_rate_proxy_score",
        "tickers": ["TLT", "IEF"],
    },
    "dollar_proxy": {
        "column": "yahoo_dollar_proxy_score",
        "tickers": ["UUP", "KRW=X"],
    },
    "energy_proxy": {
        "column": "yahoo_energy_proxy_score",
        "tickers": ["USO", "XLE"],
    },
    "volatility_proxy": {
        "column": "yahoo_volatility_proxy_score",
        "tickers": ["^VIX"],
    },
    "korea_proxy": {
        "column": "yahoo_korea_proxy_score",
        "tickers": ["EWY"],
    },
}


def save_csv(path, rows: List[Dict[str, Any]]):
    write_csv(path, rows, CSV_COLUMNS)


def to_int(value, default=0) -> int:
    return int(parse_number(value, default))


def to_float_or_none(value):
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def get_snapshot_date(price_features_raw: Dict[str, Any]) -> str:
    base_date = price_features_raw.get("base_date")

    if base_date:
        return str(base_date)

    kst = timezone.utc
    return datetime.now(kst).strftime("%Y%m%d")


def build_universe(
    price_features: Dict[str, Any],
    flow_features: Dict[str, Any],
    account_summary: Dict[str, Any],
    recommendation: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    universe: Dict[str, Dict[str, Any]] = {}

    for item in DEFAULT_UNIVERSE:
        code = normalize_code(item["stock_code"])
        universe[code] = {
            "stock_code": code,
            "stock_name": item["stock_name"],
            "sector": item["sector"],
        }

    for code, item in price_features.items():
        code = normalize_code(code)
        universe.setdefault(code, {})
        universe[code]["stock_code"] = code
        universe[code]["stock_name"] = item.get("stock_name") or universe[code].get("stock_name") or code
        universe[code]["sector"] = item.get("sector") or universe[code].get("sector") or KNOWN_STOCK_SECTOR.get(code, "unknown")

    for code, item in flow_features.items():
        code = normalize_code(code)
        universe.setdefault(code, {})
        universe[code]["stock_code"] = code
        universe[code]["stock_name"] = item.get("stock_name") or universe[code].get("stock_name") or code
        universe[code]["sector"] = item.get("sector") or universe[code].get("sector") or KNOWN_STOCK_SECTOR.get(code, "unknown")

    for item in account_summary.get("holdings", []):
        code = normalize_code(item.get("stock_code"))
        if not code:
            continue

        universe.setdefault(code, {})
        universe[code]["stock_code"] = code
        universe[code]["stock_name"] = item.get("stock_name") or universe[code].get("stock_name") or code
        universe[code]["sector"] = item.get("sector") or universe[code].get("sector") or KNOWN_STOCK_SECTOR.get(code, "unknown")

    for section in recommendation.get("combined_sectors", []):
        for asset in section.get("candidate_assets", []):
            code = normalize_code(asset.get("code"))
            if not code or code == "CASH":
                continue

            universe.setdefault(code, {})
            universe[code]["stock_code"] = code
            universe[code]["stock_name"] = asset.get("name") or universe[code].get("stock_name") or code
            universe[code]["sector"] = section.get("sector") or universe[code].get("sector") or KNOWN_STOCK_SECTOR.get(code, "unknown")

    return universe


def build_holding_map(account_summary: Dict[str, Any], total_equity: float) -> Dict[str, Dict[str, Any]]:
    result = {}

    for item in account_summary.get("holdings", []):
        code = normalize_code(item.get("stock_code"))

        if not code:
            continue

        eval_amount = parse_number(item.get("evaluation_amount"), 0)
        weight = eval_amount / total_equity if total_equity > 0 else 0

        result[code] = {
            "is_holding": 1,
            "holding_quantity": to_int(item.get("quantity"), 0),
            "holding_eval_amount": int(eval_amount),
            "holding_weight": round(weight, 6),
            "holding_pnl": to_int(item.get("pnl"), 0),
            "holding_return_rate": parse_number(item.get("return_rate"), 0),
        }

    return result


def build_recommendation_maps(recommendation: Dict[str, Any]):
    sector_map = {}
    stock_map = {}

    for idx, item in enumerate(recommendation.get("combined_sectors", []), 1):
        sector = item.get("sector")
        if not sector:
            continue

        sector_map[sector] = {
            "research_score": parse_number(item.get("research_score"), 0),
            "price_score": parse_number(item.get("price_score"), 0),
            "flow_score": parse_number(item.get("flow_score"), 0),
            "final_score": parse_number(item.get("final_score"), 0),
            "recommendation_rank": idx,
        }

    for item in recommendation.get("holding_recommendations", []):
        code = normalize_code(item.get("stock_code"))
        if not code:
            continue

        stock_map[code] = {
            "research_score": parse_number(item.get("sector_score"), 0),
            "price_score": parse_number(item.get("price_score"), 0),
            "flow_score": parse_number(item.get("flow_score"), 0),
            "final_score": parse_number(item.get("combined_quality"), 0),
            "recommendation_rank": 0,
        }

    target_weight_by_sector = {}

    for sector, weight in recommendation.get("target_allocation", {}).items():
        target_weight_by_sector[sector] = parse_number(weight, 0)

    return sector_map, stock_map, target_weight_by_sector


def normalize_dart_features(dart_raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    features = dart_raw.get("features", {}) if isinstance(dart_raw, dict) else {}
    if not isinstance(features, dict):
        return {}

    return {
        normalize_code(code): item
        for code, item in features.items()
        if isinstance(item, dict)
    }


def build_dart_columns(feature: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(feature, dict):
        result = {
            "dart_event_count_7d": 0,
            "dart_event_count_30d": 0,
            "dart_event_count_90d": 0,
            "dart_positive_event_count_30d": 0,
            "dart_negative_event_count_30d": 0,
            "dart_major_event_count_30d": 0,
            "dart_score": None,
            "dart_label": None,
        }
        result.update({column: 0 for column in DART_FLAG_COLUMNS.values()})
        return result

    flags = feature.get("feature_flags", {})
    if not isinstance(flags, dict):
        flags = {}

    result = {
        "dart_event_count_7d": feature.get("event_count_7d", 0),
        "dart_event_count_30d": feature.get("event_count_30d", 0),
        "dart_event_count_90d": feature.get("event_count_90d", 0),
        "dart_positive_event_count_30d": feature.get("positive_event_count_30d", 0),
        "dart_negative_event_count_30d": feature.get("negative_event_count_30d", 0),
        "dart_major_event_count_30d": feature.get("major_event_count_30d", 0),
        "dart_score": feature.get("dart_score"),
        "dart_label": feature.get("dart_label"),
    }

    for flag, column in DART_FLAG_COLUMNS.items():
        result[column] = 1 if flags.get(flag) else 0

    return result


def normalize_news_features(news_raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    features = news_raw.get("features", {}) if isinstance(news_raw, dict) else {}
    if not isinstance(features, dict):
        return {}

    return {
        normalize_code(code): item
        for code, item in features.items()
        if isinstance(item, dict)
    }


def build_news_columns(feature: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(feature, dict):
        return {
            "naver_news_count": 0,
            "naver_news_count_1d": 0,
            "naver_news_count_7d": 0,
            "naver_positive_keyword_count_7d": 0,
            "naver_negative_keyword_count_7d": 0,
            "naver_risk_keyword_count_7d": 0,
            "naver_attention_score": None,
            "naver_sentiment_score": None,
            "naver_news_label": None,
        }

    return {
        "naver_news_count": feature.get("news_count", 0),
        "naver_news_count_1d": feature.get("news_count_1d", 0),
        "naver_news_count_7d": feature.get("news_count_7d", 0),
        "naver_positive_keyword_count_7d": feature.get("positive_keyword_count_7d", 0),
        "naver_negative_keyword_count_7d": feature.get("negative_keyword_count_7d", 0),
        "naver_risk_keyword_count_7d": feature.get("risk_keyword_count_7d", 0),
        "naver_attention_score": feature.get("attention_score"),
        "naver_sentiment_score": feature.get("sentiment_score"),
        "naver_news_label": feature.get("news_label"),
    }


def normalize_yahoo_features(yahoo_raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(yahoo_raw, dict):
        yahoo_raw = {}

    features = yahoo_raw.get("features", {})
    sector_scores = yahoo_raw.get("sector_global_scores", {})
    proxy_group_scores = yahoo_raw.get("proxy_group_scores", {})

    if not isinstance(features, dict):
        features = {}
    if not isinstance(sector_scores, dict):
        sector_scores = {}
    if not isinstance(proxy_group_scores, dict):
        proxy_group_scores = {}

    return {
        "features": {
            str(ticker): item
            for ticker, item in features.items()
            if isinstance(item, dict)
        },
        "sector_global_scores": {
            str(sector): item
            for sector, item in sector_scores.items()
            if isinstance(item, dict)
        },
        "proxy_group_scores": {
            str(group): item
            for group, item in proxy_group_scores.items()
            if isinstance(item, dict)
        },
    }


def average_proxy_score(features: Dict[str, Dict[str, Any]], tickers: List[str], default: float = 50.0) -> float:
    values = [
        parse_number(features[ticker].get("global_signal_score"), default)
        for ticker in tickers
        if ticker in features
    ]
    if not values:
        return default
    return round(sum(values) / len(values), 4)


def yahoo_group_score(yahoo_context: Dict[str, Any], group_key: str, tickers: List[str]) -> float:
    group_scores = yahoo_context.get("proxy_group_scores", {})
    group = group_scores.get(group_key, {})
    if isinstance(group, dict) and group.get("proxy_count"):
        return round(parse_number(group.get("global_signal_score"), 50.0), 4)
    return average_proxy_score(yahoo_context.get("features", {}), tickers)


def build_yahoo_columns(sector: str, yahoo_context: Dict[str, Any]) -> Dict[str, Any]:
    sector_scores = yahoo_context.get("sector_global_scores", {})
    sector_score = sector_scores.get(sector, {}) if isinstance(sector_scores, dict) else {}
    if not isinstance(sector_score, dict):
        sector_score = {}

    labels = sector_score.get("related_proxy_labels")
    if isinstance(labels, list):
        related_proxy_labels = "|".join(str(label) for label in labels)
    else:
        related_proxy_labels = None

    result = {
        "yahoo_sector_global_signal_score": round(parse_number(sector_score.get("global_signal_score"), 50.0), 4),
        "yahoo_sector_global_risk_score": round(parse_number(sector_score.get("risk_score"), 0.0), 4),
        "yahoo_sector_proxy_count": int(parse_number(sector_score.get("proxy_count"), 0)),
        "yahoo_related_proxy_labels": related_proxy_labels,
    }

    for group_key, meta in YAHOO_PROXY_GROUPS.items():
        result[meta["column"]] = yahoo_group_score(yahoo_context, group_key, meta["tickers"])

    return result


def create_dataset_rows() -> List[Dict[str, Any]]:
    candidates = load_json(CANDIDATES_PATH, {"items": []})
    price_raw = load_json(PRICE_FEATURES_PATH, {"features": {}})
    flow_raw = load_json(FLOW_FEATURES_PATH, {"features": {}})
    dart_raw = load_json(DART_EVENT_FEATURES_PATH, {"features": {}})
    news_raw = load_json(NEWS_EVENT_FEATURES_PATH, {"features": {}})
    yahoo_raw = load_json(YAHOO_GLOBAL_FEATURES_PATH, {"features": {}, "sector_global_scores": {}, "proxy_group_scores": {}})
    account_summary = load_json(ACCOUNT_SUMMARY_PATH, {})
    recommendation = load_json(PORTFOLIO_RECOMMENDATION_PATH, {})

    price_features = price_raw.get("features", {})
    flow_features = flow_raw.get("features", {})

    if not isinstance(price_features, dict):
        price_features = {}

    if not isinstance(flow_features, dict):
        flow_features = {}

    normalized_price_features = {
        normalize_code(code): item
        for code, item in price_features.items()
    }

    normalized_flow_features = {
        normalize_code(code): item
        for code, item in flow_features.items()
    }
    normalized_dart_features = normalize_dart_features(dart_raw)
    normalized_news_features = normalize_news_features(news_raw)
    yahoo_context = normalize_yahoo_features(yahoo_raw)

    snapshot_date = get_snapshot_date(price_raw)
    generated_at = datetime.now(timezone.utc).isoformat()

    account = recommendation.get("account", {})
    research = recommendation.get("research", {})

    account_total_equity = parse_number(
        account.get("total_equity")
        or account_summary.get("estimated_total_equity"),
        0,
    )

    account_cash = parse_number(
        account.get("cash")
        or account_summary.get("available_cash")
        or account_summary.get("orderable_100_amount"),
        0,
    )

    if account_total_equity <= 0:
        account_total_equity = account_cash

    account_cash_weight = parse_number(account.get("cash_weight"), 0)
    account_invested_weight = parse_number(account.get("invested_weight"), 0)

    pending_or_reserved_amount = parse_number(
        account.get("pending_or_reserved_amount")
        or account_summary.get("pending_or_reserved_amount"),
        0,
    )

    holding_map = build_holding_map(account_summary, account_total_equity)
    sector_map, stock_map, target_weight_by_sector = build_recommendation_maps(recommendation)

    universe = build_universe(
        price_features=normalized_price_features,
        flow_features=normalized_flow_features,
        account_summary=account_summary,
        recommendation=recommendation,
    )

    rows = []

    for code, meta in sorted(universe.items()):
        sector = meta.get("sector") or KNOWN_STOCK_SECTOR.get(code, "unknown")
        stock_name = meta.get("stock_name") or code

        price = normalized_price_features.get(code, {})
        flow = normalized_flow_features.get(code, {})
        dart_columns = build_dart_columns(normalized_dart_features.get(code))
        news_columns = build_news_columns(normalized_news_features.get(code))
        yahoo_columns = build_yahoo_columns(sector, yahoo_context)
        holding = holding_map.get(code, {})

        sector_rec = sector_map.get(sector, {})
        stock_rec = stock_map.get(code, {})

        research_score = stock_rec.get("research_score", sector_rec.get("research_score", 0))
        price_score = stock_rec.get("price_score", sector_rec.get("price_score", 0))
        flow_score = stock_rec.get("flow_score", sector_rec.get("flow_score", 0))
        final_score = stock_rec.get("final_score", sector_rec.get("final_score", 0))
        recommendation_rank = stock_rec.get("recommendation_rank", sector_rec.get("recommendation_rank", 0))
        target_weight = target_weight_by_sector.get(sector, 0)

        row = {
            "snapshot_date": snapshot_date,
            "generated_at": generated_at,
            "stock_code": code,
            "stock_name": stock_name,
            "sector": sector,

            "is_holding": holding.get("is_holding", 0),
            "holding_quantity": holding.get("holding_quantity", 0),
            "holding_eval_amount": holding.get("holding_eval_amount", 0),
            "holding_weight": holding.get("holding_weight", 0),
            "holding_pnl": holding.get("holding_pnl", 0),
            "holding_return_rate": holding.get("holding_return_rate", 0),

            "account_total_equity": int(account_total_equity),
            "account_cash": int(account_cash),
            "account_cash_weight": round(account_cash_weight, 6),
            "account_invested_weight": round(account_invested_weight, 6),
            "pending_or_reserved_amount": int(pending_or_reserved_amount),

            "research_score": round(research_score, 4),
            "price_score": round(price_score, 4),
            "flow_score": round(flow_score, 4),
            "final_score": round(final_score, 4),
            "target_weight": round(target_weight, 6),
            "recommendation_rank": recommendation_rank,

            "latest_date": price.get("latest_date"),
            "latest_close": price.get("latest_close"),
            "latest_volume": price.get("latest_volume"),
            "return_1d": price.get("return_1d"),
            "return_5d": price.get("return_5d"),
            "return_20d": price.get("return_20d"),
            "volatility_20d": price.get("volatility_20d"),
            "ma20_gap": price.get("ma20_gap"),
            "ma60_gap": price.get("ma60_gap"),
            "drawdown_20d": price.get("drawdown_20d"),
            "volume_ratio_20": price.get("volume_ratio_20"),
            "trend_label": price.get("trend_label"),
            "price_risk_label": price.get("risk_label"),

            "foreign_net_1d": flow.get("foreign_net_1d"),
            "foreign_net_5d": flow.get("foreign_net_5d"),
            "foreign_net_20d": flow.get("foreign_net_20d"),
            "institution_net_1d": flow.get("institution_net_1d"),
            "institution_net_5d": flow.get("institution_net_5d"),
            "institution_net_20d": flow.get("institution_net_20d"),
            "combined_net_1d": flow.get("combined_net_1d"),
            "combined_net_5d": flow.get("combined_net_5d"),
            "combined_net_20d": flow.get("combined_net_20d"),
            "combined_net_5d_to_avg_volume_20": flow.get("combined_net_5d_to_avg_volume_20"),
            "combined_net_20d_to_avg_volume_20": flow.get("combined_net_20d_to_avg_volume_20"),
            "foreign_weight": flow.get("foreign_weight"),
            "foreign_limit_exhaustion_rate": flow.get("foreign_limit_exhaustion_rate"),
            "flow_label": flow.get("flow_label"),

            **dart_columns,
            **news_columns,
            **yahoo_columns,

            "risk_level": research.get("risk_level"),
            "risk_score": research.get("risk_score"),
            "opportunity_score": research.get("opportunity_score"),
            "macro_pressure_score": research.get("macro_pressure_score"),

            "future_return_1d": None,
            "future_return_5d": None,
            "future_return_20d": None,
            "future_outperform_5d": None,
            "future_max_drawdown_5d": None,
            "label_status": "unlabeled",
        }

        rows.append(row)

    return rows


def merge_with_existing_dataset(new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing_rows = load_jsonl(DATASET_JSONL_PATH)

    merged = {}

    for row in existing_rows:
        key = (str(row.get("snapshot_date")), str(row.get("stock_code")))

        if key[0] and key[1]:
            merged[key] = row

    for row in new_rows:
        key = (str(row.get("snapshot_date")), str(row.get("stock_code")))

        if key[0] and key[1]:
            existing = merged.get(key)

            if existing:
                # 기존 라벨이 있으면 보존
                for label_key in [
                    "future_return_1d",
                    "future_return_5d",
                    "future_return_20d",
                    "future_outperform_5d",
                    "future_max_drawdown_5d",
                    "label_status",
                ]:
                    if existing.get(label_key) is not None and existing.get("label_status") == "labeled":
                        row[label_key] = existing.get(label_key)

            merged[key] = row

    rows = list(merged.values())
    rows.sort(key=lambda x: (str(x.get("snapshot_date")), str(x.get("stock_code"))))

    return rows


def build_markdown_report(new_rows: List[Dict[str, Any]], all_rows: List[Dict[str, Any]]) -> str:
    lines = []

    lines.append("# 모델 학습 데이터셋 스냅샷")
    lines.append("")
    lines.append(f"- 생성 시각 UTC: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- 이번 스냅샷 종목 수: {len(new_rows)}")
    lines.append(f"- 누적 데이터셋 row 수: {len(all_rows)}")
    lines.append(f"- JSONL: `{DATASET_JSONL_PATH}`")
    lines.append(f"- CSV: `{DATASET_CSV_PATH}`")
    lines.append("")

    lines.append("## 1. 이번 스냅샷 상위 종목")
    ranked = sorted(
        new_rows,
        key=lambda x: parse_number(x.get("final_score"), 0),
        reverse=True,
    )

    for row in ranked[:10]:
        lines.append(
            f"- {row.get('stock_name')}({row.get('stock_code')}) "
            f"/ sector={row.get('sector')} "
            f"/ final={row.get('final_score')} "
            f"/ research={row.get('research_score')} "
            f"/ price={row.get('price_score')} "
            f"/ flow={row.get('flow_score')} "
            f"/ 5D={row.get('return_5d')} "
            f"/ vol20D={row.get('volatility_20d')} "
            f"/ flow_label={row.get('flow_label')}"
        )

    lines.append("")
    lines.append("## 2. 라벨 상태")
    labeled_count = sum(1 for row in all_rows if row.get("label_status") == "labeled")
    unlabeled_count = sum(1 for row in all_rows if row.get("label_status") != "labeled")

    lines.append(f"- labeled rows: {labeled_count}")
    lines.append(f"- unlabeled rows: {unlabeled_count}")
    lines.append("")

    lines.append("## 3. 다음 단계")
    lines.append("- 며칠간 스냅샷을 누적한 뒤 `label_dataset.py`에서 미래 수익률 라벨을 붙입니다.")
    lines.append("- 라벨이 충분히 쌓이면 `train_model.py`에서 XGBoost 또는 룰+모델 혼합 신호를 학습합니다.")
    lines.append("- 시계열 데이터이므로 랜덤 분할이 아니라 시간 순서 기반 검증을 사용해야 합니다.")

    return "\n".join(lines)


def main():
    new_rows = create_dataset_rows()
    all_rows = merge_with_existing_dataset(new_rows)

    save_jsonl(DATASET_JSONL_PATH, all_rows)
    save_csv(DATASET_CSV_PATH, all_rows)

    markdown = build_markdown_report(new_rows, all_rows)
    SNAPSHOT_REPORT_PATH.write_text(markdown, encoding="utf-8")

    print(f"Saved dataset jsonl: {DATASET_JSONL_PATH}")
    print(f"Saved dataset csv: {DATASET_CSV_PATH}")
    print(f"Saved report: {SNAPSHOT_REPORT_PATH}")
    print("")
    print(markdown)


if __name__ == "__main__":
    main()
