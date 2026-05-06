#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from _bootstrap import add_src_to_path

add_src_to_path()

from order_ledger import append_ledger_event, make_proposal_id
from risk_guard import run_risk_guard
from policylink.paths import (
    DART_EVENT_FEATURES_PATH,
    FLOW_FEATURES_PATH,
    KIWOOM_ACCOUNT_SUMMARY_PATH,
    MODEL_DATASET_CSV_PATH,
    NEWS_EVENT_FEATURES_PATH,
    ORDER_LEDGER_PATH,
    ORDER_PROPOSALS_JSON_PATH,
    ORDER_PROPOSALS_MD_PATH,
    PORTFOLIO_RECOMMENDATION_JSON_PATH,
    PRICE_FEATURES_PATH,
    REPORTS_DIR,
    YAHOO_GLOBAL_FEATURES_PATH,
)
from policylink.utils import load_json, normalize_code, parse_number


KST = timezone(timedelta(hours=9))
ML_SIGNALS_PATH = REPORTS_DIR / "ml_signals.json"

DEFAULT_SIGNAL_POLICY = {
    "primary_signal": "rule",
    "recommended_policy": "rule_first_ml_modifier",
    "rule_weight": 0.75,
    "ml_weight": 0.25,
    "auto_order_allowed": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def today_yyyymmdd() -> str:
    return datetime.now(KST).strftime("%Y%m%d")


def normalize_feature_map(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    features = raw.get("features", {}) if isinstance(raw, dict) else {}
    if not isinstance(features, dict):
        return {}
    return {
        normalize_code(code): item
        for code, item in features.items()
        if isinstance(item, dict)
    }


def load_model_dataset_latest(path: Path = MODEL_DATASET_CSV_PATH) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}

    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}

    if not rows:
        return {}

    latest_date = max(str(row.get("snapshot_date") or "") for row in rows)
    return {
        normalize_code(row.get("stock_code")): row
        for row in rows
        if str(row.get("snapshot_date") or "") == latest_date and normalize_code(row.get("stock_code"))
    }


def holdings_by_code(account_summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result = {}
    for item in account_summary.get("holdings", []) or []:
        if not isinstance(item, dict):
            continue
        code = normalize_code(item.get("stock_code") or item.get("stk_cd") or item.get("code"))
        if code:
            result[code] = item
    return result


def holding_quantity(item: Dict[str, Any]) -> int:
    return int(parse_number(item.get("available_quantity") or item.get("quantity") or item.get("rmnd_qty") or item.get("qty"), 0))


def holding_name(item: Dict[str, Any], code: str) -> str:
    return str(item.get("stock_name") or item.get("stk_nm") or item.get("name") or code)


def current_price_for(code: str, price_features: Dict[str, Dict[str, Any]]) -> float:
    feature = price_features.get(code, {})
    return parse_number(feature.get("latest_close"), 0.0)


def calc_recommended_qty(side: str, price: float, max_order_amount: int, holding_qty: int = 0, risk_level: str = "low") -> int:
    if side == "sell":
        return max(0, min(holding_qty, 1))
    if price <= 0:
        return 0

    effective_amount = max_order_amount
    if risk_level == "high":
        effective_amount = min(effective_amount, 500_000)

    qty = int(effective_amount // price)
    if qty <= 0:
        return 0
    return min(qty, 1)


def score_from_model_dataset(code: str, model_rows: Dict[str, Dict[str, Any]], key: str, default: Optional[float] = None) -> Optional[float]:
    row = model_rows.get(code, {})
    if key not in row or row.get(key) in {"", None}:
        return default
    return parse_number(row.get(key), default if default is not None else 0.0)


def yahoo_sector_score(sector: str, yahoo_features: Dict[str, Any]) -> Optional[float]:
    sector_scores = yahoo_features.get("sector_global_scores", {}) if isinstance(yahoo_features, dict) else {}
    item = sector_scores.get(sector, {}) if isinstance(sector_scores, dict) else {}
    if not isinstance(item, dict):
        return None
    return parse_number(item.get("global_signal_score"), 50.0)


def yahoo_sector_risk_off(sector: str, yahoo_features: Dict[str, Any]) -> bool:
    sector_scores = yahoo_features.get("sector_global_scores", {}) if isinstance(yahoo_features, dict) else {}
    item = sector_scores.get(sector, {}) if isinstance(sector_scores, dict) else {}
    if not isinstance(item, dict):
        return False
    return parse_number(item.get("global_signal_score"), 50.0) < 45 or parse_number(item.get("risk_score"), 0.0) >= 75


def load_inputs(allow_ml_signals: bool) -> Dict[str, Any]:
    ml_signals = load_json(ML_SIGNALS_PATH, {"signals": {}})
    if not isinstance(ml_signals, dict) or ml_signals.get("model_status") != "available":
        ml_signals = {"signals": {}, "signal_policy": DEFAULT_SIGNAL_POLICY, "warning": ["no_available_ml_model"]}
    signal_policy = ml_signals.get("signal_policy") if isinstance(ml_signals.get("signal_policy"), dict) else DEFAULT_SIGNAL_POLICY

    return {
        "portfolio": load_json(PORTFOLIO_RECOMMENDATION_JSON_PATH, {}),
        "account_summary": load_json(KIWOOM_ACCOUNT_SUMMARY_PATH, {}),
        "price_features": normalize_feature_map(load_json(PRICE_FEATURES_PATH, {"features": {}})),
        "flow_features": normalize_feature_map(load_json(FLOW_FEATURES_PATH, {"features": {}})),
        "dart_features": normalize_feature_map(load_json(DART_EVENT_FEATURES_PATH, {"features": {}})),
        "news_features": normalize_feature_map(load_json(NEWS_EVENT_FEATURES_PATH, {"features": {}})),
        "yahoo_features": load_json(YAHOO_GLOBAL_FEATURES_PATH, {"sector_global_scores": {}, "features": {}}),
        "model_rows": load_model_dataset_latest(),
        "ml_signals": ml_signals,
        "signal_policy": signal_policy,
    }


def should_exclude_buy(
    code: str,
    sector: str,
    price_feature: Dict[str, Any],
    flow_feature: Dict[str, Any],
    dart_feature: Dict[str, Any],
    news_feature: Dict[str, Any],
    yahoo_features: Dict[str, Any],
    ml_signal: Dict[str, Any],
    args: argparse.Namespace,
) -> List[str]:
    reasons = []

    if not price_feature:
        reasons.append("가격 피처 없음")
    else:
        if str(price_feature.get("trend_label") or "") == "downtrend":
            reasons.append("가격 downtrend")
        if str(price_feature.get("risk_label") or "") == "high_volatility":
            reasons.append("가격 high_volatility")
        if parse_number(price_feature.get("return_20d"), 0.0) > 0.15:
            reasons.append("20일 급등으로 추격매수 주의")

    if not flow_feature:
        reasons.append("수급 피처 없음")
    else:
        if parse_number(flow_feature.get("flow_score"), 0.0) < args.min_flow_score:
            reasons.append("flow_score 기준 미달")
        if parse_number(flow_feature.get("foreign_net_5d"), 0.0) < 0 and parse_number(flow_feature.get("institution_net_5d"), 0.0) < 0:
            reasons.append("외국인/기관 동시 순매도")

    if str(dart_feature.get("dart_label") or "") == "disclosure_risk":
        reasons.append("DART disclosure_risk")

    if str(news_feature.get("news_label") or "") == "negative_news_flow":
        reasons.append("Naver negative_news_flow")

    if yahoo_sector_risk_off(sector, yahoo_features):
        reasons.append("Yahoo global risk_off")

    if ml_signal and parse_number(ml_signal.get("predicted_drawdown_5d"), 0.0) < -0.04:
        reasons.append("ML predicted_drawdown_5d risk")

    if ml_signal and str(ml_signal.get("signal_label") or "") == "avoid_or_sell_candidate":
        reasons.append("ML signal_label=avoid_or_sell_candidate")

    return reasons


def make_scores(
    code: str,
    sector: str,
    section: Dict[str, Any],
    price_feature: Dict[str, Any],
    flow_feature: Dict[str, Any],
    dart_feature: Dict[str, Any],
    news_feature: Dict[str, Any],
    yahoo_features: Dict[str, Any],
    ml_signal: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "final_score": parse_number(section.get("final_score"), 0.0),
        "research_score": parse_number(section.get("research_score"), 0.0),
        "price_score": parse_number(price_feature.get("price_score"), section.get("price_score") or 0.0),
        "flow_score": parse_number(flow_feature.get("flow_score"), section.get("flow_score") or 0.0),
        "dart_score": dart_feature.get("dart_score"),
        "news_sentiment_score": news_feature.get("sentiment_score"),
        "yahoo_global_score": yahoo_sector_score(sector, yahoo_features),
        "ml_outperform_prob_5d": ml_signal.get("outperform_prob_5d") if ml_signal else None,
        "ml_predicted_return_5d": ml_signal.get("predicted_return_5d") if ml_signal else None,
        "ml_predicted_drawdown_5d": ml_signal.get("predicted_drawdown_5d") if ml_signal else None,
    }


def build_proposal(
    proposal_id: str,
    created_at: str,
    stock_code: str,
    stock_name: str,
    sector: str,
    side: str,
    proposal_type: str,
    recommended_qty: int,
    estimated_price: float,
    scores: Dict[str, Any],
    reasons: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    if side == "buy":
        order_method = "current_price_plus_buffer_limit"
    elif side == "sell":
        order_method = "current_price_minus_buffer_limit"
    else:
        order_method = "none"

    return {
        "proposal_id": proposal_id,
        "created_at": created_at,
        "mode": "mock_proposal_only",
        "proposal_only": True,
        "order_enabled": False,
        "stock_code": stock_code,
        "stock_name": stock_name,
        "sector": sector,
        "side": side,
        "proposal_type": proposal_type,
        "recommended_qty": int(recommended_qty),
        "estimated_price": round(float(estimated_price), 2) if estimated_price else 0,
        "estimated_amount": int(round(float(estimated_price) * recommended_qty)) if estimated_price and recommended_qty else 0,
        "order_method": order_method,
        "price_buffer_bps": 50 if side in {"buy", "sell"} else 0,
        "scores": scores,
        "reasons": reasons,
        "warnings": warnings,
        "risk_check_status": "pending",
        "execution_status": "not_executed",
        "telegram_approval_required": True,
    }


def generate_buy_proposals(inputs: Dict[str, Any], args: argparse.Namespace, start_seq: int, created_at: str) -> List[Dict[str, Any]]:
    portfolio = inputs["portfolio"]
    account_summary = inputs["account_summary"]
    price_features = inputs["price_features"]
    flow_features = inputs["flow_features"]
    dart_features = inputs["dart_features"]
    news_features = inputs["news_features"]
    yahoo_features = inputs["yahoo_features"]
    ml_signals = (inputs["ml_signals"].get("signals") or {}) if isinstance(inputs["ml_signals"], dict) else {}
    signal_policy = inputs.get("signal_policy") or DEFAULT_SIGNAL_POLICY
    holdings = holdings_by_code(account_summary)
    risk_level = str((portfolio.get("research") or {}).get("risk_level") or "low")

    proposals: List[Dict[str, Any]] = []
    seen_codes = set()
    seq = start_seq

    for section in portfolio.get("combined_sectors", []) or []:
        if len([item for item in proposals if item["side"] == "buy"]) >= args.max_buy_candidates:
            break

        if parse_number(section.get("final_score"), 0.0) < args.min_final_score:
            continue
        if parse_number(section.get("price_score"), 0.0) < args.min_price_score:
            continue
        if parse_number(section.get("flow_score"), 0.0) < args.min_flow_score:
            continue

        sector = str(section.get("sector") or "unknown")
        for asset in section.get("candidate_assets", []) or []:
            if len([item for item in proposals if item["side"] == "buy"]) >= args.max_buy_candidates:
                break
            if not isinstance(asset, dict):
                continue
            code = normalize_code(asset.get("code"))
            if not code or code == "CASH" or code in seen_codes:
                continue
            seen_codes.add(code)

            stock_name = str(asset.get("name") or code)
            price_feature = price_features.get(code, {})
            flow_feature = flow_features.get(code, {})
            dart_feature = dart_features.get(code, {})
            news_feature = news_features.get(code, {})
            ml_signal = ml_signals.get(code, {}) if isinstance(ml_signals, dict) else {}
            estimated_price = current_price_for(code, price_features)
            qty = calc_recommended_qty("buy", estimated_price, args.max_order_amount, risk_level=risk_level)
            scores = make_scores(code, sector, section, price_feature, flow_feature, dart_feature, news_feature, yahoo_features, ml_signal)

            warnings = []
            reasons = [
                f"섹터 최종점수 {section.get('final_score')} 기준 후보",
                f"가격점수 {scores['price_score']} / 수급점수 {scores['flow_score']}",
                f"Signal policy: {signal_policy.get('recommended_policy', 'rule_first_ml_modifier')}",
            ]
            exclude_reasons = should_exclude_buy(
                code, sector, price_feature, flow_feature, dart_feature, news_feature, yahoo_features, ml_signal, args
            )
            if exclude_reasons:
                warnings.extend(exclude_reasons)

            if parse_number(account_summary.get("pending_or_reserved_amount"), 0.0) > 0:
                warnings.append("미체결/주문대기 금액이 있어 risk_guard에서 reject될 수 있습니다.")

            if estimated_price <= 0 or qty <= 0:
                side = "watch"
                proposal_type = "watch_only"
                qty = 0
                warnings.append("현재가 또는 주문한도 기준으로 1주 후보 산출이 불가능해 관망 후보로 둡니다.")
            elif exclude_reasons:
                side = "watch"
                proposal_type = "watch_only"
                qty = 0
            elif code in holdings:
                side = "buy"
                proposal_type = "add_buy"
            else:
                side = "buy"
                proposal_type = "new_buy"

            if ml_signal and parse_number(ml_signal.get("outperform_prob_5d"), 0.0) >= 0.60:
                reasons.append("ML outperform_prob_5d >= 0.60 보조 긍정 근거")
                if ml_signal.get("model_source") != "active" or not ml_signal.get("calibrated"):
                    warnings.append("ML 긍정 신호는 단독 매수 근거가 아니며 rule-first modifier로만 사용")
            if ml_signal and parse_number(ml_signal.get("outperform_prob_5d"), 0.0) >= parse_number(ml_signal.get("threshold_used") or inputs["ml_signals"].get("best_threshold"), 0.60):
                if ml_signal.get("model_source") == "active" and ml_signal.get("calibrated"):
                    reasons.append("calibrated active ML threshold 이상")
                elif ml_signal.get("model_source") == "active":
                    reasons.append("active ML threshold 이상")
                else:
                    warnings.append("fallback ML threshold 이상이나 낮은 가중치로 참고")
            if ml_signal and parse_number(ml_signal.get("predicted_return_5d"), 0.0) > 0:
                reasons.append("ML predicted_return_5d 양수")
            if ml_signal and str(ml_signal.get("signal_label") or "") == "strong_buy_candidate":
                warnings.append("ML strong_buy_candidate도 단독 매수 근거로 사용하지 않습니다.")
            if ml_signal and parse_number(ml_signal.get("outperform_prob_5d"), 1.0) < 0.40:
                warnings.append("ML outperform_prob_5d 약세")
            if ml_signal and str(ml_signal.get("signal_label") or "") == "avoid_or_sell_candidate":
                warnings.append(f"ML avoid_or_sell_candidate source={ml_signal.get('model_source')}")
            if ml_signal and not ml_signal.get("calibrated"):
                warnings.append("ML probability is not calibrated; risk_guard 우선")

            proposal = build_proposal(
                proposal_id=make_proposal_id(snapshot_date=today_yyyymmdd(), sequence=seq),
                created_at=created_at,
                stock_code=code,
                stock_name=stock_name,
                sector=sector,
                side=side,
                proposal_type=proposal_type,
                recommended_qty=qty,
                estimated_price=estimated_price,
                scores=scores,
                reasons=reasons,
                warnings=warnings,
            )
            proposals.append(proposal)
            seq += 1

    return proposals


def sell_reasons_for(
    code: str,
    holding: Dict[str, Any],
    holding_rec: Dict[str, Any],
    price_feature: Dict[str, Any],
    flow_feature: Dict[str, Any],
    dart_feature: Dict[str, Any],
    news_feature: Dict[str, Any],
    ml_signal: Dict[str, Any],
) -> List[str]:
    reasons = []
    price_score = parse_number(holding_rec.get("price_score") or price_feature.get("price_score"), 50.0)
    flow_score = parse_number(holding_rec.get("flow_score") or flow_feature.get("flow_score"), 50.0)
    combined_quality = parse_number(holding_rec.get("combined_quality"), 50.0)

    if price_score < 40:
        reasons.append("price_score 40 미만")
    if flow_score < 40:
        reasons.append("flow_score 40 미만")
    if str(price_feature.get("trend_label") or "") == "downtrend":
        reasons.append("가격 downtrend")
    if str(flow_feature.get("flow_label") or "") == "outflow_pressure":
        reasons.append("수급 outflow_pressure")
    if str(dart_feature.get("dart_label") or "") == "disclosure_risk":
        reasons.append("DART disclosure_risk")
    if str(news_feature.get("news_label") or "") == "negative_news_flow":
        reasons.append("Naver negative_news_flow")
    if combined_quality < 40:
        reasons.append("combined_quality 낮음")
    if ml_signal and parse_number(ml_signal.get("outperform_prob_5d"), 1.0) < 0.40:
        reasons.append("ML outperform_prob_5d 약세")

    return reasons


def generate_sell_proposals(inputs: Dict[str, Any], args: argparse.Namespace, start_seq: int, created_at: str) -> List[Dict[str, Any]]:
    portfolio = inputs["portfolio"]
    price_features = inputs["price_features"]
    flow_features = inputs["flow_features"]
    dart_features = inputs["dart_features"]
    news_features = inputs["news_features"]
    yahoo_features = inputs["yahoo_features"]
    ml_signals = (inputs["ml_signals"].get("signals") or {}) if isinstance(inputs["ml_signals"], dict) else {}
    holdings = holdings_by_code(inputs["account_summary"])
    holding_rec_by_code = {
        normalize_code(item.get("stock_code")): item
        for item in portfolio.get("holding_recommendations", []) or []
        if isinstance(item, dict)
    }

    proposals: List[Dict[str, Any]] = []
    seq = start_seq

    for code, holding in holdings.items():
        if len(proposals) >= args.max_sell_candidates:
            break

        holding_rec = holding_rec_by_code.get(code, {})
        sector = str(holding.get("sector") or holding_rec.get("sector") or "unknown")
        price_feature = price_features.get(code, {})
        flow_feature = flow_features.get(code, {})
        dart_feature = dart_features.get(code, {})
        news_feature = news_features.get(code, {})
        ml_signal = ml_signals.get(code, {}) if isinstance(ml_signals, dict) else {}
        reasons = sell_reasons_for(code, holding, holding_rec, price_feature, flow_feature, dart_feature, news_feature, ml_signal)
        if not reasons:
            continue

        estimated_price = current_price_for(code, price_features)
        qty = calc_recommended_qty("sell", estimated_price, args.max_order_amount, holding_qty=holding_quantity(holding))
        scores = make_scores(
            code=code,
            sector=sector,
            section={
                "final_score": holding_rec.get("combined_quality", 0),
                "research_score": holding_rec.get("sector_score", 0),
                "price_score": holding_rec.get("price_score", 0),
                "flow_score": holding_rec.get("flow_score", 0),
            },
            price_feature=price_feature,
            flow_feature=flow_feature,
            dart_feature=dart_feature,
            news_feature=news_feature,
            yahoo_features=yahoo_features,
            ml_signal=ml_signal,
        )

        proposals.append(build_proposal(
            proposal_id=make_proposal_id(snapshot_date=today_yyyymmdd(), sequence=seq),
            created_at=created_at,
            stock_code=code,
            stock_name=holding_name(holding, code),
            sector=sector,
            side="sell",
            proposal_type="reduce_sell",
            recommended_qty=qty,
            estimated_price=estimated_price,
            scores=scores,
            reasons=reasons,
            warnings=["매도 후보도 proposal_only이며 자동 주문으로 넘어가지 않습니다."],
        ))
        seq += 1

    return proposals


def apply_risk_results(proposals: List[Dict[str, Any]], risk_result: Dict[str, Any]) -> None:
    status_by_id = {
        item.get("proposal_id"): item
        for item in risk_result.get("checked_proposals", []) or []
        if isinstance(item, dict)
    }

    for proposal in proposals:
        checked = status_by_id.get(proposal.get("proposal_id"), {})
        if not checked:
            continue
        proposal["risk_check_status"] = checked.get("proposal_status", "pending")
        proposal["risk_reasons"] = checked.get("risk_reasons", [])
        proposal["risk_warnings"] = checked.get("risk_warnings", [])
        proposal["recommended_amount_after_guard"] = checked.get("recommended_amount_after_guard", 0)
        proposal["order_enabled"] = False
        proposal["execution_status"] = "not_executed"


def build_payload(proposals: List[Dict[str, Any]], risk_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "mode": "mock_proposal_only",
        "proposal_only": True,
        "order_enabled": False,
        "execution_status": "not_executed",
        "source": "portfolio_recommendation",
        "signal_policy": DEFAULT_SIGNAL_POLICY,
        "proposals": proposals,
        "risk_guard_summary": (risk_result or {}).get("summary", {}),
        "risk_guard_report": str(ORDER_PROPOSALS_MD_PATH.parent / "order_risk_check.md"),
    }


def save_proposals(payload: Dict[str, Any]) -> None:
    ORDER_PROPOSALS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    ORDER_PROPOSALS_JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ORDER_PROPOSALS_MD_PATH.write_text(build_markdown_report(payload), encoding="utf-8")


def append_proposals_to_ledger(proposals: List[Dict[str, Any]]) -> None:
    for proposal in proposals:
        append_ledger_event({
            "event_type": "proposal",
            "snapshot_date": today_yyyymmdd(),
            "proposal_id": proposal.get("proposal_id"),
            "stock_code": proposal.get("stock_code"),
            "stock_name": proposal.get("stock_name"),
            "sector": proposal.get("sector"),
            "side": proposal.get("side"),
            "proposal_type": proposal.get("proposal_type"),
            "risk_check_status": proposal.get("risk_check_status"),
            "estimated_amount": proposal.get("estimated_amount"),
            "recommended_amount_after_guard": proposal.get("recommended_amount_after_guard"),
            "order_enabled": False,
            "proposal_only": True,
            "execution_status": "not_executed",
        })


def build_markdown_report(payload: Dict[str, Any]) -> str:
    proposals = payload.get("proposals", [])
    risk_summary = payload.get("risk_guard_summary", {})
    buys = [item for item in proposals if item.get("side") == "buy"]
    sells = [item for item in proposals if item.get("side") == "sell"]
    watches = [item for item in proposals if item.get("side") == "watch"]
    signal_policy = payload.get("signal_policy") or DEFAULT_SIGNAL_POLICY

    lines = [
        "# 주문 후보 리포트",
        "",
        f"- 생성 시각: {payload.get('generated_at')}",
        "- 실제 주문 실행: 비활성화",
        "- order_enabled=false",
        "- proposal_only=true",
        "- execution_status=not_executed",
        f"- Signal policy: {signal_policy.get('recommended_policy', 'rule_first_ml_modifier')}",
        f"- Rule weight: {signal_policy.get('rule_weight', 0.75)}",
        f"- ML weight: {signal_policy.get('ml_weight', 0.25)}",
        "",
        "## Signal Policy",
        f"- primary_signal: {signal_policy.get('primary_signal', 'rule')}",
        f"- recommended_policy: {signal_policy.get('recommended_policy', 'rule_first_ml_modifier')}",
        "- ML probability is not calibrated이면 ML은 보조 modifier로만 사용합니다.",
        "- Signal policy: rule-first, ML modifier",
        "",
        "## 오늘의 매수 후보",
    ]

    if buys:
        for item in buys:
            lines.append(
                f"- {item.get('proposal_id')} / {item.get('stock_name')}({item.get('stock_code')}) "
                f"/ qty={item.get('recommended_qty')} / amount={item.get('estimated_amount'):,}원 "
                f"/ risk={item.get('risk_check_status')}"
            )
            lines.append(f"  - 이유: {', '.join(item.get('reasons', [])[:5])}")
            if item.get("warnings"):
                lines.append(f"  - 경고: {', '.join(item.get('warnings', [])[:5])}")
            if item.get("risk_reasons"):
                lines.append(f"  - risk reject: {', '.join(item.get('risk_reasons', [])[:5])}")
    else:
        lines.append("- 매수 후보 없음")

    lines.extend(["", "## 오늘의 매도 후보"])
    if sells:
        for item in sells:
            lines.append(
                f"- {item.get('proposal_id')} / {item.get('stock_name')}({item.get('stock_code')}) "
                f"/ qty={item.get('recommended_qty')} / risk={item.get('risk_check_status')}"
            )
            lines.append(f"  - 이유: {', '.join(item.get('reasons', [])[:5])}")
    else:
        lines.append("- 매도 후보 없음")

    lines.extend(["", "## 관망 후보"])
    if watches:
        for item in watches:
            lines.append(
                f"- {item.get('proposal_id')} / {item.get('stock_name')}({item.get('stock_code')}) "
                f"/ status={item.get('risk_check_status')}"
            )
            if item.get("warnings"):
                lines.append(f"  - 사유: {', '.join(item.get('warnings', [])[:5])}")
    else:
        lines.append("- 관망 후보 없음")

    lines.extend([
        "",
        "## 리스크 가드 결과 요약",
        f"- 전체 후보: {risk_summary.get('total_proposals', 0)}",
        f"- 검토 가능: {risk_summary.get('approved_for_review', 0)}",
        f"- 축소 필요: {risk_summary.get('reduced_size', 0)}",
        f"- reject: {risk_summary.get('rejected', 0)}",
        f"- 관망: {risk_summary.get('watch_only', 0)}",
        "",
        "## Telegram 승인 문구 예시",
        '- "승인 기능은 아직 비활성화되어 있습니다. 다음 단계에서 execute_approved_mock_order.py를 만들 때 사용합니다."',
        "",
        "## 주의사항",
        "- 이 리포트는 주문 후보와 리스크 검사 결과만 제공합니다.",
        "- 키움 주문 API와 주문 TR은 호출하지 않습니다.",
        "- 자동매매는 아직 비활성화 상태입니다.",
    ])

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate proposal-only buy/sell candidates from portfolio recommendations.")
    parser.add_argument("--max-buy-candidates", type=int, default=3)
    parser.add_argument("--max-sell-candidates", type=int, default=3)
    parser.add_argument("--max-order-amount", type=int, default=1_000_000)
    parser.add_argument("--proposal-only", default="true")
    parser.add_argument("--allow-ml-signals", action="store_true")
    parser.add_argument("--min-final-score", type=float, default=70.0)
    parser.add_argument("--min-price-score", type=float, default=55.0)
    parser.add_argument("--min-flow-score", type=float, default=55.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(args.proposal_only).lower() not in {"true", "1", "yes"}:
        print("generate_order_proposals only supports proposal-only mode.")
        return 1

    inputs = load_inputs(allow_ml_signals=args.allow_ml_signals)
    created_at = utc_now()

    buy_proposals = generate_buy_proposals(inputs, args, start_seq=1, created_at=created_at)
    sell_proposals = generate_sell_proposals(inputs, args, start_seq=len(buy_proposals) + 1, created_at=created_at)
    proposals = buy_proposals + sell_proposals

    pre_risk_payload = build_payload(proposals)
    pre_risk_payload["signal_policy"] = inputs.get("signal_policy") or DEFAULT_SIGNAL_POLICY
    save_proposals(pre_risk_payload)

    risk_result = run_risk_guard(pre_risk_payload)
    apply_risk_results(proposals, risk_result)

    final_payload = build_payload(proposals, risk_result=risk_result)
    final_payload["signal_policy"] = inputs.get("signal_policy") or DEFAULT_SIGNAL_POLICY
    save_proposals(final_payload)
    append_proposals_to_ledger(proposals)

    print(f"Saved proposals json: {ORDER_PROPOSALS_JSON_PATH}")
    print(f"Saved proposals report: {ORDER_PROPOSALS_MD_PATH}")
    print(f"Saved risk check json: {ORDER_PROPOSALS_JSON_PATH.parent / 'order_risk_check.json'}")
    print(f"Saved risk check report: {ORDER_PROPOSALS_JSON_PATH.parent / 'order_risk_check.md'}")
    print(f"Saved ledger: {ORDER_LEDGER_PATH}")
    print(f"order_enabled={str(final_payload.get('order_enabled')).lower()}")
    print(f"proposal_only={str(final_payload.get('proposal_only')).lower()}")
    print(f"execution_status={final_payload.get('execution_status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
