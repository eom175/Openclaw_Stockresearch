#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from _bootstrap import add_src_to_path

add_src_to_path()

from order_ledger import today_existing_proposals
from policylink.paths import (
    DART_EVENT_FEATURES_PATH,
    FLOW_FEATURES_PATH,
    KIWOOM_ACCOUNT_SUMMARY_PATH,
    NEWS_EVENT_FEATURES_PATH,
    ORDER_PROPOSALS_JSON_PATH,
    ORDER_RISK_CHECK_JSON_PATH,
    ORDER_RISK_CHECK_MD_PATH,
    PORTFOLIO_RECOMMENDATION_JSON_PATH,
    PRICE_FEATURES_PATH,
    YAHOO_GLOBAL_FEATURES_PATH,
)
from policylink.utils import load_json, normalize_code, parse_number


KST = timezone(timedelta(hours=9))

MAX_ORDER_AMOUNT_PER_SYMBOL = 1_000_000
MAX_SINGLE_SYMBOL_WEIGHT = 0.05
MAX_SECTOR_WEIGHT = 0.15


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def today_yyyymmdd() -> str:
    return datetime.now(KST).strftime("%Y%m%d")


def clamp_amount(value: float) -> int:
    return max(0, int(value))


def normalize_feature_map(raw: Dict[str, Any], code_keys: bool = True) -> Dict[str, Dict[str, Any]]:
    features = raw.get("features", {}) if isinstance(raw, dict) else {}
    if not isinstance(features, dict):
        return {}

    result = {}
    for key, item in features.items():
        if not isinstance(item, dict):
            continue
        normalized = normalize_code(key) if code_keys else str(key)
        result[normalized] = item
    return result


def holdings_by_code(account_summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result = {}
    for item in account_summary.get("holdings", []) or []:
        if not isinstance(item, dict):
            continue
        code = normalize_code(item.get("stock_code") or item.get("stk_cd") or item.get("code"))
        if not code:
            continue
        result[code] = item
    return result


def holding_eval_amount(item: Optional[Dict[str, Any]]) -> float:
    if not item:
        return 0.0
    return parse_number(item.get("evaluation_amount") or item.get("eval_amount") or item.get("evlt_amt"), 0.0)


def holding_quantity(item: Optional[Dict[str, Any]]) -> int:
    if not item:
        return 0
    return int(parse_number(item.get("available_quantity") or item.get("quantity") or item.get("rmnd_qty") or item.get("qty"), 0))


def get_total_equity(account_summary: Dict[str, Any]) -> float:
    total = parse_number(account_summary.get("estimated_total_equity"), 0.0)
    if total <= 0:
        total = parse_number(account_summary.get("total_equity"), 0.0)
    if total <= 0:
        cash = get_orderable_cash(account_summary)
        holdings_value = sum(holding_eval_amount(item) for item in account_summary.get("holdings", []) or [])
        total = cash + holdings_value
    return total if total > 0 else 1.0


def get_orderable_cash(account_summary: Dict[str, Any]) -> float:
    cash = parse_number(account_summary.get("orderable_100_amount"), 0.0)
    if cash <= 0:
        cash = parse_number(account_summary.get("available_cash"), 0.0)
    if cash <= 0:
        cash = parse_number(account_summary.get("cash"), 0.0)
    return cash


def current_sector_exposure(account_summary: Dict[str, Any], portfolio: Dict[str, Any]) -> Dict[str, float]:
    sector_by_code = {}
    for section in portfolio.get("combined_sectors", []) or []:
        if not isinstance(section, dict):
            continue
        sector = section.get("sector")
        for asset in section.get("candidate_assets", []) or []:
            if not isinstance(asset, dict):
                continue
            code = normalize_code(asset.get("code"))
            if code:
                sector_by_code[code] = sector

    result: Dict[str, float] = {}
    for item in account_summary.get("holdings", []) or []:
        if not isinstance(item, dict):
            continue
        code = normalize_code(item.get("stock_code") or item.get("stk_cd") or item.get("code"))
        sector = item.get("sector") or sector_by_code.get(code) or "unknown"
        result[sector] = result.get(sector, 0.0) + holding_eval_amount(item)
    return result


def max_daily_new_buy_amount(risk_level: str) -> int:
    if risk_level == "high":
        return 1_000_000
    if risk_level == "medium":
        return 2_000_000
    return 3_000_000


def hard_reject_reasons(proposal: Dict[str, Any], mock_mode: bool, status: str, risk_reasons: List[str]) -> List[str]:
    reasons: List[str] = []
    code = normalize_code(proposal.get("stock_code"))
    side = str(proposal.get("side") or "")
    estimated_price = parse_number(proposal.get("estimated_price"), 0.0)
    estimated_amount = parse_number(proposal.get("estimated_amount"), 0.0)

    if not isinstance(proposal, dict) or not proposal.get("proposal_id"):
        reasons.append("malformed proposal")
    if not code:
        reasons.append("missing stock_code")
    if side not in {"buy", "sell", "watch"}:
        reasons.append("invalid side")
    if estimated_price <= 0:
        reasons.append("missing current price")
    if side in {"buy", "sell"} and estimated_amount <= 0:
        reasons.append("invalid amount")
    if not mock_mode:
        reasons.append("non-mock mode")
    if proposal.get("order_enabled") is not False or proposal.get("proposal_only") is not True:
        reasons.append("security issue: proposal_only/order_enabled=false required")

    hard_markers = [
        "side 값",
        "proposal_only/order_enabled=false",
        "mock mode",
        "예상 주문금액이 0 이하",
        "가격 피처가 없어",
    ]
    for reason in risk_reasons:
        if any(marker in str(reason) for marker in hard_markers):
            reasons.append(str(reason))

    return list(dict.fromkeys(reasons))


def paper_tracking_decision(
    proposal: Dict[str, Any],
    mock_mode: bool,
    status: str,
    risk_reasons: List[str],
    risk_warnings: List[str],
) -> Tuple[str, List[str], List[str]]:
    hard = hard_reject_reasons(proposal, mock_mode, status, risk_reasons)
    if proposal.get("tracking_status") == "excluded":
        hard.append("proposal tracking_status=excluded")

    if hard:
        return "do_not_track", [], hard

    reasons = list(proposal.get("paper_tracking_reason") or [])
    if status in {"approved_for_review", "reduced_size"}:
        reasons.append("execution_candidate passed risk guard review")
    elif status == "watch_only":
        reasons.append("watch_only candidate is useful for signal tracking")
    elif status == "rejected":
        reasons.append("soft rejected candidate kept for paper tracking only")

    soft_markers = [
        "high_volatility",
        "downtrend",
        "risk_off",
        "현금",
        "미체결",
        "주문대기",
        "섹터 비중",
        "종목 비중",
        "뉴스",
        "flow_score",
        "순매도",
        "disclosure_risk",
        "변동성",
    ]
    for reason in risk_reasons + risk_warnings:
        text = str(reason)
        if any(marker in text for marker in soft_markers):
            reasons.append(text)

    return "track", list(dict.fromkeys(reasons)), []


def is_mock_mode(account_summary: Dict[str, Any], portfolio: Dict[str, Any], proposals_payload: Dict[str, Any]) -> bool:
    mode_text = " ".join([
        str(account_summary.get("env") or ""),
        str(account_summary.get("mode") or ""),
        str(portfolio.get("mode") or ""),
        str(proposals_payload.get("mode") or ""),
    ]).lower()
    return "mock" in mode_text


def previous_duplicate_keys(snapshot_date: str) -> set:
    keys = set()
    for event in today_existing_proposals(snapshot_date):
        code = normalize_code(event.get("stock_code"))
        side = str(event.get("side") or "")
        if code and side:
            keys.add((code, side))
    return keys


def load_context() -> Dict[str, Any]:
    return {
        "account_summary": load_json(KIWOOM_ACCOUNT_SUMMARY_PATH, {}),
        "portfolio": load_json(PORTFOLIO_RECOMMENDATION_JSON_PATH, {}),
        "price_features": normalize_feature_map(load_json(PRICE_FEATURES_PATH, {"features": {}})),
        "flow_features": normalize_feature_map(load_json(FLOW_FEATURES_PATH, {"features": {}})),
        "dart_features": normalize_feature_map(load_json(DART_EVENT_FEATURES_PATH, {"features": {}})),
        "news_features": normalize_feature_map(load_json(NEWS_EVENT_FEATURES_PATH, {"features": {}})),
        "yahoo_global": load_json(YAHOO_GLOBAL_FEATURES_PATH, {"sector_global_scores": {}, "features": {}}),
    }


def check_buy_risks(
    proposal: Dict[str, Any],
    context: Dict[str, Any],
    daily_state: Dict[str, Any],
    duplicate_keys: set,
) -> Tuple[str, List[str], List[str], int, int]:
    code = normalize_code(proposal.get("stock_code"))
    sector = str(proposal.get("sector") or "unknown")
    account_summary = context["account_summary"]
    portfolio = context["portfolio"]
    price = context["price_features"].get(code)
    flow = context["flow_features"].get(code)
    dart = context["dart_features"].get(code, {})
    news = context["news_features"].get(code, {})
    yahoo_sector = (context["yahoo_global"].get("sector_global_scores") or {}).get(sector, {})

    reasons: List[str] = []
    warnings: List[str] = []
    estimated_amount = clamp_amount(parse_number(proposal.get("estimated_amount"), 0.0))
    cash = get_orderable_cash(account_summary)
    total_equity = get_total_equity(account_summary)
    holding = daily_state["holdings"].get(code)
    current_amount = holding_eval_amount(holding)
    sector_amount = daily_state["sector_exposure"].get(sector, 0.0)
    pending_amount = parse_number(account_summary.get("pending_or_reserved_amount"), 0.0)
    daily_limit = daily_state["daily_buy_limit"]
    daily_remaining = max(0, daily_limit - daily_state["approved_buy_amount"])
    max_allowed = min(MAX_ORDER_AMOUNT_PER_SYMBOL, cash, daily_remaining)

    if (code, "buy") in duplicate_keys:
        warnings.append("duplicate_warning: 같은 종목/side의 오늘 기존 proposal이 ledger에 있습니다.")

    if pending_amount > 0:
        reasons.append("미체결/주문대기 금액이 있어 신규 매수 후보는 reject합니다.")

    if estimated_amount <= 0:
        reasons.append("예상 주문금액이 0 이하입니다.")

    if estimated_amount > cash:
        reasons.append("주문 가능 현금보다 예상 주문금액이 큽니다.")

    if not price:
        reasons.append("가격 피처가 없어 신규 매수 후보를 reject합니다.")
    else:
        trend_label = str(price.get("trend_label") or "")
        risk_label = str(price.get("risk_label") or "")
        volatility_20d = parse_number(price.get("volatility_20d"), 0.0)
        return_20d = parse_number(price.get("return_20d"), 0.0)
        if trend_label == "downtrend":
            reasons.append("가격 trend_label=downtrend로 신규 매수 후보를 reject합니다.")
        if risk_label == "high_volatility":
            reasons.append("가격 risk_label=high_volatility로 신규 매수 후보를 reject합니다.")
        if volatility_20d >= 0.04:
            reasons.append("20일 변동성이 4% 이상입니다.")
        if return_20d > 0.15:
            warnings.append("20일 수익률 15% 초과로 추격매수 risk가 있습니다.")
            max_allowed = min(max_allowed, max(0, estimated_amount // 2))

    if not flow:
        reasons.append("수급 피처가 없어 신규 매수 후보를 reject합니다.")
    else:
        flow_score = parse_number(flow.get("flow_score"), 0.0)
        foreign_5d = parse_number(flow.get("foreign_net_5d"), 0.0)
        institution_5d = parse_number(flow.get("institution_net_5d"), 0.0)
        if flow_score < 40:
            reasons.append("flow_score가 40 미만입니다.")
        if foreign_5d < 0 and institution_5d < 0:
            reasons.append("외국인5D와 기관5D가 모두 순매도입니다.")

    if str(dart.get("dart_label") or "") == "disclosure_risk":
        reasons.append("DART disclosure_risk가 감지되었습니다.")

    news_sentiment = parse_number(news.get("sentiment_score") or proposal.get("scores", {}).get("news_sentiment_score"), 50.0)
    if news_sentiment < 40:
        reasons.append("Naver 뉴스 sentiment가 40 미만입니다.")

    yahoo_score = parse_number(yahoo_sector.get("global_signal_score"), proposal.get("scores", {}).get("yahoo_global_score") or 50.0)
    yahoo_risk = parse_number(yahoo_sector.get("risk_score"), 0.0)
    if yahoo_score < 45 or yahoo_risk >= 75:
        reasons.append("Yahoo 글로벌 proxy risk_off가 강합니다.")
    elif yahoo_score < 50 or yahoo_risk >= 60:
        warnings.append("Yahoo 글로벌 proxy 리스크로 주문금액을 축소합니다.")
        max_allowed = min(max_allowed, max(0, estimated_amount // 2))

    projected_single_weight = (current_amount + estimated_amount) / total_equity if total_equity > 0 else 1.0
    if projected_single_weight > MAX_SINGLE_SYMBOL_WEIGHT:
        reasons.append("추천 주문 후 종목 비중이 5%를 초과합니다.")

    projected_sector_weight = (sector_amount + estimated_amount) / total_equity if total_equity > 0 else 1.0
    if projected_sector_weight > MAX_SECTOR_WEIGHT:
        reasons.append("추천 주문 후 섹터 비중이 15%를 초과합니다.")

    if estimated_amount > MAX_ORDER_AMOUNT_PER_SYMBOL:
        warnings.append("종목당 최대 주문금액 100만원을 초과해 축소합니다.")

    if estimated_amount > daily_remaining:
        warnings.append("하루 신규 진입 한도를 초과해 축소합니다.")

    recommended_after_guard = min(estimated_amount, clamp_amount(max_allowed))

    if reasons:
        return "rejected", reasons, warnings, clamp_amount(max_allowed), 0

    if recommended_after_guard < estimated_amount:
        if recommended_after_guard <= 0:
            return "rejected", ["리스크 가드 후 허용 주문금액이 0원입니다."], warnings, clamp_amount(max_allowed), 0
        daily_state["approved_buy_amount"] += recommended_after_guard
        return "reduced_size", reasons, warnings, clamp_amount(max_allowed), recommended_after_guard

    daily_state["approved_buy_amount"] += recommended_after_guard
    return "approved_for_review", reasons, warnings, clamp_amount(max_allowed), recommended_after_guard


def check_sell_risks(
    proposal: Dict[str, Any],
    context: Dict[str, Any],
    duplicate_keys: set,
) -> Tuple[str, List[str], List[str], int, int]:
    code = normalize_code(proposal.get("stock_code"))
    holdings = holdings_by_code(context["account_summary"])
    holding = holdings.get(code)
    reasons: List[str] = []
    warnings: List[str] = []
    estimated_amount = clamp_amount(parse_number(proposal.get("estimated_amount"), 0.0))

    if (code, "sell") in duplicate_keys:
        warnings.append("duplicate_warning: 같은 종목/side의 오늘 기존 proposal이 ledger에 있습니다.")

    if not holding:
        reasons.append("보유 종목이 아니므로 매도 후보를 reject합니다.")
    elif holding_quantity(holding) <= 0:
        reasons.append("매도 가능 수량이 0 이하입니다.")

    if reasons:
        return "rejected", reasons, warnings, estimated_amount, 0

    warnings.append("매도 후보도 proposal_only이며 자동 주문으로 넘어가지 않습니다.")
    return "approved_for_review", reasons, warnings, estimated_amount, estimated_amount


def check_proposals(proposals_payload: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    context = context or load_context()
    account_summary = context["account_summary"]
    portfolio = context["portfolio"]
    proposals = proposals_payload.get("proposals", [])
    if not isinstance(proposals, list):
        proposals = []

    snapshot_date = today_yyyymmdd()
    duplicate_keys = previous_duplicate_keys(snapshot_date)
    daily_state = {
        "approved_buy_amount": 0,
        "daily_buy_limit": max_daily_new_buy_amount(str((portfolio.get("research") or {}).get("risk_level") or "low")),
        "holdings": holdings_by_code(account_summary),
        "sector_exposure": current_sector_exposure(account_summary, portfolio),
    }

    mock_mode = is_mock_mode(account_summary, portfolio, proposals_payload)
    checked = []
    seen_in_run = set()

    for proposal in proposals:
        code = normalize_code(proposal.get("stock_code"))
        side = str(proposal.get("side") or "")
        risk_reasons: List[str] = []
        risk_warnings: List[str] = []

        if (code, side) in seen_in_run:
            risk_warnings.append("duplicate_warning: 같은 실행 안에서 중복 proposal이 생성되었습니다.")
        seen_in_run.add((code, side))

        if not mock_mode:
            status = "rejected"
            risk_reasons.append("mock mode가 아니므로 주문 후보를 reject합니다.")
            max_allowed_amount = 0
            recommended_after_guard = 0
        elif proposal.get("order_enabled") is not False or proposal.get("proposal_only") is not True:
            status = "rejected"
            risk_reasons.append("proposal_only/order_enabled=false 상태가 아니므로 reject합니다.")
            max_allowed_amount = 0
            recommended_after_guard = 0
        elif side == "watch" or proposal.get("proposal_type") == "watch_only":
            status = "watch_only"
            max_allowed_amount = 0
            recommended_after_guard = 0
            risk_warnings.append("관망 후보입니다. 주문 검토 대상이 아닙니다.")
        elif side == "buy":
            status, risk_reasons, extra_warnings, max_allowed_amount, recommended_after_guard = check_buy_risks(
                proposal=proposal,
                context=context,
                daily_state=daily_state,
                duplicate_keys=duplicate_keys,
            )
            risk_warnings.extend(extra_warnings)
        elif side == "sell":
            status, risk_reasons, extra_warnings, max_allowed_amount, recommended_after_guard = check_sell_risks(
                proposal=proposal,
                context=context,
                duplicate_keys=duplicate_keys,
            )
            risk_warnings.extend(extra_warnings)
        else:
            status = "rejected"
            risk_reasons.append("side 값이 buy/sell/watch 중 하나가 아닙니다.")
            max_allowed_amount = 0
            recommended_after_guard = 0

        checked.append({
            "proposal_id": proposal.get("proposal_id"),
            "stock_code": code,
            "stock_name": proposal.get("stock_name"),
            "side": side,
            "proposal_status": status,
            "execution_status": status,
            "risk_reasons": risk_reasons,
            "risk_warnings": risk_warnings,
            "max_allowed_amount": max_allowed_amount,
            "recommended_amount_after_guard": recommended_after_guard,
            "tracking_status": proposal.get("tracking_status", "execution_candidate"),
            "paper_tracking_status": paper_tracking_decision(
                proposal, mock_mode, status, risk_reasons, risk_warnings
            )[0],
            "paper_tracking_reason": paper_tracking_decision(
                proposal, mock_mode, status, risk_reasons, risk_warnings
            )[1],
            "hard_reject_reason": paper_tracking_decision(
                proposal, mock_mode, status, risk_reasons, risk_warnings
            )[2],
            "can_execute": False,
            "order_enabled": False,
        })

    summary = {
        "total_proposals": len(checked),
        "approved_for_review": sum(1 for item in checked if item["proposal_status"] == "approved_for_review"),
        "rejected": sum(1 for item in checked if item["proposal_status"] == "rejected"),
        "reduced_size": sum(1 for item in checked if item["proposal_status"] == "reduced_size"),
        "watch_only": sum(1 for item in checked if item["proposal_status"] == "watch_only"),
        "paper_trackable": sum(1 for item in checked if item.get("paper_tracking_status") == "track"),
        "paper_do_not_track": sum(1 for item in checked if item.get("paper_tracking_status") == "do_not_track"),
    }

    return {
        "generated_at": utc_now(),
        "mode": "mock_proposal_risk_check",
        "order_enabled": False,
        "proposal_only": True,
        "summary": summary,
        "limits": {
            "max_daily_new_buy_amount": daily_state["daily_buy_limit"],
            "max_order_amount_per_symbol": MAX_ORDER_AMOUNT_PER_SYMBOL,
            "max_single_symbol_weight": MAX_SINGLE_SYMBOL_WEIGHT,
            "max_sector_weight": MAX_SECTOR_WEIGHT,
        },
        "checked_proposals": checked,
    }


def build_markdown_report(result: Dict[str, Any]) -> str:
    summary = result.get("summary", {})
    lines = [
        "# 주문 후보 리스크 검사",
        "",
        f"- 생성 시각: {result.get('generated_at')}",
        "- 실제 주문 실행: 비활성화",
        "- order_enabled=false",
        "- proposal_only=true",
        "",
        "## 요약",
        f"- 전체 후보: {summary.get('total_proposals', 0)}",
        f"- 검토 가능: {summary.get('approved_for_review', 0)}",
        f"- 축소 필요: {summary.get('reduced_size', 0)}",
        f"- reject: {summary.get('rejected', 0)}",
        f"- 관망: {summary.get('watch_only', 0)}",
        f"- paper tracking 가능: {summary.get('paper_trackable', 0)}",
        f"- paper tracking 제외: {summary.get('paper_do_not_track', 0)}",
        "",
        "## 후보별 검사",
    ]

    for item in result.get("checked_proposals", []):
        lines.append(
            f"- {item.get('proposal_id')} / {item.get('stock_name')}({item.get('stock_code')}) "
            f"/ {item.get('side')} / status={item.get('proposal_status')} "
            f"/ paper_tracking={item.get('paper_tracking_status')} "
            f"/ after_guard={item.get('recommended_amount_after_guard'):,}원"
        )
        if item.get("risk_reasons"):
            lines.append(f"  - reject reasons: {', '.join(item.get('risk_reasons', []))}")
        if item.get("risk_warnings"):
            lines.append(f"  - warnings: {', '.join(item.get('risk_warnings', []))}")
        if item.get("paper_tracking_reason"):
            lines.append(f"  - paper tracking reason: {', '.join(item.get('paper_tracking_reason', [])[:6])}")
        if item.get("hard_reject_reason"):
            lines.append(f"  - hard reject: {', '.join(item.get('hard_reject_reason', []))}")

    lines.extend([
        "",
        "## 주의사항",
        "- 이 검사는 주문 후보 검토용이며 주문 API를 호출하지 않습니다.",
        "- kt10000/kt10001/kt10002/kt10003 주문 TR은 사용하지 않습니다.",
        "- Telegram 승인 기능은 아직 비활성화되어 있습니다.",
    ])

    return "\n".join(lines)


def save_risk_check(result: Dict[str, Any]) -> None:
    ORDER_RISK_CHECK_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    ORDER_RISK_CHECK_JSON_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    ORDER_RISK_CHECK_MD_PATH.write_text(build_markdown_report(result), encoding="utf-8")


def run_risk_guard(proposals_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if proposals_payload is None:
        proposals_payload = load_json(ORDER_PROPOSALS_JSON_PATH, {"proposals": []})
    result = check_proposals(proposals_payload)
    save_risk_check(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Risk-check order proposals without executing orders.")
    parser.add_argument("--proposals", default=str(ORDER_PROPOSALS_JSON_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    proposals_payload = load_json(args.proposals, {"proposals": []})
    result = check_proposals(proposals_payload)
    save_risk_check(result)
    print(f"Saved risk check json: {ORDER_RISK_CHECK_JSON_PATH}")
    print(f"Saved risk check report: {ORDER_RISK_CHECK_MD_PATH}")
    print(f"order_enabled={str(result.get('order_enabled')).lower()}")
    print(f"proposal_only={str(result.get('proposal_only')).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
