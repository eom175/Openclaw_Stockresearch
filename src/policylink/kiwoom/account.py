import json
from datetime import datetime, timezone

from policylink.kiwoom.client import KiwoomRestClient
from policylink.paths import (
    KIWOOM_ACCOUNT_RAW_PATH,
    KIWOOM_ACCOUNT_REPORT_PATH,
    KIWOOM_ACCOUNT_SUMMARY_PATH,
    REPORTS_DIR,
)
from policylink.utils import (
    get_return_status,
    normalize_code,
    parse_kiwoom_float,
    parse_kiwoom_int,
    pick_first,
    safe_body,
)


REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_body(result):
    return safe_body(result)


def normalize_stock_code(value):
    return normalize_code(value, empty="-")


def fetch_account_data():
    client = KiwoomRestClient()

    deposit = client.post(
        endpoint="/api/dostk/acnt",
        api_id="kt00001",
        data={
            "qry_tp": "3"
        }
    )

    evaluation_balance = client.post(
        endpoint="/api/dostk/acnt",
        api_id="kt00018",
        data={
            "qry_tp": "1",
            "dmst_stex_tp": "KRX"
        }
    )

    order_status = client.post(
        endpoint="/api/dostk/acnt",
        api_id="kt00009",
        data={
            "ord_dt": "",
            "stk_bond_tp": "0",
            "mrkt_tp": "0",
            "sell_tp": "0",
            "qry_tp": "0",
            "stk_cd": "",
            "fr_ord_no": "",
            "dmst_stex_tp": "KRX"
        }
    )

    return deposit, evaluation_balance, order_status


def parse_deposit(deposit_body):
    cash = parse_kiwoom_int(
        pick_first(deposit_body, ["entr"], "0")
    )

    withdrawable_amount = parse_kiwoom_int(
        pick_first(deposit_body, ["pymn_alow_amt"], "0")
    )

    orderable_amount = parse_kiwoom_int(
        pick_first(deposit_body, ["ord_alow_amt"], "0")
    )

    orderable_100_amount = parse_kiwoom_int(
        pick_first(deposit_body, ["100stk_ord_alow_amt"], orderable_amount)
    )

    d1_cash = parse_kiwoom_int(
        pick_first(deposit_body, ["d1_entra"], cash)
    )

    d2_cash = parse_kiwoom_int(
        pick_first(deposit_body, ["d2_entra"], cash)
    )

    pending_or_reserved = max(0, cash - orderable_100_amount)

    return {
        "cash": cash,
        "withdrawable_amount": withdrawable_amount,
        "orderable_amount": orderable_amount,
        "orderable_100_amount": orderable_100_amount,
        "d1_cash": d1_cash,
        "d2_cash": d2_cash,
        "pending_or_reserved": pending_or_reserved,
    }


def parse_holdings_from_kt00018(balance_body):
    raw_items = balance_body.get("acnt_evlt_remn_indv_tot", [])

    if not isinstance(raw_items, list):
        raw_items = []

    holdings = []

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        stock_code = normalize_stock_code(item.get("stk_cd"))
        stock_name = str(item.get("stk_nm", "이름 확인 필요"))

        quantity = parse_kiwoom_int(item.get("rmnd_qty"))
        tradable_quantity = parse_kiwoom_int(item.get("trde_able_qty"))

        current_price = parse_kiwoom_int(item.get("cur_prc"))
        purchase_price = parse_kiwoom_int(item.get("pur_pric"))

        purchase_amount = parse_kiwoom_int(item.get("pur_amt"))
        evaluation_amount = parse_kiwoom_int(item.get("evlt_amt"))
        pnl = parse_kiwoom_int(item.get("evltv_prft"))
        return_rate = parse_kiwoom_float(item.get("prft_rt"))

        today_buy_quantity = parse_kiwoom_int(item.get("tdy_buyq"))
        today_sell_quantity = parse_kiwoom_int(item.get("tdy_sellq"))

        commission_sum = parse_kiwoom_int(item.get("sum_cmsn"))
        tax = parse_kiwoom_int(item.get("tax"))

        holdings.append({
            "stock_code": stock_code,
            "stock_name": stock_name,
            "quantity": quantity,
            "tradable_quantity": tradable_quantity,
            "current_price": current_price,
            "purchase_price": purchase_price,
            "purchase_amount": purchase_amount,
            "evaluation_amount": evaluation_amount,
            "pnl": pnl,
            "return_rate": return_rate,
            "today_buy_quantity": today_buy_quantity,
            "today_sell_quantity": today_sell_quantity,
            "commission_sum": commission_sum,
            "tax": tax,
            "raw_keys": sorted(list(item.keys())),
        })

    return holdings


def parse_evaluation_summary_from_kt00018(balance_body):
    return {
        "estimated_deposit_asset_amount": parse_kiwoom_int(
            balance_body.get("prsm_dpst_aset_amt")
        ),
        "total_purchase_amount": parse_kiwoom_int(
            balance_body.get("tot_pur_amt")
        ),
        "total_evaluation_amount": parse_kiwoom_int(
            balance_body.get("tot_evlt_amt")
        ),
        "total_pnl": parse_kiwoom_int(
            balance_body.get("tot_evlt_pl")
        ),
        "total_return_rate": parse_kiwoom_float(
            balance_body.get("tot_prft_rt")
        ),
        "total_loan_amount": parse_kiwoom_int(
            balance_body.get("tot_loan_amt")
        ),
        "total_credit_loan_amount": parse_kiwoom_int(
            balance_body.get("tot_crd_loan_amt")
        ),
    }


def parse_order_status(order_body):
    raw_items = order_body.get("acnt_ord_cntr_prst_array", [])

    if not isinstance(raw_items, list):
        raw_items = []

    parsed_orders = []

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        order_no = str(
            pick_first(item, ["ord_no", "odno", "order_no", "주문번호"], "-")
        )

        stock_code = normalize_stock_code(
            pick_first(item, ["stk_cd", "stock_code", "code", "종목코드"], "-")
        )

        stock_name = str(
            pick_first(item, ["stk_nm", "stock_name", "name", "종목명"], "-")
        )

        order_quantity = parse_kiwoom_int(
            pick_first(item, ["ord_qty", "order_qty", "주문수량"], "0")
        )

        filled_quantity = parse_kiwoom_int(
            pick_first(item, ["cntr_qty", "filled_qty", "체결수량"], "0")
        )

        unfilled_quantity = parse_kiwoom_int(
            pick_first(item, ["untr_qty", "unfilled_qty", "미체결수량"], "0")
        )

        order_price = parse_kiwoom_int(
            pick_first(item, ["ord_uv", "ord_prc", "order_price", "주문가격"], "0")
        )

        filled_price = parse_kiwoom_int(
            pick_first(
                item,
                ["cntr_pric", "cntr_prc", "cntr_uv", "filled_price", "체결가", "체결단가"],
                "0"
            )
        )

        parsed_orders.append({
            "order_no": order_no,
            "stock_code": stock_code,
            "stock_name": stock_name,
            "order_quantity": order_quantity,
            "filled_quantity": filled_quantity,
            "unfilled_quantity": unfilled_quantity,
            "order_price": order_price,
            "filled_price": filled_price,
            "raw_keys": sorted(list(item.keys())),
        })

    pending_orders = [
        order for order in parsed_orders
        if order["unfilled_quantity"] > 0
    ]

    return {
        "orders": parsed_orders,
        "pending_orders": pending_orders,
        "pending_order_count": len(pending_orders),
    }


def build_summary(deposit, evaluation_balance, order_status):
    deposit_body = get_body(deposit)
    balance_body = get_body(evaluation_balance)
    order_body = get_body(order_status)

    deposit_summary = parse_deposit(deposit_body)
    evaluation_summary = parse_evaluation_summary_from_kt00018(balance_body)
    holdings = parse_holdings_from_kt00018(balance_body)
    order_summary = parse_order_status(order_body)

    holdings_evaluation_amount = sum(
        item["evaluation_amount"] for item in holdings
    )

    holdings_purchase_amount = sum(
        item["purchase_amount"] for item in holdings
    )

    holdings_pnl = sum(
        item["pnl"] for item in holdings
    )

    total_equity_candidate = evaluation_summary["estimated_deposit_asset_amount"]

    if total_equity_candidate <= 0:
        total_equity_candidate = deposit_summary["cash"] + holdings_evaluation_amount

    if total_equity_candidate <= 0:
        total_equity_candidate = deposit_summary["orderable_100_amount"] + holdings_evaluation_amount

    available_cash = deposit_summary["orderable_100_amount"]

    cash_weight = (
        available_cash / total_equity_candidate
        if total_equity_candidate > 0
        else 1.0
    )

    invested_weight = (
        holdings_evaluation_amount / total_equity_candidate
        if total_equity_candidate > 0
        else 0.0
    )

    for item in holdings:
        item["weight"] = (
            round(item["evaluation_amount"] / total_equity_candidate, 4)
            if total_equity_candidate > 0
            else 0.0
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "env": "mock",
        "api_status": {
            "deposit": get_return_status(deposit),
            "evaluation_balance": get_return_status(evaluation_balance),
            "order_status": get_return_status(order_status),
        },
        "cash": deposit_summary["cash"],
        "withdrawable_amount": deposit_summary["withdrawable_amount"],
        "orderable_amount": deposit_summary["orderable_amount"],
        "orderable_100_amount": deposit_summary["orderable_100_amount"],
        "pending_or_reserved_amount": deposit_summary["pending_or_reserved"],
        "d1_cash": deposit_summary["d1_cash"],
        "d2_cash": deposit_summary["d2_cash"],
        "estimated_total_equity": total_equity_candidate,
        "available_cash": available_cash,
        "cash_weight": round(cash_weight, 4),
        "invested_weight": round(invested_weight, 4),
        "holding_count": len(holdings),
        "holdings_evaluation_amount": holdings_evaluation_amount,
        "holdings_purchase_amount": holdings_purchase_amount,
        "holdings_pnl": holdings_pnl,
        "evaluation_summary": evaluation_summary,
        "holdings": holdings,
        "orders": order_summary["orders"],
        "pending_orders": order_summary["pending_orders"],
        "pending_order_count": order_summary["pending_order_count"],
        "note": "모의투자 계좌 조회용 요약입니다. 실제 주문은 실행하지 않았습니다.",
    }


def build_markdown_report(summary):
    lines = []

    lines.append("# 키움 모의투자 계좌 요약")
    lines.append("")
    lines.append(f"- 생성 시각 UTC: {summary['generated_at']}")
    lines.append("- 환경: 모의투자")
    lines.append("- 주문 실행 여부: 없음")
    lines.append("")

    lines.append("## 1. 현금 / 주문가능금액")
    lines.append(f"- 예수금: {summary['cash']:,}원")
    lines.append(f"- 출금가능금액: {summary['withdrawable_amount']:,}원")
    lines.append(f"- 주문가능금액: {summary['orderable_amount']:,}원")
    lines.append(f"- 100% 증거금 기준 주문가능금액: {summary['orderable_100_amount']:,}원")
    lines.append(f"- 주문대기/미체결로 묶인 금액 추정: {summary['pending_or_reserved_amount']:,}원")
    lines.append(f"- D+1 예수금: {summary['d1_cash']:,}원")
    lines.append(f"- D+2 예수금: {summary['d2_cash']:,}원")
    lines.append("")

    lines.append("## 2. 자산 요약")
    lines.append(f"- 추정 총자산: {summary['estimated_total_equity']:,}원")
    lines.append(f"- 사용 가능 현금: {summary['available_cash']:,}원")
    lines.append(f"- 보유종목 평가금액: {summary['holdings_evaluation_amount']:,}원")
    lines.append(f"- 보유종목 매입금액: {summary['holdings_purchase_amount']:,}원")
    lines.append(f"- 보유종목 평가손익: {summary['holdings_pnl']:,}원")
    lines.append(f"- 현금 비중: {summary['cash_weight'] * 100:.2f}%")
    lines.append(f"- 투자 비중: {summary['invested_weight'] * 100:.2f}%")
    lines.append(f"- 보유 종목 수: {summary['holding_count']}개")
    lines.append(f"- 계좌 총수익률 후보: {summary['evaluation_summary']['total_return_rate']:.2f}%")
    lines.append("")

    lines.append("## 3. 보유 종목")
    if summary["holdings"]:
        for i, item in enumerate(summary["holdings"], 1):
            lines.append(
                f"{i}. {item['stock_name']}({item['stock_code']}) "
                f"/ 수량: {item['quantity']:,}주 "
                f"/ 매도가능: {item['tradable_quantity']:,}주 "
                f"/ 현재가: {item['current_price']:,}원 "
                f"/ 매입가: {item['purchase_price']:,}원 "
                f"/ 평가금액: {item['evaluation_amount']:,}원 "
                f"/ 손익: {item['pnl']:,}원 "
                f"/ 수익률: {item['return_rate']:.2f}% "
                f"/ 비중: {item['weight'] * 100:.2f}%"
            )
    else:
        lines.append("- 현재 파싱된 보유 종목이 없습니다.")
    lines.append("")

    lines.append("## 4. 주문 / 미체결 상태")
    lines.append(f"- 주문 목록 수: {len(summary['orders'])}개")
    lines.append(f"- 미체결 주문 수: {summary['pending_order_count']}개")

    if summary["pending_orders"]:
        for i, order in enumerate(summary["pending_orders"], 1):
            lines.append(
                f"{i}. {order['stock_name']}({order['stock_code']}) "
                f"/ 주문번호: {order['order_no']} "
                f"/ 주문수량: {order['order_quantity']:,}주 "
                f"/ 체결수량: {order['filled_quantity']:,}주 "
                f"/ 미체결수량: {order['unfilled_quantity']:,}주 "
                f"/ 주문가: {order['order_price']:,}원"
            )
    else:
        lines.append("- 현재 파싱된 미체결 주문은 없습니다.")
    lines.append("")

    lines.append("## 5. API 상태")
    for name, status in summary["api_status"].items():
        lines.append(
            f"- {name}: HTTP {status.get('status_code')} "
            f"/ return_code={status.get('return_code')} "
            f"/ message={status.get('return_msg')}"
        )
    lines.append("")

    lines.append("## 6. 주의")
    lines.append("- 이 보고서는 키움 모의투자 조회 결과입니다.")
    lines.append("- 이 스크립트는 주문을 실행하지 않습니다.")
    lines.append("- API Key, Secret, Access Token은 보고서에 저장하지 않습니다.")
    lines.append("- 실제 투자 판단 전에는 키움 HTS/MTS의 계좌 화면과 대조하세요.")

    return "\n".join(lines)


def main():
    deposit, evaluation_balance, order_status = fetch_account_data()

    raw = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "env": "mock",
        "deposit": deposit,
        "evaluation_balance": evaluation_balance,
        "order_status": order_status,
    }

    raw_path = KIWOOM_ACCOUNT_RAW_PATH
    raw_path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = build_summary(
        deposit=deposit,
        evaluation_balance=evaluation_balance,
        order_status=order_status,
    )

    summary_path = KIWOOM_ACCOUNT_SUMMARY_PATH
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown = build_markdown_report(summary)

    report_path = KIWOOM_ACCOUNT_REPORT_PATH
    report_path.write_text(markdown, encoding="utf-8")

    print(f"Saved raw: {raw_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved report: {report_path}")
    print("")
    print(markdown)


if __name__ == "__main__":
    main()
