#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone, timedelta

from policylink.kiwoom.client import KiwoomRestClient
from policylink.paths import REPORTS_DIR
from policylink.utils import find_first_list, normalize_code, parse_kiwoom_int, pick_first


REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_kst_today_yyyymmdd():
    kst = timezone(timedelta(hours=9))
    return datetime.now(kst).strftime("%Y%m%d")


def normalize_stock_code(value):
    return normalize_code(value)


def fetch_order_status(client: KiwoomRestClient):
    today = get_kst_today_yyyymmdd()

    return client.post(
        endpoint="/api/dostk/acnt",
        api_id="kt00009",
        data={
            "ord_dt": today,
            "stk_bond_tp": "0",
            "mrkt_tp": "0",
            "sell_tp": "0",
            "qry_tp": "0",
            "stk_cd": "",
            "fr_ord_no": "",
            "dmst_stex_tp": "KRX"
        }
    )


def normalize_order_items(body):
    candidate_keys = [
        "ord_cntr_stat",
        "ord_cntr",
        "order_list",
        "items",
        "output",
        "list",
    ]

    raw_items = []
    if isinstance(body, dict):
        for key in candidate_keys:
            value = body.get(key)
            if isinstance(value, list):
                raw_items = value
                break

    if not raw_items:
        raw_items = find_first_list(body)

    parsed = []

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        order_no = pick_first(
            item,
            ["ord_no", "odno", "order_no", "원주문번호", "주문번호"],
            ""
        )

        stock_code = normalize_stock_code(
            pick_first(
                item,
                ["stk_cd", "stock_code", "code", "종목코드"],
                ""
            )
        )

        stock_name = pick_first(
            item,
            ["stk_nm", "stock_name", "name", "종목명"],
            ""
        )

        side = pick_first(
            item,
            ["sell_buy_tp", "ord_tp", "side", "매매구분"],
            ""
        )

        order_qty = parse_kiwoom_int(
            pick_first(
                item,
                ["ord_qty", "order_qty", "주문수량"],
                "0"
            )
        )

        filled_qty = parse_kiwoom_int(
            pick_first(
                item,
                ["cntr_qty", "filled_qty", "체결수량"],
                "0"
            )
        )

        unfilled_qty = parse_kiwoom_int(
            pick_first(
                item,
                ["untr_qty", "unfilled_qty", "미체결수량", "rmn_qty", "remn_qty"],
                "0"
            )
        )

        order_price = parse_kiwoom_int(
            pick_first(
                item,
                ["ord_uv", "ord_prc", "order_price", "주문가격"],
                "0"
            )
        )

        status = pick_first(
            item,
            ["ord_stat", "status", "주문상태"],
            ""
        )

        parsed.append({
            "order_no": str(order_no),
            "stock_code": str(stock_code),
            "stock_name": str(stock_name),
            "side": str(side),
            "status": str(status),
            "order_qty": order_qty,
            "filled_qty": filled_qty,
            "unfilled_qty": unfilled_qty,
            "order_price": order_price,
            "raw_keys": sorted(list(item.keys())),
            "raw": item,
        })

    return parsed


def cancel_order(client: KiwoomRestClient, order_no: str, stock_code: str, qty: int):
    """
    kt10003 = 주식 취소주문.

    주의:
    키움 REST API의 취소주문 body 필드명은 환경/문서 버전에 따라
    orgn_ord_no 또는 orig_ord_no 계열일 수 있다.
    현재 코드는 일반적으로 쓰이는 원주문번호/종목코드/취소수량 구조로 작성했다.
    만약 필드명 오류가 나면 mock_order_result.json의 응답 메시지를 기준으로 필드명을 조정해야 한다.
    """
    return client.post(
        endpoint="/api/dostk/ordr",
        api_id="kt10003",
        data={
            "dmst_stex_tp": "KRX",
            "orig_ord_no": str(order_no),
            "stk_cd": str(stock_code),
            "cncl_qty": str(qty),
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Cancel Kiwoom mock pending orders")
    parser.add_argument("--stock-code", default="", help="특정 종목코드만 취소. 비우면 전체 후보")
    parser.add_argument("--order-no", default="", help="특정 주문번호만 취소")
    parser.add_argument("--yes", action="store_true", help="실제 취소 실행")
    args = parser.parse_args()

    if not args.yes:
        print("안전장치: 취소 주문을 실행하지 않았습니다. 실행하려면 --yes를 붙이세요.")
        return

    client = KiwoomRestClient()

    status_result = fetch_order_status(client)
    body = status_result.get("body", {}) if isinstance(status_result, dict) else {}
    orders = normalize_order_items(body)

    cancel_candidates = []
    for order in orders:
        if args.order_no and order["order_no"] != args.order_no:
            continue

        if args.stock_code and order["stock_code"] != args.stock_code:
            continue

        if order["unfilled_qty"] > 0 and order["order_no"]:
            cancel_candidates.append(order)

    cancel_results = []

    for order in cancel_candidates:
        result = cancel_order(
            client=client,
            order_no=order["order_no"],
            stock_code=order["stock_code"],
            qty=order["unfilled_qty"],
        )

        cancel_results.append({
            "target_order": order,
            "cancel_result": result,
        })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "env": "mock",
        "status_result": status_result,
        "parsed_orders": orders,
        "cancel_candidates": cancel_candidates,
        "cancel_results": cancel_results,
        "note": "모의투자 미체결 취소 결과입니다."
    }

    out_path = REPORTS_DIR / "kiwoom_cancel_pending_order_result.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Saved cancel result to {out_path}")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
