#!/usr/bin/env python3
import argparse
import json
import math
from datetime import datetime, timezone

from policylink.kiwoom.client import KiwoomRestClient
from policylink.paths import REPORTS_DIR
from policylink.utils import parse_kiwoom_float, parse_kiwoom_int, pick_first


REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_stock_code(value):
    if value is None:
        return "-"

    text = str(value).strip()

    if text.startswith("A") and len(text) == 7:
        return text[1:]

    return text


def get_tick_size(price: int) -> int:
    """
    국내주식 일반 호가단위 근사.
    모의투자 테스트용이며, 실제 제도/ETF/ETN/시장구분에 따라 예외가 있을 수 있다.
    """
    if price < 2000:
        return 1
    if price < 5000:
        return 5
    if price < 20000:
        return 10
    if price < 50000:
        return 50
    if price < 200000:
        return 100
    if price < 500000:
        return 500
    return 1000


def round_down_to_tick(price: int) -> int:
    tick = get_tick_size(price)
    return int(math.floor(price / tick) * tick)


def fetch_stock_basic_info(client: KiwoomRestClient, stock_code: str):
    """
    ka10001 = 주식기본정보요청.
    """
    return client.post(
        endpoint="/api/dostk/stkinfo",
        api_id="ka10001",
        data={
            "stk_cd": stock_code
        }
    )


def parse_stock_price(info_result):
    body = info_result.get("body", {}) if isinstance(info_result, dict) else {}

    stock_name = pick_first(
        body,
        ["stk_nm", "stock_name", "name", "종목명"],
        "-"
    )

    current_price = parse_kiwoom_int(
        pick_first(
            body,
            ["cur_prc", "current_price", "현재가"],
            "0"
        )
    )

    upper_limit = parse_kiwoom_int(
        pick_first(
            body,
            ["upl_prc", "up_prc", "상한가"],
            "0"
        )
    )

    lower_limit = parse_kiwoom_int(
        pick_first(
            body,
            ["lst_prc", "down_prc", "하한가"],
            "0"
        )
    )

    fluctuation_rate = parse_kiwoom_float(
        pick_first(
            body,
            ["flu_rt", "fluctuation_rate", "등락율"],
            "0"
        )
    )

    if current_price <= 0:
        raise RuntimeError(f"현재가 파싱 실패: {body}")

    return {
        "stock_name": str(stock_name),
        "current_price": current_price,
        "upper_limit": upper_limit,
        "lower_limit": lower_limit,
        "fluctuation_rate": fluctuation_rate,
        "raw_body": body,
    }


def fetch_account_balance(client: KiwoomRestClient):
    """
    kt00018 = 계좌평가잔고내역요청.
    보유종목의 매도가능수량 확인용.
    """
    return client.post(
        endpoint="/api/dostk/acnt",
        api_id="kt00018",
        data={
            "qry_tp": "1",
            "dmst_stex_tp": "KRX"
        }
    )


def parse_tradable_quantity(balance_result, stock_code: str):
    body = balance_result.get("body", {}) if isinstance(balance_result, dict) else {}
    raw_items = body.get("acnt_evlt_remn_indv_tot", [])

    if not isinstance(raw_items, list):
        raw_items = []

    target_code = normalize_stock_code(stock_code)

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        item_code = normalize_stock_code(item.get("stk_cd"))

        if item_code != target_code:
            continue

        quantity = parse_kiwoom_int(item.get("rmnd_qty"))
        tradable_quantity = parse_kiwoom_int(item.get("trde_able_qty"))

        return {
            "found": True,
            "stock_code": item_code,
            "stock_name": str(item.get("stk_nm", "-")),
            "quantity": quantity,
            "tradable_quantity": tradable_quantity,
            "raw_item": item,
        }

    return {
        "found": False,
        "stock_code": target_code,
        "stock_name": "-",
        "quantity": 0,
        "tradable_quantity": 0,
        "raw_item": None,
    }


def calculate_marketable_sell_price(current_price: int, slippage_bps: int, lower_limit: int = 0):
    """
    current_price에서 slippage_bps만큼 낮춘 체결 우선 지정가.
    예: 50bps = 0.50%
    """
    buffered = int(current_price * (1 - slippage_bps / 10000))
    price = round_down_to_tick(buffered)

    if lower_limit > 0:
        price = max(price, lower_limit)

    return price


def place_sell_order(client: KiwoomRestClient, stock_code: str, qty: int, price: int):
    """
    kt10001 = 주식 매도주문.
    """
    return client.post(
        endpoint="/api/dostk/ordr",
        api_id="kt10001",
        data={
            "dmst_stex_tp": "KRX",
            "stk_cd": stock_code,
            "ord_qty": str(qty),
            "ord_uv": str(price),
            "trde_tp": "0"
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Kiwoom mock smart sell by current price")
    parser.add_argument("--stock-code", default="005930")
    parser.add_argument("--qty", type=int, default=1)
    parser.add_argument(
        "--slippage-bps",
        type=int,
        default=50,
        help="현재가 대비 낮추는 폭. 50 = 0.50%",
    )
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    if not args.yes:
        print("안전장치: 매도 주문을 실행하지 않았습니다.")
        print("실행하려면 --yes를 붙이세요.")
        return

    if args.qty != 1:
        raise RuntimeError("첫 테스트는 반드시 1주만 허용합니다.")

    if args.slippage_bps < 0 or args.slippage_bps > 300:
        raise RuntimeError("slippage-bps는 0~300 사이로 제한합니다.")

    client = KiwoomRestClient()

    balance_result = fetch_account_balance(client)
    balance_info = parse_tradable_quantity(balance_result, args.stock_code)

    if not balance_info["found"]:
        raise RuntimeError(f"보유종목에서 {args.stock_code}를 찾지 못했습니다.")

    if balance_info["tradable_quantity"] < args.qty:
        raise RuntimeError(
            f"매도가능수량 부족: 요청 {args.qty}주, 매도가능 {balance_info['tradable_quantity']}주"
        )

    info_result = fetch_stock_basic_info(client, args.stock_code)
    price_info = parse_stock_price(info_result)

    sell_price = calculate_marketable_sell_price(
        current_price=price_info["current_price"],
        slippage_bps=args.slippage_bps,
        lower_limit=price_info["lower_limit"],
    )

    order_result = place_sell_order(
        client=client,
        stock_code=args.stock_code,
        qty=args.qty,
        price=sell_price,
    )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "env": "mock",
        "strategy": "current_price_minus_buffer_limit_sell",
        "stock_code": args.stock_code,
        "stock_name": balance_info["stock_name"],
        "qty": args.qty,
        "current_price": price_info["current_price"],
        "upper_limit": price_info["upper_limit"],
        "lower_limit": price_info["lower_limit"],
        "fluctuation_rate": price_info["fluctuation_rate"],
        "slippage_bps": args.slippage_bps,
        "calculated_sell_price": sell_price,
        "balance_info": balance_info,
        "stock_info_result": info_result,
        "order_result": order_result,
        "note": "현재가 조회 후 현재가보다 약간 낮은 지정가로 모의 매도한 결과입니다."
    }

    out_path = REPORTS_DIR / "kiwoom_smart_sell_result.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Saved smart sell result to {out_path}")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
