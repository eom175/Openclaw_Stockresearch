#!/usr/bin/env python3
import argparse
import json
import math
from datetime import datetime, timezone

from policylink.kiwoom.client import KiwoomRestClient
from policylink.paths import REPORTS_DIR
from policylink.utils import parse_kiwoom_float, parse_kiwoom_int, pick_first


REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_tick_size(price: int) -> int:
    """
    국내주식 일반 호가단위 근사.
    시장 제도 변경/ETF/ETN/저유동성 종목에 따라 예외가 있을 수 있으므로
    모의투자 테스트용으로 사용한다.
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


def round_up_to_tick(price: int) -> int:
    tick = get_tick_size(price)
    return int(math.ceil(price / tick) * tick)


def fetch_stock_basic_info(client: KiwoomRestClient, stock_code: str):
    """
    ka10001 = 주식기본정보요청.
    URL은 /api/dostk/stkinfo.
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


def calculate_marketable_buy_price(current_price: int, slippage_bps: int, upper_limit: int = 0):
    """
    current_price에 slippage_bps만큼 여유를 둔 체결 우선 지정가.
    예: 50bps = 0.50%
    """
    buffered = int(current_price * (1 + slippage_bps / 10000))
    price = round_up_to_tick(buffered)

    if upper_limit > 0:
        price = min(price, upper_limit)

    return price


def place_buy_order(client: KiwoomRestClient, stock_code: str, qty: int, price: int):
    return client.post(
        endpoint="/api/dostk/ordr",
        api_id="kt10000",
        data={
            "dmst_stex_tp": "KRX",
            "stk_cd": stock_code,
            "ord_qty": str(qty),
            "ord_uv": str(price),
            "trde_tp": "0"
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Kiwoom mock smart buy by current price")
    parser.add_argument("--stock-code", default="005930")
    parser.add_argument("--qty", type=int, default=1)
    parser.add_argument(
        "--slippage-bps",
        type=int,
        default=50,
        help="현재가 대비 여유폭. 50 = 0.50%",
    )
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    if not args.yes:
        print("안전장치: 매수 주문을 실행하지 않았습니다.")
        print("실행하려면 --yes를 붙이세요.")
        return

    if args.qty != 1:
        raise RuntimeError("첫 테스트는 반드시 1주만 허용합니다.")

    if args.slippage_bps < 0 or args.slippage_bps > 300:
        raise RuntimeError("slippage-bps는 0~300 사이로 제한합니다.")

    client = KiwoomRestClient()

    info_result = fetch_stock_basic_info(client, args.stock_code)
    price_info = parse_stock_price(info_result)

    buy_price = calculate_marketable_buy_price(
        current_price=price_info["current_price"],
        slippage_bps=args.slippage_bps,
        upper_limit=price_info["upper_limit"],
    )

    order_result = place_buy_order(
        client=client,
        stock_code=args.stock_code,
        qty=args.qty,
        price=buy_price,
    )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "env": "mock",
        "strategy": "current_price_plus_buffer_limit_buy",
        "stock_code": args.stock_code,
        "stock_name": price_info["stock_name"],
        "qty": args.qty,
        "current_price": price_info["current_price"],
        "upper_limit": price_info["upper_limit"],
        "lower_limit": price_info["lower_limit"],
        "fluctuation_rate": price_info["fluctuation_rate"],
        "slippage_bps": args.slippage_bps,
        "calculated_buy_price": buy_price,
        "stock_info_result": info_result,
        "order_result": order_result,
        "note": "현재가 조회 후 현재가보다 약간 높은 지정가로 모의 매수한 결과입니다."
    }

    out_path = REPORTS_DIR / "kiwoom_smart_buy_result.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Saved smart buy result to {out_path}")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
