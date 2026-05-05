from typing import Any, Dict, Optional

from policylink.kiwoom.client import KiwoomRestClient
from policylink.utils import normalize_code, parse_kiwoom_float, parse_kiwoom_int, pick_first


def get_tick_size(price: int) -> int:
    if price < 2_000:
        return 1
    if price < 5_000:
        return 5
    if price < 20_000:
        return 10
    if price < 50_000:
        return 50
    if price < 200_000:
        return 100
    if price < 500_000:
        return 500
    return 1_000


def round_up_to_tick(price: int) -> int:
    tick = get_tick_size(price)
    return ((price + tick - 1) // tick) * tick


def round_down_to_tick(price: int) -> int:
    tick = get_tick_size(price)
    return (price // tick) * tick


def fetch_stock_basic_info(client: KiwoomRestClient, stock_code: str) -> Dict[str, Any]:
    return client.post(
        endpoint="/api/dostk/stkinfo",
        api_id="ka10001",
        data={"stk_cd": normalize_code(stock_code)},
    )


def parse_stock_price(info_result: Dict[str, Any]) -> Dict[str, Any]:
    body = info_result.get("body", {}) if isinstance(info_result, dict) else {}
    if not isinstance(body, dict):
        body = {}

    current_price = parse_kiwoom_int(
        pick_first(body, ["cur_prc", "현재가", "stck_prpr", "price"]),
        0,
    )
    upper_limit = parse_kiwoom_int(
        pick_first(body, ["upl_prc", "상한가", "upper_limit"]),
        0,
    )
    lower_limit = parse_kiwoom_int(
        pick_first(body, ["lst_prc", "하한가", "lower_limit"]),
        0,
    )
    change_rate = parse_kiwoom_float(
        pick_first(body, ["flu_rt", "등락율", "change_rate"]),
        0.0,
    )

    return {
        "stock_code": normalize_code(pick_first(body, ["stk_cd", "종목코드"], "")),
        "stock_name": pick_first(body, ["stk_nm", "종목명"], ""),
        "current_price": abs(current_price),
        "upper_limit": abs(upper_limit),
        "lower_limit": abs(lower_limit),
        "change_rate": change_rate,
    }


def calculate_marketable_buy_price(current_price: int, slippage_bps: int, upper_limit: int = 0) -> int:
    raw_price = int(current_price * (1 + slippage_bps / 10_000))
    price = round_up_to_tick(raw_price)
    if upper_limit > 0:
        price = min(price, upper_limit)
    return price


def calculate_marketable_sell_price(current_price: int, slippage_bps: int, lower_limit: int = 0) -> int:
    raw_price = int(current_price * (1 - slippage_bps / 10_000))
    price = round_down_to_tick(raw_price)
    if lower_limit > 0:
        price = max(price, lower_limit)
    return price


def place_buy_order(client: KiwoomRestClient, stock_code: str, qty: int, price: int) -> Dict[str, Any]:
    return client.post(
        endpoint="/api/dostk/ordr",
        api_id="kt10000",
        data={
            "dmst_stex_tp": "KRX",
            "stk_cd": normalize_code(stock_code),
            "ord_qty": str(qty),
            "ord_uv": str(price),
            "trde_tp": "0",
        },
    )


def place_sell_order(client: KiwoomRestClient, stock_code: str, qty: int, price: int) -> Dict[str, Any]:
    return client.post(
        endpoint="/api/dostk/ordr",
        api_id="kt10001",
        data={
            "dmst_stex_tp": "KRX",
            "stk_cd": normalize_code(stock_code),
            "ord_qty": str(qty),
            "ord_uv": str(price),
            "trde_tp": "0",
        },
    )


def cancel_order(client: KiwoomRestClient, order_no: str, stock_code: str, qty: int, price: Optional[int] = None) -> Dict[str, Any]:
    data = {
        "dmst_stex_tp": "KRX",
        "orig_ord_no": order_no,
        "stk_cd": normalize_code(stock_code),
        "cncl_qty": str(qty),
    }
    if price is not None:
        data["ord_uv"] = str(price)

    return client.post(
        endpoint="/api/dostk/ordr",
        api_id="kt10003",
        data=data,
    )
