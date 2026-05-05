#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone

from policylink.kiwoom.client import KiwoomRestClient
from policylink.paths import REPORTS_DIR


REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def place_buy_order(client: KiwoomRestClient, stock_code: str, qty: int, price: int):
    """
    kt10000 = 주식 매수주문
    첫 테스트는 반드시 모의투자 + 지정가 + 1주로만 실행한다.
    """
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


def place_sell_order(client: KiwoomRestClient, stock_code: str, qty: int, price: int):
    """
    kt10001 = 주식 매도주문
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
    parser = argparse.ArgumentParser(description="Kiwoom mock trading order test")
    parser.add_argument("--side", choices=["buy", "sell"], required=True)
    parser.add_argument("--stock-code", default="005930")
    parser.add_argument("--qty", type=int, default=1)
    parser.add_argument("--price", type=int, required=True)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    if not args.yes:
        print("안전장치: 주문을 실행하지 않았습니다.")
        print("실행하려면 --yes 옵션을 붙이세요.")
        return

    if args.qty != 1:
        raise RuntimeError("첫 모의주문 테스트는 반드시 1주만 허용합니다.")

    if args.price <= 0:
        raise RuntimeError("price는 0보다 커야 합니다.")

    client = KiwoomRestClient()

    if args.side == "buy":
        result = place_buy_order(
            client=client,
            stock_code=args.stock_code,
            qty=args.qty,
            price=args.price,
        )
    else:
        result = place_sell_order(
            client=client,
            stock_code=args.stock_code,
            qty=args.qty,
            price=args.price,
        )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "env": "mock",
        "warning": "모의투자 주문 테스트 결과입니다. 실거래 주문이 아닙니다.",
        "side": args.side,
        "stock_code": args.stock_code,
        "qty": args.qty,
        "price": args.price,
        "result": result,
    }

    out_path = REPORTS_DIR / "mock_order_result.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Saved mock order result to {out_path}")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
