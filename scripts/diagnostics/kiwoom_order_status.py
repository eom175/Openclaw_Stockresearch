import json

from _bootstrap import add_project_paths

add_project_paths()

from policylink.kiwoom.client import KiwoomRestClient
from policylink.paths import REPORTS_DIR
from policylink.utils import find_first_list, get_kst_today_yyyymmdd, parse_kiwoom_int, pick_first


REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_order_items(body):
    candidate_keys = [
        "ord_cntr_stat",
        "ord_cntr",
        "order_list",
        "items",
        "output",
        "list",
    ]

    for key in candidate_keys:
        value = body.get(key) if isinstance(body, dict) else None
        if isinstance(value, list):
            raw_items = value
            break
    else:
        raw_items = find_first_list(body)

    parsed = []

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        order_no = pick_first(
            item,
            ["ord_no", "order_no", "주문번호"],
            "-"
        )

        stock_code = pick_first(
            item,
            ["stk_cd", "stock_code", "code", "종목코드"],
            "-"
        )

        stock_name = pick_first(
            item,
            ["stk_nm", "stock_name", "name", "종목명"],
            "-"
        )

        side = pick_first(
            item,
            ["sell_buy_tp", "ord_tp", "side", "매매구분"],
            "-"
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
                ["untr_qty", "unfilled_qty", "미체결수량"],
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

        filled_price = parse_kiwoom_int(
            pick_first(
                item,
                ["cntr_prc", "filled_price", "체결가격"],
                "0"
            )
        )

        status = pick_first(
            item,
            ["ord_stat", "status", "주문상태"],
            "-"
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
            "filled_price": filled_price,
            "raw_keys": sorted(list(item.keys())),
        })

    return parsed


def fetch_order_status():
    client = KiwoomRestClient()

    today = get_kst_today_yyyymmdd()

    result = client.post(
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

    return result


def build_markdown(result, parsed_orders):
    body = result.get("body", {}) if isinstance(result, dict) else {}

    lines = []
    lines.append("# 키움 모의투자 주문체결현황")
    lines.append("")
    lines.append(f"- 생성 시각 UTC: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- HTTP status: {result.get('status_code')}")
    lines.append(f"- return_code: {body.get('return_code')}")
    lines.append(f"- return_msg: {body.get('return_msg')}")
    lines.append("")

    lines.append("## 주문 목록")
    if parsed_orders:
        for i, item in enumerate(parsed_orders, 1):
            lines.append(
                f"{i}. {item['stock_name']}({item['stock_code']}) "
                f"/ 주문번호: {item['order_no']} "
                f"/ 구분: {item['side']} "
                f"/ 상태: {item['status']} "
                f"/ 주문수량: {item['order_qty']:,} "
                f"/ 체결수량: {item['filled_qty']:,} "
                f"/ 미체결수량: {item['unfilled_qty']:,} "
                f"/ 주문가: {item['order_price']:,} "
                f"/ 체결가: {item['filled_price']:,}"
            )
    else:
        lines.append("- 파싱된 주문 목록이 없습니다.")
        lines.append("- 주문이 없거나, 주문체결현황 응답 key를 추가 확인해야 할 수 있습니다.")

    lines.append("")
    lines.append("## 해석")
    lines.append("- 주문가능금액이 줄었는데 보유종목이 없다면, 미체결 주문으로 금액이 묶였을 가능성이 있습니다.")
    lines.append("- 체결수량이 0이고 미체결수량이 1 이상이면 아직 보유종목에 반영되지 않는 것이 정상입니다.")
    lines.append("- 체결 완료 후 다시 계좌 요약을 실행하면 보유종목에 반영되어야 합니다.")

    return "\n".join(lines)


def main():
    result = fetch_order_status()
    body = result.get("body", {}) if isinstance(result, dict) else {}
    parsed_orders = normalize_order_items(body)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "env": "mock",
        "raw": result,
        "parsed_orders": parsed_orders,
    }

    json_path = REPORTS_DIR / "kiwoom_order_status.json"
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    markdown = build_markdown(result, parsed_orders)

    md_path = REPORTS_DIR / "kiwoom_order_status.md"
    md_path.write_text(markdown, encoding="utf-8")

    print(f"Saved order status json: {json_path}")
    print(f"Saved order status report: {md_path}")
    print("")
    print(markdown)


if __name__ == "__main__":
    main()
