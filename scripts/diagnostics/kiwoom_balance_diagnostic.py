import json
from datetime import datetime, timezone

from _bootstrap import add_project_paths

add_project_paths()

from policylink.kiwoom.client import KiwoomRestClient
from policylink.paths import REPORTS_DIR


REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def find_lists(obj, path="$"):
    """
    응답 body 내부의 모든 list 위치를 찾는다.
    보유종목/체결잔고 리스트가 어떤 key에 들어오는지 모를 때 사용한다.
    """
    results = []

    if isinstance(obj, list):
        sample_keys = []
        if obj and isinstance(obj[0], dict):
            sample_keys = sorted(list(obj[0].keys()))

        results.append({
            "path": path,
            "length": len(obj),
            "sample_keys": sample_keys,
            "sample_first_item": obj[0] if obj else None,
        })

    elif isinstance(obj, dict):
        for key, value in obj.items():
            results.extend(find_lists(value, f"{path}.{key}"))

    return results


def compact_body_preview(body):
    """
    너무 큰 응답을 터미널에 전부 찍지 않고 핵심만 보기 위한 preview.
    """
    if not isinstance(body, dict):
        return body

    preview = {}

    for key, value in body.items():
        if isinstance(value, list):
            preview[key] = {
                "type": "list",
                "length": len(value),
                "sample_keys": sorted(list(value[0].keys())) if value and isinstance(value[0], dict) else [],
                "sample_first_item": value[0] if value else None,
            }
        elif isinstance(value, dict):
            preview[key] = {
                "type": "dict",
                "keys": sorted(list(value.keys()))[:50],
            }
        else:
            preview[key] = value

    return preview


def call_account_tr(client, api_id, data, label):
    result = client.post(
        endpoint="/api/dostk/acnt",
        api_id=api_id,
        data=data,
    )

    body = result.get("body", {}) if isinstance(result, dict) else {}

    return {
        "label": label,
        "api_id": api_id,
        "request_data": data,
        "status_code": result.get("status_code"),
        "headers": result.get("headers"),
        "return_code": body.get("return_code") if isinstance(body, dict) else None,
        "return_msg": body.get("return_msg") if isinstance(body, dict) else None,
        "body_keys": sorted(list(body.keys())) if isinstance(body, dict) else [],
        "list_locations": find_lists(body),
        "body_preview": compact_body_preview(body),
        "raw": result,
    }


def main():
    client = KiwoomRestClient()

    diagnostics = []

    # 현재 체결된 주문 기준으로 삼성전자 잔고가 어디서 나오는지 확인하기 위한 진단.
    # TR별 request body가 문서/환경에 따라 민감할 수 있어 여러 후보를 순차 테스트한다.
    test_cases = [
        {
            "label": "kt00005_empty_body",
            "api_id": "kt00005",
            "data": {},
        },
        {
            "label": "kt00005_krx_only",
            "api_id": "kt00005",
            "data": {
                "dmst_stex_tp": "KRX",
            },
        },
        {
            "label": "kt00005_qry_krx",
            "api_id": "kt00005",
            "data": {
                "qry_tp": "1",
                "dmst_stex_tp": "KRX",
            },
        },
        {
            "label": "kt00005_stock_krx",
            "api_id": "kt00005",
            "data": {
                "stk_cd": "005930",
                "dmst_stex_tp": "KRX",
            },
        },
        {
            "label": "kt00018_qry_krx",
            "api_id": "kt00018",
            "data": {
                "qry_tp": "1",
                "dmst_stex_tp": "KRX",
            },
        },
        {
            "label": "kt00009_order_status",
            "api_id": "kt00009",
            "data": {
                "ord_dt": "",
                "stk_bond_tp": "0",
                "mrkt_tp": "0",
                "sell_tp": "0",
                "qry_tp": "0",
                "stk_cd": "",
                "fr_ord_no": "",
                "dmst_stex_tp": "KRX",
            },
        },
    ]

    for case in test_cases:
        try:
            diagnostics.append(
                call_account_tr(
                    client=client,
                    api_id=case["api_id"],
                    data=case["data"],
                    label=case["label"],
                )
            )
        except Exception as e:
            diagnostics.append({
                "label": case["label"],
                "api_id": case["api_id"],
                "request_data": case["data"],
                "error": str(e),
            })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "env": "mock",
        "purpose": "Find which Kiwoom account TR returns actual filled stock balance after mock buy.",
        "diagnostics": diagnostics,
    }

    json_path = REPORTS_DIR / "kiwoom_balance_diagnostic.json"
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_lines = []
    md_lines.append("# 키움 잔고 조회 진단")
    md_lines.append("")
    md_lines.append(f"- 생성 시각 UTC: {output['generated_at']}")
    md_lines.append("- 목적: 모의매수 체결 후 삼성전자 1주 잔고가 어떤 TR/key로 내려오는지 확인")
    md_lines.append("")

    for item in diagnostics:
        md_lines.append(f"## {item.get('label')}")
        md_lines.append(f"- api_id: {item.get('api_id')}")
        md_lines.append(f"- request_data: `{json.dumps(item.get('request_data'), ensure_ascii=False)}`")

        if "error" in item:
            md_lines.append(f"- error: {item['error']}")
            md_lines.append("")
            continue

        md_lines.append(f"- HTTP status: {item.get('status_code')}")
        md_lines.append(f"- return_code: {item.get('return_code')}")
        md_lines.append(f"- return_msg: {item.get('return_msg')}")
        md_lines.append(f"- body_keys: {item.get('body_keys')}")

        list_locations = item.get("list_locations", [])
        if list_locations:
            md_lines.append("- list_locations:")
            for loc in list_locations:
                md_lines.append(
                    f"  - path={loc.get('path')} / length={loc.get('length')} / sample_keys={loc.get('sample_keys')}"
                )
        else:
            md_lines.append("- list_locations: 없음")

        md_lines.append("")

    md_path = REPORTS_DIR / "kiwoom_balance_diagnostic.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved diagnostic json: {json_path}")
    print(f"Saved diagnostic report: {md_path}")
    print("")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
