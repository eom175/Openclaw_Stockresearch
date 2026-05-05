#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime, timezone

from kiwoom_client import KiwoomRestClient


BASE_DIR = Path.home() / "policy-research"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    client = KiwoomRestClient()

    results = {}

    # 1. 예수금상세현황요청 kt00001
    # qry_tp: 3 = 추정조회, 2 = 일반조회
    # 공식/샘플 문서에서 kt00001의 필수 body는 qry_tp로 안내됨.
    results["deposit"] = client.post(
        endpoint="/api/dostk/acnt",
        api_id="kt00001",
        data={
            "qry_tp": "3"
        }
    )

    # 2. 계좌평가현황요청 kt00004
    # qry_tp: 0 = 전체, 1 = 상장폐지종목 제외
    # dmst_stex_tp: KRX 또는 NXT
    results["account_evaluation"] = client.post(
        endpoint="/api/dostk/acnt",
        api_id="kt00004",
        data={
            "qry_tp": "1",
            "dmst_stex_tp": "KRX"
        }
    )

    # 3. 계좌평가잔고내역요청 kt00018
    # 실제 보유 종목 리스트를 보기엔 kt00018이 더 보기 좋을 수 있음.
    results["holdings_detail"] = client.post(
        endpoint="/api/dostk/acnt",
        api_id="kt00018",
        data={
            "qry_tp": "1",
            "dmst_stex_tp": "KRX"
        }
    )

    # 4. 주문체결현황 kt00009
    # 모의주문 테스트 후 체결/미체결 확인용.
    results["order_status"] = client.post(
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

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "env": "mock",
        "results": results,
    }

    out_path = REPORTS_DIR / "account_snapshot_raw.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\nSaved account snapshot to {out_path}")


if __name__ == "__main__":
    main()
