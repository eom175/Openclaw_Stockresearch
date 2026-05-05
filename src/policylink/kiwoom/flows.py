import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from policylink.kiwoom.client import KiwoomRestClient
from policylink.paths import (
    DATA_DIR,
    FLOW_FEATURES_PATH,
    FLOW_FEATURES_REPORT_PATH,
    FLOW_SYNC_DIAGNOSTIC_PATH,
    FLOWS_DAILY_PATH,
    REPORTS_DIR,
)
from policylink.universe import universe_for_market_data
from policylink.utils import (
    find_first_list,
    get_return_status,
    normalize_code,
    parse_kiwoom_float,
    parse_kiwoom_int,
    pick_first,
    safe_body,
)


DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


FLOW_DIAGNOSTIC_PATH = FLOW_SYNC_DIAGNOSTIC_PATH
FLOW_REPORT_PATH = FLOW_FEATURES_REPORT_PATH


DEFAULT_UNIVERSE = universe_for_market_data()


def normalize_stock_code(value) -> str:
    return normalize_code(value)


def fetch_foreign_trading_trend(client: KiwoomRestClient, stock_code: str) -> Dict[str, Any]:
    """
    ka10008 = 주식외국인종목별매매동향.
    endpoint: /api/dostk/frgnistt
    required body: stk_cd
    """
    return client.post(
        endpoint="/api/dostk/frgnistt",
        api_id="ka10008",
        data={
            "stk_cd": stock_code,
        },
    )


def fetch_institutional_stock(client: KiwoomRestClient, stock_code: str) -> Dict[str, Any]:
    """
    ka10009 = 주식기관요청.
    공식 TR 목록상 기관/외국인 카테고리에 있음.
    body 필드는 ka10008과 유사하게 stk_cd를 우선 사용한다.
    응답 key가 다르면 flow_sync_diagnostic.json에서 확인해 parser를 조정한다.
    """
    return client.post(
        endpoint="/api/dostk/frgnistt",
        api_id="ka10009",
        data={
            "stk_cd": stock_code,
        },
    )


def extract_foreign_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    body = safe_body(result)

    rows = body.get("stk_frgnr")

    if isinstance(rows, list):
        return rows

    rows = body.get("foreign_trading_trend")

    if isinstance(rows, list):
        return rows

    rows = body.get("output")

    if isinstance(rows, list):
        return rows

    return find_first_list(body)


def extract_institution_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    body = safe_body(result)

    candidate_keys = [
        "stk_orgn",
        "stk_inst",
        "institution",
        "institutional_stock",
        "output",
        "items",
        "list",
    ]

    for key in candidate_keys:
        value = body.get(key)

        if isinstance(value, list):
            return value

    return find_first_list(body)


def normalize_foreign_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict):
        return None

    date = pick_first(row, ["dt", "date", "trd_dt", "일자"], None)

    close_price = parse_kiwoom_int(
        pick_first(row, ["close_pric", "close_prc", "close", "종가"], None),
        0,
    )

    trading_volume = parse_kiwoom_int(
        pick_first(row, ["trde_qty", "volume", "거래량"], None),
        0,
    )

    # ka10008 공식 응답의 chg_qty는 외국인 변동수량으로 쓰인다.
    foreign_net_qty = parse_kiwoom_int(
        pick_first(
            row,
            [
                "chg_qty",
                "change_qty",
                "frgnr_net_qty",
                "foreign_net_qty",
                "순매수수량",
                "변동수량",
            ],
            0,
        ),
        0,
    )

    foreign_holding_qty = parse_kiwoom_int(
        pick_first(
            row,
            ["poss_stkcnt", "holding_qty", "foreign_holding_qty", "보유주식수"],
            0,
        ),
        0,
    )

    foreign_weight = parse_kiwoom_float(
        pick_first(row, ["wght", "weight", "foreign_weight", "비중"], 0.0),
        0.0,
    )

    limit_exhaustion_rate = parse_kiwoom_float(
        pick_first(row, ["limit_exh_rt", "limit_exhaustion_rate", "한도소진률"], 0.0),
        0.0,
    )

    if not date:
        return None

    return {
        "date": str(date),
        "close_price": close_price,
        "trading_volume": trading_volume,
        "foreign_net_qty": foreign_net_qty,
        "foreign_holding_qty": foreign_holding_qty,
        "foreign_weight": foreign_weight,
        "foreign_limit_exhaustion_rate": limit_exhaustion_rate,
        "raw_keys": sorted(list(row.keys())),
    }


def infer_institution_net_qty(row: Dict[str, Any]) -> int:
    """
    ka10009 응답 필드명이 환경/버전에 따라 다를 수 있어,
    명시 후보를 먼저 보고, 안 잡히면 key 이름으로 추론한다.
    """
    explicit_value = pick_first(
        row,
        [
            "orgn_netprps",
            "inst_netprps",
            "institution_net_qty",
            "institution_net_buy_qty",
            "inst_net_qty",
            "기관순매수",
            "기관순매수수량",
            "순매수수량",
            "net_buy_qty",
        ],
        None,
    )

    if explicit_value is not None:
        return parse_kiwoom_int(explicit_value, 0)

    # fallback: key 이름에 기관/순매수/수량 느낌이 있는 숫자값을 탐색
    priority_keywords = [
        "orgn",
        "inst",
        "institution",
        "기관",
    ]

    net_keywords = [
        "net",
        "순",
        "chg",
        "변동",
    ]

    qty_keywords = [
        "qty",
        "q",
        "수량",
    ]

    for key, value in row.items():
        key_lower = str(key).lower()

        if (
            any(word in key_lower for word in priority_keywords)
            and any(word in key_lower for word in net_keywords)
            and any(word in key_lower for word in qty_keywords)
        ):
            return parse_kiwoom_int(value, 0)

    return 0


def normalize_institution_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict):
        return None

    date = pick_first(row, ["dt", "date", "trd_dt", "일자"], None)

    close_price = parse_kiwoom_int(
        pick_first(row, ["close_pric", "close_prc", "close", "종가"], None),
        0,
    )

    trading_volume = parse_kiwoom_int(
        pick_first(row, ["trde_qty", "volume", "거래량"], None),
        0,
    )

    institution_net_qty = infer_institution_net_qty(row)

    if not date:
        return None

    return {
        "date": str(date),
        "close_price": close_price,
        "trading_volume": trading_volume,
        "institution_net_qty": institution_net_qty,
        "raw_keys": sorted(list(row.keys())),
    }


def merge_flow_rows(
    stock_code: str,
    stock_name: str,
    sector: str,
    foreign_rows: List[Dict[str, Any]],
    institution_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_date: Dict[str, Dict[str, Any]] = {}

    for row in foreign_rows:
        normalized = normalize_foreign_row(row)

        if not normalized:
            continue

        date = normalized["date"]

        if date not in by_date:
            by_date[date] = {
                "date": date,
                "stock_code": stock_code,
                "stock_name": stock_name,
                "sector": sector,
                "close_price": normalized["close_price"],
                "trading_volume": normalized["trading_volume"],
                "foreign_net_qty": 0,
                "institution_net_qty": 0,
                "foreign_holding_qty": 0,
                "foreign_weight": 0.0,
                "foreign_limit_exhaustion_rate": 0.0,
            }

        by_date[date]["close_price"] = normalized["close_price"] or by_date[date]["close_price"]
        by_date[date]["trading_volume"] = normalized["trading_volume"] or by_date[date]["trading_volume"]
        by_date[date]["foreign_net_qty"] = normalized["foreign_net_qty"]
        by_date[date]["foreign_holding_qty"] = normalized["foreign_holding_qty"]
        by_date[date]["foreign_weight"] = normalized["foreign_weight"]
        by_date[date]["foreign_limit_exhaustion_rate"] = normalized["foreign_limit_exhaustion_rate"]

    for row in institution_rows:
        normalized = normalize_institution_row(row)

        if not normalized:
            continue

        date = normalized["date"]

        if date not in by_date:
            by_date[date] = {
                "date": date,
                "stock_code": stock_code,
                "stock_name": stock_name,
                "sector": sector,
                "close_price": normalized["close_price"],
                "trading_volume": normalized["trading_volume"],
                "foreign_net_qty": 0,
                "institution_net_qty": 0,
                "foreign_holding_qty": 0,
                "foreign_weight": 0.0,
                "foreign_limit_exhaustion_rate": 0.0,
            }

        by_date[date]["institution_net_qty"] = normalized["institution_net_qty"]

        if normalized["close_price"] > 0:
            by_date[date]["close_price"] = normalized["close_price"]

        if normalized["trading_volume"] > 0:
            by_date[date]["trading_volume"] = normalized["trading_volume"]

    rows = sorted(by_date.values(), key=lambda x: x["date"])

    return rows


def sum_last(values: List[int], window: int) -> int:
    if not values:
        return 0

    return int(sum(values[-window:]))


def avg_last(values: List[float], window: int) -> Optional[float]:
    if not values:
        return None

    sliced = values[-window:]

    if not sliced:
        return None

    return sum(sliced) / len(sliced)


def build_flow_feature(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None

    rows = sorted(rows, key=lambda x: x["date"])
    latest = rows[-1]

    foreign_net_values = [int(row.get("foreign_net_qty", 0)) for row in rows]
    institution_net_values = [int(row.get("institution_net_qty", 0)) for row in rows]
    volume_values = [float(row.get("trading_volume", 0)) for row in rows]
    close_values = [float(row.get("close_price", 0)) for row in rows]

    foreign_net_1d = foreign_net_values[-1] if foreign_net_values else 0
    institution_net_1d = institution_net_values[-1] if institution_net_values else 0

    foreign_net_5d = sum_last(foreign_net_values, 5)
    foreign_net_20d = sum_last(foreign_net_values, 20)

    institution_net_5d = sum_last(institution_net_values, 5)
    institution_net_20d = sum_last(institution_net_values, 20)

    combined_net_1d = foreign_net_1d + institution_net_1d
    combined_net_5d = foreign_net_5d + institution_net_5d
    combined_net_20d = foreign_net_20d + institution_net_20d

    avg_volume_20 = avg_last(volume_values, 20) or 0.0
    latest_volume = latest.get("trading_volume", 0) or 0

    combined_net_5d_to_volume = (
        combined_net_5d / avg_volume_20
        if avg_volume_20 > 0
        else 0.0
    )

    combined_net_20d_to_volume = (
        combined_net_20d / avg_volume_20
        if avg_volume_20 > 0
        else 0.0
    )

    close_price = latest.get("close_price", 0) or 0
    foreign_net_value_5d = foreign_net_5d * close_price
    institution_net_value_5d = institution_net_5d * close_price
    combined_net_value_5d = combined_net_5d * close_price

    flow_score = 50.0
    reasons = []

    if foreign_net_5d > 0:
        flow_score += 10
        reasons.append("외국인 5일 순매수")
    elif foreign_net_5d < 0:
        flow_score -= 10
        reasons.append("외국인 5일 순매도")

    if institution_net_5d > 0:
        flow_score += 10
        reasons.append("기관 5일 순매수")
    elif institution_net_5d < 0:
        flow_score -= 10
        reasons.append("기관 5일 순매도")

    if foreign_net_20d > 0:
        flow_score += 6
        reasons.append("외국인 20일 누적 순매수")
    elif foreign_net_20d < 0:
        flow_score -= 6
        reasons.append("외국인 20일 누적 순매도")

    if institution_net_20d > 0:
        flow_score += 6
        reasons.append("기관 20일 누적 순매수")
    elif institution_net_20d < 0:
        flow_score -= 6
        reasons.append("기관 20일 누적 순매도")

    if foreign_net_5d > 0 and institution_net_5d > 0:
        flow_score += 10
        reasons.append("외국인·기관 동반 5일 순매수")

    if foreign_net_5d < 0 and institution_net_5d < 0:
        flow_score -= 10
        reasons.append("외국인·기관 동반 5일 순매도")

    if combined_net_5d_to_volume > 0.05:
        flow_score += 5
        reasons.append("5일 순매수 규모가 평균 거래량 대비 의미 있음")
    elif combined_net_5d_to_volume < -0.05:
        flow_score -= 5
        reasons.append("5일 순매도 규모가 평균 거래량 대비 부담")

    flow_score = max(0.0, min(100.0, flow_score))

    if flow_score >= 70:
        flow_label = "strong_inflow"
        action_hint = "수급 양호 / 리서치·가격 조건과 함께 관심"
    elif flow_score >= 55:
        flow_label = "mild_inflow"
        action_hint = "수급 중립 이상 / 관찰 유지"
    elif flow_score >= 40:
        flow_label = "neutral"
        action_hint = "수급 중립 / 가격 조건 우선 확인"
    else:
        flow_label = "outflow_pressure"
        action_hint = "수급 부담 / 신규 진입 보수적 접근"

    return {
        "stock_code": latest["stock_code"],
        "stock_name": latest["stock_name"],
        "sector": latest["sector"],
        "latest_date": latest["date"],
        "latest_close": int(close_price),
        "latest_volume": int(latest_volume),
        "foreign_net_1d": foreign_net_1d,
        "foreign_net_5d": foreign_net_5d,
        "foreign_net_20d": foreign_net_20d,
        "institution_net_1d": institution_net_1d,
        "institution_net_5d": institution_net_5d,
        "institution_net_20d": institution_net_20d,
        "combined_net_1d": combined_net_1d,
        "combined_net_5d": combined_net_5d,
        "combined_net_20d": combined_net_20d,
        "foreign_net_value_5d": int(foreign_net_value_5d),
        "institution_net_value_5d": int(institution_net_value_5d),
        "combined_net_value_5d": int(combined_net_value_5d),
        "combined_net_5d_to_avg_volume_20": round(combined_net_5d_to_volume, 4),
        "combined_net_20d_to_avg_volume_20": round(combined_net_20d_to_volume, 4),
        "foreign_weight": latest.get("foreign_weight", 0.0),
        "foreign_limit_exhaustion_rate": latest.get("foreign_limit_exhaustion_rate", 0.0),
        "flow_score": round(flow_score, 2),
        "flow_label": flow_label,
        "action_hint": action_hint,
        "reasons": reasons,
        "row_count": len(rows),
    }


def build_sector_flow_scores(flow_features: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for code, feature in flow_features.items():
        sector = feature.get("sector", "unknown")

        if sector not in grouped:
            grouped[sector] = []

        grouped[sector].append(feature)

    sector_scores = {}

    for sector, items in grouped.items():
        if not items:
            continue

        avg_score = sum(float(item.get("flow_score", 50.0)) for item in items) / len(items)
        combined_net_value_5d = sum(int(item.get("combined_net_value_5d", 0)) for item in items)

        sector_scores[sector] = {
            "flow_score": round(avg_score, 2),
            "combined_net_value_5d": int(combined_net_value_5d),
            "items": items,
        }

    return sector_scores


def build_markdown_report(features_output: Dict[str, Any]) -> str:
    lines = []

    lines.append("# 국내주식 수급 피처 리포트")
    lines.append("")
    lines.append(f"- 생성 시각 UTC: {features_output.get('generated_at')}")
    lines.append(f"- 소스: {features_output.get('source')}")
    lines.append("- 기준: 외국인/기관 수급 기반 1차 룰 피처")
    lines.append("")

    features = features_output.get("features", {})
    sector_scores = features_output.get("sector_flow_scores", {})

    lines.append("## 1. 섹터별 수급 점수")
    if sector_scores:
        ranked = sorted(
            sector_scores.items(),
            key=lambda kv: kv[1].get("flow_score", 0),
            reverse=True,
        )

        for sector, data in ranked:
            lines.append(
                f"- {sector}: flow_score={data.get('flow_score')} "
                f"/ 5일 합산 순매수대금 후보={data.get('combined_net_value_5d'):,}원"
            )
    else:
        lines.append("- 섹터 수급 점수가 없습니다.")
    lines.append("")

    lines.append("## 2. 종목별 수급 피처")
    if features:
        ranked_items = sorted(
            features.values(),
            key=lambda x: x.get("flow_score", 0),
            reverse=True,
        )

        for item in ranked_items:
            lines.append(
                f"- {item.get('stock_name')}({item.get('stock_code')}) "
                f"/ flow_score={item.get('flow_score')} "
                f"/ label={item.get('flow_label')} "
                f"/ 외국인5D={item.get('foreign_net_5d'):,}주 "
                f"/ 기관5D={item.get('institution_net_5d'):,}주 "
                f"/ 합산5D={item.get('combined_net_5d'):,}주 "
                f"/ 힌트={item.get('action_hint')}"
            )

            if item.get("reasons"):
                lines.append(f"  - 근거: {', '.join(item.get('reasons', [])[:6])}")
    else:
        lines.append("- 종목별 수급 피처가 없습니다.")
    lines.append("")

    lines.append("## 3. 주의")
    lines.append("- 기관 수급 응답 key는 환경에 따라 다를 수 있으므로 flow_sync_diagnostic.json에서 sample_raw_keys를 확인해야 합니다.")
    lines.append("- 429 rate limit이 발생하면 --sleep 값을 늘리거나 --max-stocks를 줄이세요.")
    lines.append("- 이 리포트는 추천 보조 피처이며, 주문을 실행하지 않습니다.")

    return "\n".join(lines)


def sync_flows(max_stocks: int, sleep_seconds: float):
    client = KiwoomRestClient()
    universe = DEFAULT_UNIVERSE[:max_stocks]

    flows_daily: Dict[str, Any] = {}
    flow_features: Dict[str, Dict[str, Any]] = {}
    diagnostics = []

    for item in universe:
        code = normalize_stock_code(item["code"])
        name = item["name"]
        sector = item["sector"]

        print(f"Fetching flows: {name}({code})")

        foreign_result = None
        institution_result = None
        merged_rows: List[Dict[str, Any]] = []

        try:
            foreign_result = fetch_foreign_trading_trend(client, code)
            time.sleep(sleep_seconds)

            institution_result = fetch_institutional_stock(client, code)
            time.sleep(sleep_seconds)

            foreign_rows = extract_foreign_rows(foreign_result)
            institution_rows = extract_institution_rows(institution_result)

            merged_rows = merge_flow_rows(
                stock_code=code,
                stock_name=name,
                sector=sector,
                foreign_rows=foreign_rows,
                institution_rows=institution_rows,
            )

            flows_daily[code] = {
                "stock_code": code,
                "stock_name": name,
                "sector": sector,
                "rows": merged_rows,
            }

            feature = build_flow_feature(merged_rows)

            if feature:
                flow_features[code] = feature

            foreign_body = safe_body(foreign_result)
            institution_body = safe_body(institution_result)

            diagnostics.append({
                "stock_code": code,
                "stock_name": name,
                "foreign": {
                    "status": get_return_status(foreign_result),
                    "body_keys": sorted(list(foreign_body.keys())),
                    "row_count": len(foreign_rows),
                    "sample_raw_keys": sorted(list(foreign_rows[0].keys())) if foreign_rows and isinstance(foreign_rows[0], dict) else [],
                    "sample_first_row": foreign_rows[0] if foreign_rows else None,
                },
                "institution": {
                    "status": get_return_status(institution_result),
                    "body_keys": sorted(list(institution_body.keys())),
                    "row_count": len(institution_rows),
                    "sample_raw_keys": sorted(list(institution_rows[0].keys())) if institution_rows and isinstance(institution_rows[0], dict) else [],
                    "sample_first_row": institution_rows[0] if institution_rows else None,
                },
                "merged_row_count": len(merged_rows),
                "feature_created": feature is not None,
            })

        except Exception as e:
            diagnostics.append({
                "stock_code": code,
                "stock_name": name,
                "error": str(e),
            })

    sector_flow_scores = build_sector_flow_scores(flow_features)

    flows_output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "kiwoom_rest_api",
        "universe": universe,
        "flows": flows_daily,
    }

    features_output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "kiwoom_rest_api",
        "features": flow_features,
        "sector_flow_scores": sector_flow_scores,
    }

    diagnostic_output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "kiwoom_rest_api",
        "diagnostics": diagnostics,
    }

    FLOWS_DAILY_PATH.write_text(
        json.dumps(flows_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    FLOW_FEATURES_PATH.write_text(
        json.dumps(features_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    FLOW_DIAGNOSTIC_PATH.write_text(
        json.dumps(diagnostic_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown = build_markdown_report(features_output)

    FLOW_REPORT_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print(f"Saved flows daily: {FLOWS_DAILY_PATH}")
    print(f"Saved flow features: {FLOW_FEATURES_PATH}")
    print(f"Saved diagnostic: {FLOW_DIAGNOSTIC_PATH}")
    print(f"Saved report: {FLOW_REPORT_PATH}")
    print("")
    print(markdown)

    return flows_output, features_output, diagnostic_output


def main():
    parser = argparse.ArgumentParser(description="Sync foreign/institution flow data from Kiwoom REST API")
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=5,
        help="How many symbols from DEFAULT_UNIVERSE to fetch.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Sleep seconds between requests to avoid rate limit.",
    )

    args = parser.parse_args()

    sync_flows(
        max_stocks=args.max_stocks,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()
