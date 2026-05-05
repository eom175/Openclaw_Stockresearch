import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from policylink.kiwoom.client import KiwoomRestClient
from policylink.paths import (
    DATA_DIR,
    PRICE_FEATURES_PATH,
    PRICE_SYNC_DIAGNOSTIC_PATH,
    PRICES_DAILY_PATH,
    REPORTS_DIR,
)
from policylink.universe import universe_for_market_data
from policylink.utils import (
    find_first_list,
    get_kst_today_yyyymmdd,
    parse_kiwoom_int,
    pick_first,
)


DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


PRICES_PATH = PRICES_DAILY_PATH
FEATURES_PATH = PRICE_FEATURES_PATH
DIAGNOSTIC_PATH = PRICE_SYNC_DIAGNOSTIC_PATH


DEFAULT_UNIVERSE = universe_for_market_data()


def fetch_daily_chart(
    client: KiwoomRestClient,
    stock_code: str,
    base_date: str,
):
    """
    ka10081 = 주식일봉차트조회요청.

    REST 문서/버전에 따라 body 필드명이 다를 수 있으므로
    가장 흔한 필드명을 우선 사용하고, raw 응답을 저장해서 필요 시 조정한다.
    """
    return client.post(
        endpoint="/api/dostk/chart",
        api_id="ka10081",
        data={
            "stk_cd": stock_code,
            "base_dt": base_date,
            "upd_stkpc_tp": "1",
        }
    )


def extract_daily_rows(raw_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    body = raw_result.get("body", {}) if isinstance(raw_result, dict) else {}

    candidate_keys = [
        "stk_dt_pole_chart_qry",
        "stk_dd_chart_qry",
        "chart",
        "output",
        "items",
        "list",
    ]

    rows = []

    if isinstance(body, dict):
        for key in candidate_keys:
            value = body.get(key)
            if isinstance(value, list):
                rows = value
                break

    if not rows:
        rows = find_first_list(body)

    return rows if isinstance(rows, list) else []


def normalize_daily_row(row: Dict[str, Any], stock_code: str, stock_name: str, sector: str) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict):
        return None

    date = pick_first(
        row,
        ["dt", "date", "trd_dt", "일자"],
        None,
    )

    close = parse_kiwoom_int(
        pick_first(
            row,
            ["cur_prc", "close_prc", "close", "clpr", "종가", "현재가"],
            None,
        ),
        0,
    )

    open_price = parse_kiwoom_int(
        pick_first(
            row,
            ["open_pric", "open", "stck_oprc", "시가"],
            None,
        ),
        0,
    )

    high = parse_kiwoom_int(
        pick_first(
            row,
            ["high_pric", "high", "stck_hgpr", "고가"],
            None,
        ),
        0,
    )

    low = parse_kiwoom_int(
        pick_first(
            row,
            ["low_pric", "low", "stck_lwpr", "저가"],
            None,
        ),
        0,
    )

    volume = parse_kiwoom_int(
        pick_first(
            row,
            ["trde_qty", "volume", "vol", "거래량"],
            None,
        ),
        0,
    )

    trading_value = parse_kiwoom_int(
        pick_first(
            row,
            ["trde_prica", "trading_value", "value", "거래대금"],
            None,
        ),
        0,
    )

    if not date or close <= 0:
        return None

    return {
        "date": str(date),
        "stock_code": stock_code,
        "stock_name": stock_name,
        "sector": sector,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "trading_value": trading_value,
    }


def compute_returns(values: List[float], idx: int, lag: int) -> Optional[float]:
    if idx - lag < 0:
        return None

    prev = values[idx - lag]

    if prev == 0:
        return None

    return (values[idx] / prev) - 1.0


def compute_volatility(returns: List[float]) -> Optional[float]:
    if len(returns) < 2:
        return None

    mean = sum(returns) / len(returns)
    variance = sum((x - mean) ** 2 for x in returns) / (len(returns) - 1)

    return variance ** 0.5


def moving_average(values: List[float]) -> Optional[float]:
    if not values:
        return None

    return sum(values) / len(values)


def compute_drawdown(values: List[float]) -> Optional[float]:
    if not values:
        return None

    peak = max(values)

    if peak <= 0:
        return None

    current = values[-1]
    return (current / peak) - 1.0


def build_features_for_stock(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None

    # date ascending
    rows = sorted(rows, key=lambda x: x["date"])

    closes = [float(row["close"]) for row in rows]
    volumes = [float(row["volume"]) for row in rows]

    idx = len(rows) - 1
    latest = rows[-1]

    daily_returns = []

    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            daily_returns.append((closes[i] / closes[i - 1]) - 1.0)

    return_1d = compute_returns(closes, idx, 1)
    return_5d = compute_returns(closes, idx, 5)
    return_20d = compute_returns(closes, idx, 20)

    recent_returns_20 = daily_returns[-20:] if len(daily_returns) >= 20 else daily_returns
    volatility_20d = compute_volatility(recent_returns_20)

    ma20 = moving_average(closes[-20:]) if len(closes) >= 20 else moving_average(closes)
    ma60 = moving_average(closes[-60:]) if len(closes) >= 60 else moving_average(closes)

    ma20_gap = ((closes[-1] / ma20) - 1.0) if ma20 and ma20 > 0 else None
    ma60_gap = ((closes[-1] / ma60) - 1.0) if ma60 and ma60 > 0 else None

    drawdown_20d = compute_drawdown(closes[-20:]) if len(closes) >= 20 else compute_drawdown(closes)

    avg_volume_20 = moving_average(volumes[-20:]) if len(volumes) >= 20 else moving_average(volumes)
    volume_ratio_20 = (volumes[-1] / avg_volume_20) if avg_volume_20 and avg_volume_20 > 0 else None

    trend_label = "neutral"

    if ma20_gap is not None and return_5d is not None:
        if ma20_gap > 0.03 and return_5d > 0:
            trend_label = "uptrend"
        elif ma20_gap < -0.03 and return_5d < 0:
            trend_label = "downtrend"

    risk_label = "normal"

    if volatility_20d is not None:
        if volatility_20d >= 0.04:
            risk_label = "high_volatility"
        elif volatility_20d <= 0.015:
            risk_label = "low_volatility"

    return {
        "stock_code": latest["stock_code"],
        "stock_name": latest["stock_name"],
        "sector": latest["sector"],
        "latest_date": latest["date"],
        "latest_close": latest["close"],
        "latest_volume": latest["volume"],
        "return_1d": round(return_1d, 4) if return_1d is not None else None,
        "return_5d": round(return_5d, 4) if return_5d is not None else None,
        "return_20d": round(return_20d, 4) if return_20d is not None else None,
        "volatility_20d": round(volatility_20d, 4) if volatility_20d is not None else None,
        "ma20": round(ma20, 2) if ma20 is not None else None,
        "ma60": round(ma60, 2) if ma60 is not None else None,
        "ma20_gap": round(ma20_gap, 4) if ma20_gap is not None else None,
        "ma60_gap": round(ma60_gap, 4) if ma60_gap is not None else None,
        "drawdown_20d": round(drawdown_20d, 4) if drawdown_20d is not None else None,
        "volume_ratio_20": round(volume_ratio_20, 4) if volume_ratio_20 is not None else None,
        "trend_label": trend_label,
        "risk_label": risk_label,
        "row_count": len(rows),
    }


def sync_prices(max_stocks: int, sleep_seconds: float):
    client = KiwoomRestClient()
    base_date = get_kst_today_yyyymmdd()

    all_prices = {}
    all_features = {}
    diagnostics = []

    universe = DEFAULT_UNIVERSE[:max_stocks]

    for item in universe:
        code = item["code"]
        name = item["name"]
        sector = item["sector"]

        print(f"Fetching daily chart: {name}({code})")

        try:
            raw_result = fetch_daily_chart(
                client=client,
                stock_code=code,
                base_date=base_date,
            )

            raw_rows = extract_daily_rows(raw_result)

            normalized_rows = []

            for row in raw_rows:
                normalized = normalize_daily_row(
                    row=row,
                    stock_code=code,
                    stock_name=name,
                    sector=sector,
                )

                if normalized:
                    normalized_rows.append(normalized)

            # date ascending 저장
            normalized_rows = sorted(normalized_rows, key=lambda x: x["date"])

            all_prices[code] = {
                "stock_code": code,
                "stock_name": name,
                "sector": sector,
                "rows": normalized_rows,
            }

            feature = build_features_for_stock(normalized_rows)

            if feature:
                all_features[code] = feature

            body = raw_result.get("body", {}) if isinstance(raw_result, dict) else {}

            diagnostics.append({
                "stock_code": code,
                "stock_name": name,
                "status_code": raw_result.get("status_code") if isinstance(raw_result, dict) else None,
                "return_code": body.get("return_code") if isinstance(body, dict) else None,
                "return_msg": body.get("return_msg") if isinstance(body, dict) else None,
                "body_keys": sorted(list(body.keys())) if isinstance(body, dict) else [],
                "raw_row_count": len(raw_rows),
                "normalized_row_count": len(normalized_rows),
                "sample_raw_keys": sorted(list(raw_rows[0].keys())) if raw_rows and isinstance(raw_rows[0], dict) else [],
            })

        except Exception as e:
            diagnostics.append({
                "stock_code": code,
                "stock_name": name,
                "error": str(e),
            })

        time.sleep(sleep_seconds)

    prices_output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_date": base_date,
        "source": "kiwoom_rest_api",
        "universe": universe,
        "prices": all_prices,
    }

    features_output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_date": base_date,
        "source": "kiwoom_rest_api",
        "features": all_features,
    }

    diagnostic_output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_date": base_date,
        "diagnostics": diagnostics,
    }

    PRICES_PATH.write_text(
        json.dumps(prices_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    FEATURES_PATH.write_text(
        json.dumps(features_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    DIAGNOSTIC_PATH.write_text(
        json.dumps(diagnostic_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved prices: {PRICES_PATH}")
    print(f"Saved features: {FEATURES_PATH}")
    print(f"Saved diagnostic: {DIAGNOSTIC_PATH}")

    return prices_output, features_output, diagnostic_output


def main():
    parser = argparse.ArgumentParser(description="Sync daily price data from Kiwoom REST API")
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=10,
        help="How many symbols from DEFAULT_UNIVERSE to fetch.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Sleep seconds between requests to avoid rate limit.",
    )

    args = parser.parse_args()

    sync_prices(
        max_stocks=args.max_stocks,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()
