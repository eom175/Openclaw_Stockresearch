import csv
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from policylink.paths import (
    DATA_DIR,
    LABEL_DATASET_REPORT_PATH,
    MODEL_DATASET_CSV_PATH,
    MODEL_DATASET_JSONL_PATH,
    PRICES_DAILY_PATH,
    REPORTS_DIR,
)
from policylink.utils import (
    load_json,
    load_jsonl,
    normalize_code,
    parse_float,
    parse_int,
    save_jsonl,
)


DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


DATASET_JSONL_PATH = MODEL_DATASET_JSONL_PATH
DATASET_CSV_PATH = MODEL_DATASET_CSV_PATH

LABEL_REPORT_PATH = LABEL_DATASET_REPORT_PATH


LABEL_COLUMNS = [
    "future_return_1d",
    "future_return_5d",
    "future_return_20d",
    "future_outperform_5d",
    "future_max_drawdown_5d",
    "label_status",
    "label_updated_at",
]


DEFAULT_CSV_COLUMNS = [
    "snapshot_date",
    "generated_at",
    "stock_code",
    "stock_name",
    "sector",

    "is_holding",
    "holding_quantity",
    "holding_eval_amount",
    "holding_weight",
    "holding_pnl",
    "holding_return_rate",

    "account_total_equity",
    "account_cash",
    "account_cash_weight",
    "account_invested_weight",
    "pending_or_reserved_amount",

    "research_score",
    "price_score",
    "flow_score",
    "final_score",
    "target_weight",
    "recommendation_rank",

    "latest_date",
    "latest_close",
    "latest_volume",
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "ma20_gap",
    "ma60_gap",
    "drawdown_20d",
    "volume_ratio_20",
    "trend_label",
    "price_risk_label",

    "foreign_net_1d",
    "foreign_net_5d",
    "foreign_net_20d",
    "institution_net_1d",
    "institution_net_5d",
    "institution_net_20d",
    "combined_net_1d",
    "combined_net_5d",
    "combined_net_20d",
    "combined_net_5d_to_avg_volume_20",
    "combined_net_20d_to_avg_volume_20",
    "foreign_weight",
    "foreign_limit_exhaustion_rate",
    "flow_label",

    "risk_level",
    "risk_score",
    "opportunity_score",
    "macro_pressure_score",

    "future_return_1d",
    "future_return_5d",
    "future_return_20d",
    "future_outperform_5d",
    "future_max_drawdown_5d",
    "label_status",
    "label_updated_at",
]


def infer_csv_columns(rows: List[Dict[str, Any]]) -> List[str]:
    columns = list(DEFAULT_CSV_COLUMNS)
    seen = set(columns)

    for row in rows:
        for key in row.keys():
            if key not in seen:
                columns.append(key)
                seen.add(key)

    return columns


def save_csv(path, rows: List[Dict[str, Any]]):
    columns = infer_csv_columns(rows)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: row.get(key)
                for key in columns
            })


def build_price_index(prices_daily: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    prices = prices_daily.get("prices", {})

    if not isinstance(prices, dict):
        return {}

    result = {}

    for code, data in prices.items():
        normalized_code = normalize_code(code)

        rows = data.get("rows", []) if isinstance(data, dict) else []

        if not isinstance(rows, list):
            rows = []

        clean_rows = []

        for row in rows:
            if not isinstance(row, dict):
                continue

            date = str(row.get("date") or "")
            close = parse_float(row.get("close"), None)

            if not date or close is None or close <= 0:
                continue

            clean_rows.append({
                "date": date,
                "close": float(close),
                "raw": row,
            })

        clean_rows.sort(key=lambda x: x["date"])
        result[normalized_code] = clean_rows

    return result


def find_base_index(price_rows: List[Dict[str, Any]], snapshot_date: str) -> Optional[int]:
    """
    snapshot_date와 같은 거래일이 있으면 그 날짜를 사용.
    없으면 snapshot_date 이전 또는 같은 가장 가까운 거래일을 사용.
    """
    if not price_rows:
        return None

    target = str(snapshot_date)

    exact_idx = None
    previous_idx = None

    for idx, row in enumerate(price_rows):
        row_date = str(row.get("date"))

        if row_date == target:
            exact_idx = idx
            break

        if row_date <= target:
            previous_idx = idx

    if exact_idx is not None:
        return exact_idx

    return previous_idx


def future_return(price_rows: List[Dict[str, Any]], base_idx: int, horizon: int) -> Optional[float]:
    future_idx = base_idx + horizon

    if base_idx < 0 or future_idx >= len(price_rows):
        return None

    base_close = price_rows[base_idx]["close"]
    future_close = price_rows[future_idx]["close"]

    if base_close <= 0:
        return None

    return (future_close / base_close) - 1.0


def future_max_drawdown(price_rows: List[Dict[str, Any]], base_idx: int, horizon: int) -> Optional[float]:
    if base_idx < 0:
        return None

    end_idx = base_idx + horizon

    if end_idx >= len(price_rows):
        return None

    base_close = price_rows[base_idx]["close"]

    if base_close <= 0:
        return None

    future_closes = [
        row["close"]
        for row in price_rows[base_idx + 1:end_idx + 1]
    ]

    if not future_closes:
        return None

    min_close = min(future_closes)
    return (min_close / base_close) - 1.0


def label_single_row(row: Dict[str, Any], price_index: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    updated = dict(row)

    stock_code = normalize_code(row.get("stock_code"))
    snapshot_date = str(row.get("snapshot_date") or "")

    price_rows = price_index.get(stock_code, [])

    updated["label_updated_at"] = datetime.now(timezone.utc).isoformat()

    if not stock_code or not snapshot_date:
        updated["label_status"] = "missing_key"
        return updated

    if not price_rows:
        updated["label_status"] = "missing_price_history"
        return updated

    base_idx = find_base_index(price_rows, snapshot_date)

    if base_idx is None:
        updated["label_status"] = "missing_base_price"
        return updated

    ret_1d = future_return(price_rows, base_idx, 1)
    ret_5d = future_return(price_rows, base_idx, 5)
    ret_20d = future_return(price_rows, base_idx, 20)
    max_dd_5d = future_max_drawdown(price_rows, base_idx, 5)

    updated["future_return_1d"] = round(ret_1d, 6) if ret_1d is not None else None
    updated["future_return_5d"] = round(ret_5d, 6) if ret_5d is not None else None
    updated["future_return_20d"] = round(ret_20d, 6) if ret_20d is not None else None
    updated["future_max_drawdown_5d"] = round(max_dd_5d, 6) if max_dd_5d is not None else None

    if ret_5d is not None:
        updated["label_status"] = "partially_labeled"

        if ret_20d is not None:
            updated["label_status"] = "labeled"
    else:
        updated["label_status"] = "pending_future_prices"

    return updated


def add_cross_sectional_outperform_labels(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    같은 snapshot_date 안에서 future_return_5d가 median보다 높으면 outperform=1.
    아직 future_return_5d가 없으면 None.
    """
    grouped = {}

    for idx, row in enumerate(rows):
        snapshot_date = str(row.get("snapshot_date") or "")
        ret_5d = parse_float(row.get("future_return_5d"), None)

        if snapshot_date and ret_5d is not None:
            grouped.setdefault(snapshot_date, [])
            grouped[snapshot_date].append((idx, ret_5d))

    for snapshot_date, items in grouped.items():
        if not items:
            continue

        sorted_returns = sorted(ret for _, ret in items)
        n = len(sorted_returns)

        if n == 0:
            continue

        if n % 2 == 1:
            median = sorted_returns[n // 2]
        else:
            median = (sorted_returns[n // 2 - 1] + sorted_returns[n // 2]) / 2

        for idx, ret in items:
            rows[idx]["future_outperform_5d"] = 1 if ret > median else 0

    for row in rows:
        if row.get("future_return_5d") is None:
            row["future_outperform_5d"] = None

    return rows


def label_dataset() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dataset_rows = load_jsonl(DATASET_JSONL_PATH)
    prices_daily = load_json(PRICES_DAILY_PATH, {"prices": {}})
    price_index = build_price_index(prices_daily)

    labeled_rows = []

    for row in dataset_rows:
        labeled_rows.append(
            label_single_row(row, price_index)
        )

    labeled_rows = add_cross_sectional_outperform_labels(labeled_rows)

    stats = {
        "total_rows": len(labeled_rows),
        "labeled": sum(1 for row in labeled_rows if row.get("label_status") == "labeled"),
        "partially_labeled": sum(1 for row in labeled_rows if row.get("label_status") == "partially_labeled"),
        "pending_future_prices": sum(1 for row in labeled_rows if row.get("label_status") == "pending_future_prices"),
        "missing_price_history": sum(1 for row in labeled_rows if row.get("label_status") == "missing_price_history"),
        "missing_base_price": sum(1 for row in labeled_rows if row.get("label_status") == "missing_base_price"),
        "missing_key": sum(1 for row in labeled_rows if row.get("label_status") == "missing_key"),
    }

    return labeled_rows, stats


def build_markdown_report(rows: List[Dict[str, Any]], stats: Dict[str, Any]) -> str:
    lines = []

    lines.append("# 모델 데이터셋 라벨링 리포트")
    lines.append("")
    lines.append(f"- 생성 시각 UTC: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- 입력 데이터셋: `{DATASET_JSONL_PATH}`")
    lines.append(f"- 가격 데이터: `{PRICES_DAILY_PATH}`")
    lines.append("")

    lines.append("## 1. 라벨링 상태")
    lines.append(f"- total rows: {stats.get('total_rows')}")
    lines.append(f"- labeled: {stats.get('labeled')}")
    lines.append(f"- partially_labeled: {stats.get('partially_labeled')}")
    lines.append(f"- pending_future_prices: {stats.get('pending_future_prices')}")
    lines.append(f"- missing_price_history: {stats.get('missing_price_history')}")
    lines.append(f"- missing_base_price: {stats.get('missing_base_price')}")
    lines.append(f"- missing_key: {stats.get('missing_key')}")
    lines.append("")

    lines.append("## 2. 최근 라벨 예시")
    recent_rows = sorted(
        rows,
        key=lambda x: (
            str(x.get("snapshot_date")),
            str(x.get("stock_code")),
        ),
        reverse=True,
    )[:20]

    for row in recent_rows:
        lines.append(
            f"- {row.get('snapshot_date')} "
            f"{row.get('stock_name')}({row.get('stock_code')}) "
            f"/ label_status={row.get('label_status')} "
            f"/ future_1d={row.get('future_return_1d')} "
            f"/ future_5d={row.get('future_return_5d')} "
            f"/ future_20d={row.get('future_return_20d')} "
            f"/ outperform_5d={row.get('future_outperform_5d')}"
        )

    lines.append("")
    lines.append("## 3. 해석")
    lines.append("- 오늘 생성한 스냅샷은 아직 미래 가격이 없기 때문에 pending_future_prices로 남는 것이 정상입니다.")
    lines.append("- 5거래일 이상 지난 스냅샷부터 future_return_5d와 future_outperform_5d가 채워집니다.")
    lines.append("- 20거래일 이상 지난 스냅샷은 future_return_20d까지 채워지고 label_status가 labeled로 바뀝니다.")
    lines.append("")

    lines.append("## 4. 다음 단계")
    lines.append("- 며칠간 데이터를 계속 쌓은 뒤 `train_model.py`를 실행합니다.")
    lines.append("- 시계열 데이터이므로 학습/검증은 시간 순서 기반 분할을 사용해야 합니다.")
    lines.append("- XGBoost 학습 후 feature importance를 확인해 리서치·가격·수급 중 어떤 피처가 의미 있었는지 검토합니다.")

    return "\n".join(lines)


def main():
    labeled_rows, stats = label_dataset()

    save_jsonl(DATASET_JSONL_PATH, labeled_rows)
    save_csv(DATASET_CSV_PATH, labeled_rows)

    markdown = build_markdown_report(labeled_rows, stats)
    LABEL_REPORT_PATH.write_text(markdown, encoding="utf-8")

    print(f"Updated dataset jsonl: {DATASET_JSONL_PATH}")
    print(f"Updated dataset csv: {DATASET_CSV_PATH}")
    print(f"Saved report: {LABEL_REPORT_PATH}")
    print("")
    print(markdown)


if __name__ == "__main__":
    main()
