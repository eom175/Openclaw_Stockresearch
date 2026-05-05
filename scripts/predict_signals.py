#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from _bootstrap import add_src_to_path

add_src_to_path()

from policylink.paths import (
    FEATURE_COLUMNS_PATH,
    ML_SIGNALS_JSON_PATH,
    ML_SIGNALS_MD_PATH,
    MODEL_DATASET_CSV_PATH,
    PORTFOLIO_RECOMMENDATION_JSON_PATH,
    XGB_DRAWDOWN_MODEL_PATH,
    XGB_OUTPERFORM_MODEL_PATH,
    XGB_RETURN_MODEL_PATH,
)
from policylink.utils import load_json, normalize_code, parse_number


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def save_json(path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def no_model_payload(snapshot_date: Optional[str], reason: str) -> Dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "mode": "ml_signal_generation",
        "model_status": "no_model",
        "snapshot_date": snapshot_date,
        "reason": reason,
        "signals": {},
        "order_enabled": False,
    }


def build_report(payload: Dict[str, Any], top_n: int) -> str:
    lines = [
        "# ML 신호 리포트",
        "",
        f"- 생성 시각 UTC: {payload.get('generated_at')}",
        "- 실제 주문 실행: 비활성화",
        "- order_enabled=false",
        f"- model_status: {payload.get('model_status')}",
        f"- snapshot_date: {payload.get('snapshot_date')}",
        "",
    ]

    if payload.get("model_status") != "available":
        lines.extend([
            "## 예측 보류",
            f"- 사유: {payload.get('reason', '모델 파일 없음')}",
            "",
            "모델이 학습되기 전까지 주문 후보 생성은 기존 rule 기반으로 계속 동작합니다.",
        ])
        return "\n".join(lines)

    signals = list((payload.get("signals") or {}).values())
    signals.sort(key=lambda x: parse_number(x.get("signal_score"), 0.0), reverse=True)

    lines.append("## 상위 후보")
    for item in signals[:top_n]:
        lines.append(
            f"- {item.get('stock_name')}({item.get('stock_code')}) "
            f"/ signal={item.get('signal_score')} "
            f"/ label={item.get('signal_label')} "
            f"/ outperform={item.get('outperform_prob_5d')} "
            f"/ pred_return={item.get('predicted_return_5d')} "
            f"/ drawdown={item.get('predicted_drawdown_5d')}"
        )

    lines.append("")
    lines.append("## 피해야 할 후보")
    avoid = [item for item in signals if item.get("signal_label") == "avoid_or_sell_candidate"]
    if avoid:
        for item in avoid[:top_n]:
            lines.append(
                f"- {item.get('stock_name')}({item.get('stock_code')}) "
                f"/ signal={item.get('signal_score')} / pred_return={item.get('predicted_return_5d')}"
            )
    else:
        lines.append("- avoid_or_sell_candidate 신호 없음")

    lines.extend([
        "",
        "## 주의사항",
        "- ML 신호는 매수/매도 확정이 아니라 후보 scoring 보조 지표입니다.",
        "- 이 스크립트는 주문 API를 호출하지 않습니다.",
    ])
    return "\n".join(lines)


def write_payload(payload: Dict[str, Any], top_n: int) -> None:
    save_json(ML_SIGNALS_JSON_PATH, payload)
    ML_SIGNALS_MD_PATH.write_text(build_report(payload, top_n), encoding="utf-8")


def read_dataset():
    import pandas as pd

    if not MODEL_DATASET_CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MODEL_DATASET_CSV_PATH)


def build_matrix(df, numeric_columns: List[str], categorical_columns: List[str], encoded_columns: List[str]):
    import pandas as pd

    parts = []
    if numeric_columns:
        numeric = df.reindex(columns=numeric_columns).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        parts.append(numeric)
    if categorical_columns:
        categorical = df.reindex(columns=categorical_columns).fillna("missing").astype(str)
        encoded = pd.get_dummies(categorical, columns=categorical_columns, dummy_na=False)
        parts.append(encoded)

    if parts:
        matrix = pd.concat(parts, axis=1)
    else:
        matrix = pd.DataFrame(index=df.index)

    return matrix.reindex(columns=encoded_columns, fill_value=0.0).astype(float)


def return_score(value: Optional[float]) -> float:
    if value is None:
        return 12.5
    return clamp((float(value) + 0.05) / 0.10 * 25.0, 0.0, 25.0)


def drawdown_penalty(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    if value >= 0:
        return 0.0
    return min(20.0, abs(float(value)) * 300.0)


def label_for_score(score: float) -> str:
    if score >= 75:
        return "strong_buy_candidate"
    if score >= 60:
        return "buy_candidate"
    if score >= 45:
        return "watch"
    return "avoid_or_sell_candidate"


def predict(args: argparse.Namespace) -> Dict[str, Any]:
    df = read_dataset()
    if df.empty:
        return no_model_payload(args.snapshot_date, "data/model_dataset.csv가 없거나 비어 있습니다.")

    if "snapshot_date" not in df.columns:
        return no_model_payload(args.snapshot_date, "snapshot_date 컬럼이 없습니다.")

    snapshot_date = args.snapshot_date or str(df["snapshot_date"].max())
    target = df.loc[df["snapshot_date"].astype(str) == str(snapshot_date)].copy()
    if target.empty:
        return no_model_payload(str(snapshot_date), "해당 snapshot_date row가 없습니다.")

    required_paths = [FEATURE_COLUMNS_PATH, XGB_OUTPERFORM_MODEL_PATH, XGB_RETURN_MODEL_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        return no_model_payload(str(snapshot_date), f"필수 모델 파일이 없습니다: {', '.join(missing)}")

    feature_meta = load_json(FEATURE_COLUMNS_PATH, {})
    numeric_columns = feature_meta.get("numeric_columns", [])
    categorical_columns = feature_meta.get("categorical_columns", [])
    encoded_columns = feature_meta.get("encoded_columns", [])
    if not encoded_columns:
        return no_model_payload(str(snapshot_date), "feature_columns.json에 encoded_columns가 없습니다.")

    try:
        from xgboost import XGBClassifier, XGBRegressor
    except Exception as exc:
        return no_model_payload(str(snapshot_date), f"xgboost import 실패: {exc}")

    X = build_matrix(target, numeric_columns, categorical_columns, encoded_columns)
    if X.empty:
        payload = no_model_payload(str(snapshot_date), "예측 feature matrix가 비어 있습니다.")
        payload["model_status"] = "insufficient_features"
        return payload

    classifier = XGBClassifier()
    classifier.load_model(XGB_OUTPERFORM_MODEL_PATH)
    return_model = XGBRegressor()
    return_model.load_model(XGB_RETURN_MODEL_PATH)

    outperform_prob = classifier.predict_proba(X)[:, 1]
    predicted_return = return_model.predict(X)

    predicted_drawdown = [None] * len(target)
    has_drawdown = XGB_DRAWDOWN_MODEL_PATH.exists()
    if has_drawdown:
        drawdown_model = XGBRegressor()
        drawdown_model.load_model(XGB_DRAWDOWN_MODEL_PATH)
        predicted_drawdown = list(drawdown_model.predict(X))

    portfolio = load_json(PORTFOLIO_RECOMMENDATION_JSON_PATH, {})
    model_version = str(feature_meta.get("trained_at") or "xgb_unknown")
    signals = {}

    for idx, (_, row) in enumerate(target.iterrows()):
        code = normalize_code(row.get("stock_code"))
        if not code:
            continue
        prob = float(outperform_prob[idx])
        pred_ret = float(predicted_return[idx])
        pred_dd = None if predicted_drawdown[idx] is None else float(predicted_drawdown[idx])
        rule_final_score = parse_number(row.get("final_score"), 0.0)
        signal_score = (
            prob * 50.0
            + return_score(pred_ret)
            + clamp(rule_final_score, 0.0, 100.0) / 100.0 * 25.0
            - drawdown_penalty(pred_dd)
        )
        signal_score = round(clamp(signal_score), 2)

        warnings = []
        if pred_dd is not None and pred_dd < -0.04:
            warnings.append("predicted_drawdown_5d below -4%")
        if len(set(target.get("stock_code", []))) != len(target):
            warnings.append("snapshot contains duplicate stock rows")

        signals[code] = {
            "stock_code": code,
            "stock_name": row.get("stock_name"),
            "sector": row.get("sector"),
            "outperform_prob_5d": round(prob, 6),
            "predicted_return_5d": round(pred_ret, 6),
            "predicted_drawdown_5d": None if pred_dd is None else round(pred_dd, 6),
            "rule_final_score": round(rule_final_score, 4),
            "signal_score": signal_score,
            "signal_label": label_for_score(signal_score),
            "model_version": model_version,
            "feature_warning": warnings,
        }

    return {
        "generated_at": utc_now(),
        "mode": "ml_signal_generation",
        "model_status": "available",
        "snapshot_date": str(snapshot_date),
        "signals": signals,
        "portfolio_generated_at": portfolio.get("generated_at"),
        "order_enabled": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate today's ML signals from trained XGBoost models.")
    parser.add_argument("--snapshot-date")
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = predict(args)
    write_payload(payload, args.top_n)
    print(f"Saved ML signals json: {ML_SIGNALS_JSON_PATH}")
    print(f"Saved ML signals report: {ML_SIGNALS_MD_PATH}")
    print(f"model_status={payload.get('model_status')}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
