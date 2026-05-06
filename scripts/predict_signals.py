#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from _bootstrap import add_src_to_path

add_src_to_path()

from policylink.paths import (
    ACTIVE_CALIBRATED_OUTPERFORM_MODEL_PATH,
    ACTIVE_CLASSIFIER_JOBLIB_PATH,
    ACTIVE_FEATURE_COLUMNS_PATH,
    ACTIVE_XGB_DRAWDOWN_MODEL_PATH,
    ACTIVE_XGB_OUTPERFORM_MODEL_PATH,
    ACTIVE_XGB_RETURN_MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    ML_SIGNALS_JSON_PATH,
    ML_SIGNALS_MD_PATH,
    MODEL_DATASET_CSV_PATH,
    MODEL_REGISTRY_PATH,
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


def no_model_payload(snapshot_date: Optional[str], reason: str, status: str = "no_model") -> Dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "mode": "ml_signal_generation",
        "model_status": status,
        "model_source": "no_model",
        "calibrated": False,
        "snapshot_date": snapshot_date,
        "best_threshold": None,
        "threshold_used": None,
        "selected_threshold": None,
        "experiment_id": None,
        "signal_policy": {
            "primary_signal": "rule",
            "recommended_policy": "rule_first_ml_modifier",
            "rule_weight": 0.75,
            "ml_weight": 0.25,
            "auto_order_allowed": False,
        },
        "warning": ["no_model"],
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
        f"- model_source: {payload.get('model_source')}",
        f"- calibrated: {payload.get('calibrated')}",
        f"- snapshot_date: {payload.get('snapshot_date')}",
        f"- best_threshold: {payload.get('best_threshold')}",
        f"- signal_policy: {(payload.get('signal_policy') or {}).get('recommended_policy')}",
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
            lines.append(f"- {item.get('stock_name')}({item.get('stock_code')}) / signal={item.get('signal_score')}")
    else:
        lines.append("- avoid_or_sell_candidate 신호 없음")

    lines.extend([
        "",
        "## 주의사항",
        "- ML 신호는 매수/매도 확정이 아니라 후보 scoring 보조 지표입니다.",
        "- fallback 또는 calibration 미완료 모델의 확률은 보수적으로 해석합니다.",
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


def resolve_model_context(registry: Dict[str, Any]) -> Dict[str, Any]:
    if ACTIVE_CALIBRATED_OUTPERFORM_MODEL_PATH.exists() and ACTIVE_FEATURE_COLUMNS_PATH.exists():
        return {
            "model_source": "active",
            "calibrated": True,
            "classifier_path": ACTIVE_CALIBRATED_OUTPERFORM_MODEL_PATH,
            "return_model_path": ACTIVE_XGB_RETURN_MODEL_PATH if ACTIVE_XGB_RETURN_MODEL_PATH.exists() else XGB_RETURN_MODEL_PATH,
            "drawdown_model_path": ACTIVE_XGB_DRAWDOWN_MODEL_PATH if ACTIVE_XGB_DRAWDOWN_MODEL_PATH.exists() else XGB_DRAWDOWN_MODEL_PATH,
            "feature_columns_path": ACTIVE_FEATURE_COLUMNS_PATH,
            "experiment_id": registry.get("active_experiment_id") or registry.get("active_model_version"),
        }
    if ACTIVE_XGB_OUTPERFORM_MODEL_PATH.exists() and ACTIVE_FEATURE_COLUMNS_PATH.exists():
        return {
            "model_source": "active",
            "calibrated": False,
            "classifier_path": ACTIVE_XGB_OUTPERFORM_MODEL_PATH,
            "return_model_path": ACTIVE_XGB_RETURN_MODEL_PATH if ACTIVE_XGB_RETURN_MODEL_PATH.exists() else XGB_RETURN_MODEL_PATH,
            "drawdown_model_path": ACTIVE_XGB_DRAWDOWN_MODEL_PATH if ACTIVE_XGB_DRAWDOWN_MODEL_PATH.exists() else XGB_DRAWDOWN_MODEL_PATH,
            "feature_columns_path": ACTIVE_FEATURE_COLUMNS_PATH,
            "experiment_id": registry.get("active_experiment_id") or registry.get("active_model_version"),
        }
    if ACTIVE_CLASSIFIER_JOBLIB_PATH.exists() and ACTIVE_FEATURE_COLUMNS_PATH.exists():
        return {
            "model_source": "active",
            "calibrated": False,
            "classifier_path": ACTIVE_CLASSIFIER_JOBLIB_PATH,
            "return_model_path": ACTIVE_XGB_RETURN_MODEL_PATH if ACTIVE_XGB_RETURN_MODEL_PATH.exists() else XGB_RETURN_MODEL_PATH,
            "drawdown_model_path": ACTIVE_XGB_DRAWDOWN_MODEL_PATH if ACTIVE_XGB_DRAWDOWN_MODEL_PATH.exists() else XGB_DRAWDOWN_MODEL_PATH,
            "feature_columns_path": ACTIVE_FEATURE_COLUMNS_PATH,
            "experiment_id": registry.get("active_experiment_id") or registry.get("active_model_version"),
        }
    if XGB_OUTPERFORM_MODEL_PATH.exists() and FEATURE_COLUMNS_PATH.exists():
        return {
            "model_source": "fallback",
            "calibrated": False,
            "classifier_path": XGB_OUTPERFORM_MODEL_PATH,
            "return_model_path": XGB_RETURN_MODEL_PATH,
            "drawdown_model_path": XGB_DRAWDOWN_MODEL_PATH,
            "feature_columns_path": FEATURE_COLUMNS_PATH,
            "experiment_id": registry.get("latest_train_experiment_id") or registry.get("active_model_version"),
        }
    return {}


def load_classifier(path, calibrated: bool):
    if calibrated or str(path).endswith(".joblib"):
        import joblib

        return joblib.load(path)
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(path)
    return model


def load_regressor(path):
    if not path or not path.exists():
        return None
    from xgboost import XGBRegressor

    model = XGBRegressor()
    model.load_model(path)
    return model


def build_matrix(df, numeric_columns: List[str], categorical_columns: List[str], numeric_medians: Dict[str, float], encoded_columns: List[str]):
    import pandas as pd

    parts = []
    if numeric_columns:
        numeric = df.reindex(columns=numeric_columns).apply(pd.to_numeric, errors="coerce")
        for column in numeric_columns:
            numeric[column] = numeric[column].fillna(numeric_medians.get(column, 0.0))
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
    if value is None or value >= 0:
        return 0.0
    return min(10.0, abs(float(value)) * 250.0)


def label_for_score(score: float) -> str:
    if score >= 75:
        return "strong_buy_candidate"
    if score >= 60:
        return "buy_candidate"
    if score >= 45:
        return "watch"
    return "avoid_or_sell_candidate"


def maybe_downgrade_label(label: str, probability: float, best_threshold: float, use_best_threshold: bool) -> str:
    if not use_best_threshold:
        return label
    if label in {"strong_buy_candidate", "buy_candidate"} and probability < best_threshold:
        return "watch"
    return label


def cautious_label(label: str, model_source: str, calibrated: bool) -> str:
    if model_source == "fallback" or not calibrated:
        if label == "strong_buy_candidate":
            return "buy_candidate"
    return label


def predict(args: argparse.Namespace) -> Dict[str, Any]:
    df = read_dataset()
    if df.empty:
        return no_model_payload(args.snapshot_date, "data/model_dataset.csv가 없거나 비어 있습니다.")

    if "snapshot_date" not in df.columns:
        return no_model_payload(args.snapshot_date, "snapshot_date 컬럼이 없습니다.", status="insufficient_features")

    snapshot_date = args.snapshot_date or str(df["snapshot_date"].max())
    target = df.loc[df["snapshot_date"].astype(str) == str(snapshot_date)].copy()
    if target.empty:
        return no_model_payload(str(snapshot_date), "해당 snapshot_date row가 없습니다.", status="insufficient_features")

    registry = load_json(MODEL_REGISTRY_PATH, {})
    signal_policy = registry.get("signal_policy") if isinstance(registry.get("signal_policy"), dict) else {
        "primary_signal": "rule",
        "recommended_policy": "rule_first_ml_modifier",
        "rule_weight": 0.75,
        "ml_weight": 0.25,
        "auto_order_allowed": False,
    }
    model_context = resolve_model_context(registry)
    if not model_context:
        return no_model_payload(str(snapshot_date), "active/fallback 모델 파일이 없습니다.")

    feature_meta = load_json(model_context["feature_columns_path"], {})
    numeric_columns = feature_meta.get("numeric_columns", [])
    categorical_columns = feature_meta.get("categorical_columns", [])
    numeric_medians = feature_meta.get("numeric_medians", {})
    encoded_columns = feature_meta.get("encoded_columns", [])
    if not encoded_columns:
        return no_model_payload(str(snapshot_date), "feature_columns.json에 encoded_columns가 없습니다.", status="insufficient_features")

    X = build_matrix(target, numeric_columns, categorical_columns, numeric_medians, encoded_columns)
    if X.empty:
        return no_model_payload(str(snapshot_date), "예측 feature matrix가 비어 있습니다.", status="insufficient_features")

    try:
        classifier = load_classifier(model_context["classifier_path"], model_context.get("calibrated", False))
        return_model = load_regressor(model_context.get("return_model_path"))
    except Exception as exc:
        return no_model_payload(str(snapshot_date), f"모델 로드 실패: {str(exc)[:300]}")

    outperform_prob = classifier.predict_proba(X)[:, 1]
    predicted_return = return_model.predict(X) if return_model is not None else [0.0] * len(target)

    predicted_drawdown = [None] * len(target)
    drawdown_model = load_regressor(model_context.get("drawdown_model_path"))
    if drawdown_model is not None:
        predicted_drawdown = list(drawdown_model.predict(X))

    portfolio = load_json(PORTFOLIO_RECOMMENDATION_JSON_PATH, {})
    best_threshold = parse_number(registry.get("best_threshold"), 0.60)
    model_version = str(registry.get("active_model_version") or feature_meta.get("active_model_version") or "xgb_unknown")
    signals = {}

    duplicate_warning = len(set(target.get("stock_code", []))) != len(target)
    for idx, (_, row) in enumerate(target.iterrows()):
        code = normalize_code(row.get("stock_code"))
        if not code:
            continue
        prob = float(outperform_prob[idx])
        pred_ret = float(predicted_return[idx])
        pred_dd = None if predicted_drawdown[idx] is None else float(predicted_drawdown[idx])
        rule_final_score = parse_number(row.get("final_score"), 0.0)
        signal_score = (
            prob * 45.0
            + return_score(pred_ret)
            + clamp(rule_final_score, 0.0, 100.0) / 100.0 * 20.0
            - drawdown_penalty(pred_dd)
        )
        signal_score = round(clamp(signal_score), 2)
        label = maybe_downgrade_label(label_for_score(signal_score), prob, best_threshold, args.use_best_threshold)
        label = cautious_label(label, str(model_context.get("model_source") or ""), bool(model_context.get("calibrated")))

        warnings = []
        if model_context.get("model_source") == "fallback":
            warnings.append("fallback_model")
        if not model_context.get("calibrated"):
            warnings.append("calibration_missing")
        if "ml_selected_rows_too_small" in (signal_policy.get("promotion_blockers") or []):
            warnings.append("small_validation_sample")
        if pred_dd is not None and pred_dd < -0.04:
            warnings.append("predicted_drawdown_5d below -4%")
        if duplicate_warning:
            warnings.append("snapshot contains duplicate stock rows")

        signals[code] = {
            "stock_code": code,
            "stock_name": row.get("stock_name"),
            "sector": row.get("sector"),
            "model_source": model_context.get("model_source"),
            "calibrated": bool(model_context.get("calibrated")),
            "threshold_used": best_threshold,
            "selected_threshold": best_threshold,
            "signal_policy": signal_policy,
            "experiment_id": model_context.get("experiment_id"),
            "outperform_prob_5d": round(prob, 6),
            "predicted_return_5d": round(pred_ret, 6),
            "predicted_drawdown_5d": None if pred_dd is None else round(pred_dd, 6),
            "rule_final_score": round(rule_final_score, 4),
            "signal_score": signal_score,
            "signal_label": label,
            "model_version": model_version,
            "feature_warning": warnings,
            "warning": warnings,
            "feature_group_warnings": [],
        }

    return {
        "generated_at": utc_now(),
        "mode": "ml_signal_generation",
        "model_status": "available",
        "model_source": model_context.get("model_source"),
        "calibrated": bool(model_context.get("calibrated")),
        "snapshot_date": str(snapshot_date),
        "best_threshold": best_threshold,
        "threshold_used": best_threshold,
        "selected_threshold": best_threshold,
        "experiment_id": model_context.get("experiment_id"),
        "signal_policy": signal_policy,
        "warning": sorted({
            warning
            for item in signals.values()
            for warning in item.get("warning", [])
        }),
        "signals": signals,
        "portfolio_generated_at": portfolio.get("generated_at"),
        "order_enabled": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate today's ML signals from trained XGBoost models.")
    parser.add_argument("--snapshot-date")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--use-best-threshold", default="true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.use_best_threshold = str(args.use_best_threshold).lower() not in {"false", "0", "no"}
    payload = predict(args)
    write_payload(payload, args.top_n)
    print(f"Saved ML signals json: {ML_SIGNALS_JSON_PATH}")
    print(f"Saved ML signals report: {ML_SIGNALS_MD_PATH}")
    print(f"model_status={payload.get('model_status')}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
