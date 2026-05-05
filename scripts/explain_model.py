#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _bootstrap import add_src_to_path

add_src_to_path()

from calibrate_model import active_artifacts, load_model
from experiment_models import FEATURE_GROUPS, dataset_path, read_dataset
from train_model import build_matrix, labeled_frame
from policylink.paths import MODEL_EXPLAINABILITY_JSON_PATH, MODEL_EXPLAINABILITY_MD_PATH, ensure_project_dirs
from policylink.utils import load_json


LEAKAGE_TOKENS = ["future", "target", "label", "next", "tomorrow"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def no_op(reason: str, dataset: Optional[Path] = None) -> Dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "mode": "model_explainability",
        "explain_status": "no_op",
        "dataset_path": str(dataset) if dataset else None,
        "reason": reason,
        "top_features": [],
        "group_importance": {},
        "leakage_warnings": [],
        "order_enabled": False,
    }


def group_for_feature(feature: str) -> str:
    for group, selectors in FEATURE_GROUPS.items():
        for selector in selectors:
            if feature == selector or feature.startswith(selector):
                return group
    return "other"


def aggregate_groups(items: List[Dict[str, Any]]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for item in items:
        group = group_for_feature(str(item.get("feature") or ""))
        totals[group] = totals.get(group, 0.0) + float(item.get("importance") or 0.0)
    total = sum(totals.values()) or 1.0
    return {key: round(value / total, 6) for key, value in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)}


def built_in_importance(model: Any, encoded_columns: List[str]) -> List[Dict[str, Any]]:
    values = getattr(model, "feature_importances_", None)
    if values is None and hasattr(model, "named_steps"):
        final_step = list(model.named_steps.values())[-1]
        values = getattr(final_step, "feature_importances_", None)
        if values is None:
            coef = getattr(final_step, "coef_", None)
            if coef is not None:
                values = abs(coef[0])
    if values is None:
        return []
    items = [{"feature": feature, "importance": round(float(value), 8)} for feature, value in zip(encoded_columns, values)]
    items.sort(key=lambda row: row["importance"], reverse=True)
    return items


def shap_importance(model: Any, X, encoded_columns: List[str]) -> List[Dict[str, Any]]:
    import shap
    import numpy as np

    sample = X.head(min(len(X), 200))
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[-1]
    mean_abs = np.abs(values).mean(axis=0)
    items = [{"feature": feature, "importance": round(float(value), 8)} for feature, value in zip(encoded_columns, mean_abs)]
    items.sort(key=lambda row: row["importance"], reverse=True)
    return items


def permutation_importance_fallback(model: Any, X, y, encoded_columns: List[str]) -> List[Dict[str, Any]]:
    from sklearn.inspection import permutation_importance

    sample = X.tail(min(len(X), 300))
    target = y.loc[sample.index]
    result = permutation_importance(model, sample, target, n_repeats=5, random_state=42, scoring="average_precision")
    items = [{"feature": feature, "importance": round(float(value), 8)} for feature, value in zip(encoded_columns, result.importances_mean)]
    items.sort(key=lambda row: row["importance"], reverse=True)
    return items


def explain(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_project_dirs()
    path = dataset_path(args)
    df = read_dataset(path)
    labeled = labeled_frame(df)
    if labeled.empty:
        return no_op("labeled dataset이 비어 있습니다.", path)

    source, artifacts = active_artifacts()
    if not source or not artifacts.get("model_path") or not artifacts.get("feature_columns_path"):
        return no_op("active model 또는 best experiment model이 없습니다.", path)
    meta = load_json(Path(artifacts["feature_columns_path"]), {})
    encoded_columns = meta.get("encoded_columns") or []
    if not encoded_columns:
        return no_op("feature metadata에 encoded_columns가 없습니다.", path)

    X, _ = build_matrix(
        labeled,
        meta.get("numeric_columns", []),
        meta.get("categorical_columns", []),
        meta.get("numeric_medians", {}),
        encoded_columns,
    )
    y = labeled["future_outperform_5d"].astype(int)
    model = load_model(artifacts["model_path"])
    method = "built_in"
    notes = []
    items: List[Dict[str, Any]] = []
    if args.use_shap:
        try:
            items = shap_importance(model, X, encoded_columns)
            method = "shap"
        except Exception as exc:
            notes.append(f"SHAP failed, fallback used: {str(exc)[:200]}")
    if not items:
        items = built_in_importance(model, encoded_columns)
    if not items:
        try:
            items = permutation_importance_fallback(model, X, y, encoded_columns)
            method = "permutation"
        except Exception as exc:
            return no_op(f"importance 계산 실패: {str(exc)[:300]}", path)

    leakage = [feature for feature in encoded_columns if any(token in feature.lower() for token in LEAKAGE_TOKENS)]
    top = items[: args.top_n]
    return {
        "generated_at": utc_now(),
        "mode": "model_explainability",
        "explain_status": "completed",
        "dataset_path": str(path),
        "model_source": source,
        "model_path": artifacts.get("model_path"),
        "method": method,
        "top_features": top,
        "group_importance": aggregate_groups(items),
        "leakage_warnings": leakage,
        "notes": notes,
        "order_enabled": False,
    }


def build_report(result: Dict[str, Any]) -> str:
    lines = [
        "# Model Explainability Report",
        "",
        f"- generated_at: {result.get('generated_at')}",
        "- order_enabled=false",
        f"- explain_status: {result.get('explain_status')}",
        f"- dataset_path: {result.get('dataset_path')}",
        f"- method: {result.get('method')}",
        "",
    ]
    if result.get("explain_status") == "no_op":
        lines.append("## No-op")
        lines.append(f"- reason: {result.get('reason')}")
        return "\n".join(lines)
    lines.append("## Top Features")
    for item in result.get("top_features", []):
        lines.append(f"- {item.get('feature')}: {item.get('importance')}")
    lines.extend(["", "## Feature Group Importance"])
    for group, value in (result.get("group_importance") or {}).items():
        lines.append(f"- {group}: {value}")
    lines.extend(["", "## Leakage-looking Feature Warnings"])
    warnings = result.get("leakage_warnings") or []
    if warnings:
        for feature in warnings[:30]:
            lines.append(f"- {feature}")
    else:
        lines.append("- 없음")
    lines.extend(["", "## 한계", "- 설명성 지표는 모델의 연관성을 요약할 뿐 투자 확정 근거가 아닙니다."])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain active/best model feature importance.")
    parser.add_argument("--dataset-path")
    parser.add_argument("--use-historical", action="store_true")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--use-shap", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = explain(args)
    save_json(MODEL_EXPLAINABILITY_JSON_PATH, result)
    MODEL_EXPLAINABILITY_MD_PATH.write_text(build_report(result), encoding="utf-8")
    print(f"Saved explainability json: {MODEL_EXPLAINABILITY_JSON_PATH}")
    print(f"Saved explainability report: {MODEL_EXPLAINABILITY_MD_PATH}")
    print(f"explain_status={result.get('explain_status')}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
