#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _bootstrap import add_src_to_path

add_src_to_path()

from train_model import (
    build_matrix,
    choose_feature_columns,
    finite_float,
    labeled_frame as training_labeled_frame,
    ml_dependency_check,
    walk_forward_splits,
)
from policylink.paths import (
    BACKTEST_METRICS_PATH,
    BACKTEST_REPORT_PATH,
    HISTORICAL_MODEL_DATASET_CSV_PATH,
    ML_SIGNALS_JSON_PATH,
    MODEL_DATASET_CSV_PATH,
    MODEL_REGISTRY_PATH,
)
from policylink.utils import load_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_dataset_path(args: argparse.Namespace) -> Path:
    if args.dataset_path:
        return Path(args.dataset_path)
    if args.use_historical:
        return HISTORICAL_MODEL_DATASET_CSV_PATH
    return MODEL_DATASET_CSV_PATH


def read_dataset(args: argparse.Namespace):
    import pandas as pd

    dataset_path = resolve_dataset_path(args)
    if not dataset_path.exists():
        return pd.DataFrame()
    return pd.read_csv(dataset_path)


def round_trip_cost(args: argparse.Namespace) -> Tuple[float, float]:
    bps = args.round_trip_cost_bps
    if bps is None:
        bps = float(args.fee_bps) + float(args.slippage_bps)
    return float(bps), float(bps) / 10000.0


def no_op(reason: str, labeled_rows: int = 0, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "generated_at": utc_now(),
        "mode": "signal_backtest",
        "backtest_status": "no_op",
        "reason": reason,
        "labeled_rows": labeled_rows,
        "metrics": {},
        "ml_walk_forward_metrics": {},
        "order_enabled": False,
    }
    if extra:
        payload.update(extra)
    return payload


def resolve_score_column(df, requested: str) -> Optional[str]:
    aliases = {
        "ml_signal_score": ["ml_signal_score", "signal_score"],
        "outperform_prob_5d": ["outperform_prob_5d", "outperform_prob"],
        "predicted_return_5d": ["predicted_return_5d"],
        "final_score": ["final_score"],
    }
    for column in aliases.get(requested, [requested]):
        if column in df.columns:
            return column
    return None


def labeled_for_score(df, target_column: str, score_column: str):
    import pandas as pd

    actual_score_column = resolve_score_column(df, score_column)
    if df.empty or target_column not in df.columns or actual_score_column is None:
        return df.iloc[0:0].copy(), actual_score_column
    mask = df.get("label_status", pd.Series([""] * len(df))).isin(["labeled", "partially_labeled"])
    mask &= pd.to_numeric(df[target_column], errors="coerce").notna()
    mask &= pd.to_numeric(df[actual_score_column], errors="coerce").notna()
    result = df.loc[mask].copy()
    result[target_column] = pd.to_numeric(result[target_column], errors="coerce")
    result[actual_score_column] = pd.to_numeric(result[actual_score_column], errors="coerce")
    if "future_outperform_5d" in result.columns:
        result["future_outperform_5d"] = pd.to_numeric(result["future_outperform_5d"], errors="coerce")
    if "future_max_drawdown_5d" in result.columns:
        result["future_max_drawdown_5d"] = pd.to_numeric(result["future_max_drawdown_5d"], errors="coerce")
    return result.sort_values(["snapshot_date", actual_score_column]), actual_score_column


def turnover_proxy(selected_by_date: Dict[str, set]) -> float:
    dates = sorted(selected_by_date.keys())
    if len(dates) <= 1:
        return 0.0
    changes = []
    for prev, cur in zip(dates, dates[1:]):
        prev_set = selected_by_date[prev]
        cur_set = selected_by_date[cur]
        if not prev_set and not cur_set:
            changes.append(0.0)
            continue
        overlap = len(prev_set & cur_set)
        base = max(len(prev_set), len(cur_set), 1)
        changes.append(1.0 - overlap / base)
    return round(sum(changes) / len(changes), 6)


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def predicted_return_score(value: float) -> float:
    return clamp((float(value) + 0.05) / 0.10 * 30.0, 0.0, 30.0)


def score_0_100(value: Any) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    if numeric != numeric:
        return 0.0
    return clamp(numeric, 0.0, 100.0)


def signal_policy_weights(registry: Dict[str, Any], ml_signals: Dict[str, Any]) -> Dict[str, Any]:
    signal_policy = registry.get("signal_policy") if isinstance(registry.get("signal_policy"), dict) else {}
    if signal_policy:
        return {
            "rule_weight": float(signal_policy.get("rule_weight", 0.75)),
            "ml_weight": float(signal_policy.get("ml_weight", 0.25)),
            "reason": signal_policy.get("reason", "registry signal_policy"),
            "model_source": ml_signals.get("model_source") or signal_policy.get("model_source"),
            "calibrated": bool(ml_signals.get("calibrated", False)),
        }

    model_source = str(ml_signals.get("model_source") or ("active" if registry.get("active_model") else "fallback"))
    calibrated = bool(ml_signals.get("calibrated", False))
    if model_source == "fallback" or not calibrated:
        return {
            "rule_weight": 0.75,
            "ml_weight": 0.25,
            "reason": "fallback or uncalibrated ML model",
            "model_source": model_source,
            "calibrated": calibrated,
        }
    return {
        "rule_weight": 0.60,
        "ml_weight": 0.40,
        "reason": "active calibrated ML model",
        "model_source": model_source,
        "calibrated": calibrated,
    }


def backtest_ranked_rows(
    labeled,
    score_column: str,
    target_column: str,
    top_k: int,
    cost_return: float,
) -> Dict[str, Any]:
    import pandas as pd

    selected_parts = []
    selected_by_date: Dict[str, set] = {}
    for snapshot_date, group in labeled.groupby("snapshot_date"):
        top = group.sort_values(score_column, ascending=False).head(top_k)
        selected_parts.append(top)
        selected_by_date[str(snapshot_date)] = set(top.get("stock_code", []))

    if not selected_parts:
        return {}

    selected = pd.concat(selected_parts)
    target = pd.to_numeric(labeled[target_column], errors="coerce")
    top_target = pd.to_numeric(selected[target_column], errors="coerce")
    drawdown = selected["future_max_drawdown_5d"] if "future_max_drawdown_5d" in selected.columns else None
    outperform = selected["future_outperform_5d"] if "future_outperform_5d" in selected.columns else None

    gross_top = float(top_target.mean())
    gross_all = float(target.mean())
    return {
        "score_column": score_column,
        "target_column": target_column,
        "top_k": top_k,
        "date_count": int(labeled["snapshot_date"].nunique()) if "snapshot_date" in labeled.columns else 0,
        "labeled_rows": int(len(labeled)),
        "selected_rows": int(len(selected)),
        "top_k_mean_return": finite_float(gross_top),
        "all_mean_return": finite_float(gross_all),
        "excess_return": finite_float(gross_top - gross_all),
        "top_k_mean_net_return": finite_float(gross_top - cost_return),
        "all_mean_net_return": finite_float(gross_all - cost_return),
        "excess_net_return": finite_float((gross_top - cost_return) - (gross_all - cost_return)),
        "hit_rate": finite_float((target > 0).mean()),
        "top_k_hit_rate": finite_float((top_target > 0).mean()),
        "precision_at_k": finite_float(outperform.mean()) if outperform is not None and outperform.notna().any() else finite_float((top_target > 0).mean()),
        "top_k_avg_drawdown": finite_float(drawdown.mean()) if drawdown is not None and drawdown.notna().any() else None,
        "turnover_proxy": turnover_proxy(selected_by_date),
    }


def stability_score(rows, score_column: str, target_column: str, top_k: int, cost_return: float) -> Dict[str, Any]:
    if rows is None or rows.empty or "walk_forward_fold" not in rows.columns:
        return {"stability_score": None, "fold_excess_net_returns": [], "stability_warning": "fold data unavailable"}

    fold_excess: List[float] = []
    for _, fold_rows in rows.groupby("walk_forward_fold"):
        metrics = backtest_ranked_rows(fold_rows, score_column, target_column, top_k, cost_return)
        if metrics.get("excess_net_return") is not None:
            fold_excess.append(float(metrics["excess_net_return"]))
    if not fold_excess:
        return {"stability_score": None, "fold_excess_net_returns": [], "stability_warning": "fold excess unavailable"}

    import pandas as pd

    values = pd.Series(fold_excess, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    warning = "validation sample is small" if len(rows) < 100 or rows["snapshot_date"].nunique() < 30 else None
    return {
        "stability_score": finite_float(mean / (std + 1e-9)),
        "fold_excess_net_returns": [finite_float(value) for value in fold_excess],
        "fold_excess_net_return_mean": finite_float(mean),
        "fold_excess_net_return_std": finite_float(std),
        "stability_warning": warning,
    }


def fit_ml_walk_forward_predictions(labeled, args: argparse.Namespace):
    import pandas as pd
    from xgboost import XGBClassifier, XGBRegressor

    numeric_columns, categorical_columns, numeric_medians = choose_feature_columns(labeled)
    X, encoded_columns = build_matrix(labeled, numeric_columns, categorical_columns, numeric_medians)
    y_class = labeled["future_outperform_5d"].astype(int)
    y_return = labeled["future_return_5d"].astype(float)
    splits = walk_forward_splits(labeled, args.n_splits, test_size_days=max(1, args.test_size_days), gap_days=max(0, args.gap_days))
    validation_parts = []
    fold_metrics = []

    for fold_no, (train_idx, val_idx) in enumerate(splits, start=1):
        if len(set(y_class.loc[train_idx])) < 2:
            continue
        classifier = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=args.random_state,
            n_jobs=-1,
        )
        return_model = XGBRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=args.random_state,
            n_jobs=-1,
        )
        classifier.fit(X.loc[train_idx], y_class.loc[train_idx])
        return_model.fit(X.loc[train_idx], y_return.loc[train_idx])
        prob = classifier.predict_proba(X.loc[val_idx])[:, 1]
        pred_return = return_model.predict(X.loc[val_idx])
        validation = labeled.loc[val_idx].copy()
        validation["walk_forward_fold"] = fold_no
        validation["outperform_prob_5d"] = prob
        validation["predicted_return_5d"] = pred_return
        validation["ml_signal_score"] = [
            clamp(float(p) * 70.0 + predicted_return_score(float(r)))
            for p, r in zip(prob, pred_return)
        ]
        validation_parts.append(validation)
        fold_metrics.append({
            "fold": fold_no,
            "train_rows": len(train_idx),
            "validation_rows": len(val_idx),
            "validation_dates": int(validation["snapshot_date"].nunique()),
            "prob_mean": finite_float(float(pd.Series(prob).mean())),
            "predicted_return_mean": finite_float(float(pd.Series(pred_return).mean())),
        })

    if not validation_parts:
        return None, [], {"skipped": True, "reason": "valid ML walk-forward fold가 없습니다."}

    validation_rows = pd.concat(validation_parts).sort_values(["snapshot_date", "stock_code"] if "stock_code" in labeled.columns else ["snapshot_date"])
    meta = {
        "feature_count": len(encoded_columns),
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_count": len(categorical_columns),
        "fold_count": len(fold_metrics),
        "folds": fold_metrics,
    }
    return validation_rows, encoded_columns, meta


def run_rule_backtest(df, args: argparse.Namespace, cost_return: float) -> Dict[str, Any]:
    labeled, actual_score_column = labeled_for_score(df, args.target_column, args.score_column)
    if len(labeled) < args.min_labeled_rows:
        return {
            "status": "no_op",
            "reason": f"labeled rows {len(labeled)}개 < min_labeled_rows={args.min_labeled_rows}",
            "requested_score_column": args.score_column,
            "resolved_score_column": actual_score_column,
            "labeled_rows": len(labeled),
        }
    return backtest_ranked_rows(labeled, actual_score_column or args.score_column, args.target_column, args.top_k, cost_return)


def run_ml_walk_forward(df, args: argparse.Namespace, cost_return: float, registry: Dict[str, Any], ml_signals: Dict[str, Any]) -> Dict[str, Any]:
    dependency = ml_dependency_check()
    if not dependency.get("ok"):
        return {
            "status": "no_op_dependency_unavailable",
            "missing_dependencies": dependency.get("missing_dependencies", []),
            "dependency_details": dependency.get("dependency_details", []),
            "system_install_suggestions": dependency.get("system_install_suggestions", []),
            "install_command": dependency.get("install_command"),
        }
    labeled = training_labeled_frame(df)
    if len(labeled) < args.min_labeled_rows:
        return {
            "status": "no_op",
            "reason": f"labeled rows {len(labeled)}개 < min_labeled_rows={args.min_labeled_rows}",
            "labeled_rows": len(labeled),
        }
    if labeled["snapshot_date"].nunique() < 3:
        return {
            "status": "no_op",
            "reason": "ML walk-forward에 필요한 snapshot_date 수가 부족합니다.",
            "labeled_rows": len(labeled),
        }
    validation_rows, _, meta = fit_ml_walk_forward_predictions(labeled, args)
    if validation_rows is None:
        return {"status": "no_op", **meta}

    weights = signal_policy_weights(registry, ml_signals)
    validation_rows["rule_score"] = validation_rows["final_score"].apply(score_0_100) if "final_score" in validation_rows.columns else 0.0
    validation_rows["ml_score"] = validation_rows["ml_signal_score"].apply(score_0_100)
    validation_rows["ensemble_score"] = (
        validation_rows["rule_score"] * weights["rule_weight"]
        + validation_rows["ml_score"] * weights["ml_weight"]
    )

    metrics = backtest_ranked_rows(validation_rows, "ml_score", args.target_column, args.top_k, cost_return)
    metrics.update(stability_score(validation_rows, "ml_score", args.target_column, args.top_k, cost_return))
    metrics.update({
        "status": "completed",
        "score_column": "ml_score",
        "outperform_prob_backtest": backtest_ranked_rows(validation_rows, "outperform_prob_5d", args.target_column, args.top_k, cost_return),
        "predicted_return_backtest": backtest_ranked_rows(validation_rows, "predicted_return_5d", args.target_column, args.top_k, cost_return),
        "walk_forward_meta": meta,
        "policy_weights": weights,
    })
    metrics["same_window_rule_metrics"] = backtest_ranked_rows(
        validation_rows,
        "rule_score",
        args.target_column,
        args.top_k,
        cost_return,
    )
    metrics["same_window_rule_metrics"].update(stability_score(validation_rows, "rule_score", args.target_column, args.top_k, cost_return))
    metrics["same_window_ensemble_metrics"] = backtest_ranked_rows(
        validation_rows,
        "ensemble_score",
        args.target_column,
        args.top_k,
        cost_return,
    )
    metrics["same_window_ensemble_metrics"].update(stability_score(validation_rows, "ensemble_score", args.target_column, args.top_k, cost_return))
    return metrics


def compare_rule_and_ml(rule_metrics: Dict[str, Any], ml_metrics: Dict[str, Any]) -> Dict[str, Any]:
    if ml_metrics.get("status") != "completed":
        return {}
    same_rule = ml_metrics.get("same_window_rule_metrics") or {}
    ensemble = ml_metrics.get("same_window_ensemble_metrics") or {}
    return {
        "comparison_window": "ml_walk_forward_validation_dates",
        "top_k_mean_net_return_diff_ml_minus_rule": finite_float((ml_metrics.get("top_k_mean_net_return") or 0.0) - (same_rule.get("top_k_mean_net_return") or 0.0)),
        "excess_net_return_diff_ml_minus_rule": finite_float((ml_metrics.get("excess_net_return") or 0.0) - (same_rule.get("excess_net_return") or 0.0)),
        "precision_at_k_diff_ml_minus_rule": finite_float((ml_metrics.get("precision_at_k") or 0.0) - (same_rule.get("precision_at_k") or 0.0)),
        "excess_net_return_diff_ensemble_minus_rule": finite_float((ensemble.get("excess_net_return") or 0.0) - (same_rule.get("excess_net_return") or 0.0)),
        "excess_net_return_diff_ensemble_minus_ml": finite_float((ensemble.get("excess_net_return") or 0.0) - (ml_metrics.get("excess_net_return") or 0.0)),
        "selected_rows_rule": same_rule.get("selected_rows"),
        "selected_rows_ml": ml_metrics.get("selected_rows"),
        "selected_rows_ensemble": ensemble.get("selected_rows"),
    }


def judgment_lines(ml_metrics: Dict[str, Any]) -> List[str]:
    if ml_metrics.get("status") != "completed":
        return ["ML walk-forward comparison is unavailable; keep rule-first scoring."]
    lines: List[str] = []
    same_rule = ml_metrics.get("same_window_rule_metrics") or {}
    ensemble = ml_metrics.get("same_window_ensemble_metrics") or {}
    if (ml_metrics.get("selected_rows") or 0) < 100:
        lines.append("ML validation sample is small; do not use ML as primary trading signal.")
    if (ml_metrics.get("excess_net_return") or 0.0) <= (same_rule.get("excess_net_return") or 0.0):
        lines.append("ML does not dominate rule baseline on excess net return.")
    ensemble_best = (
        (ensemble.get("excess_net_return") or -999.0) > (same_rule.get("excess_net_return") or -999.0)
        and (ensemble.get("excess_net_return") or -999.0) > (ml_metrics.get("excess_net_return") or -999.0)
    )
    if ensemble_best:
        lines.append("Ensemble candidate is eligible for further paper-trading evaluation.")
    else:
        lines.append("Keep rule-first scoring and use ML as secondary modifier.")
    return lines


def run_backtest(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = resolve_dataset_path(args)
    df = read_dataset(args)
    if df.empty:
        return no_op(f"{dataset_path} 파일이 없거나 비어 있습니다.", extra={"dataset_path": str(dataset_path)})

    cost_bps, cost_return = round_trip_cost(args)
    registry = load_json(MODEL_REGISTRY_PATH, {})
    ml_signals = load_json(ML_SIGNALS_JSON_PATH, {})
    rule_metrics = run_rule_backtest(df, args, cost_return)
    ml_metrics = run_ml_walk_forward(df, args, cost_return, registry, ml_signals) if args.ml_walk_forward else {}

    if rule_metrics.get("status") == "no_op" and not args.ml_walk_forward:
        result = no_op(rule_metrics.get("reason", "백테스트 가능 row가 부족합니다."), labeled_rows=rule_metrics.get("labeled_rows", 0))
        result["dataset_path"] = str(dataset_path)
        return result

    return {
        "generated_at": utc_now(),
        "mode": "signal_backtest",
        "dataset_path": str(dataset_path),
        "backtest_status": "completed",
        "model_status": registry.get("model_status"),
        "ml_signal_status": ml_signals.get("model_status"),
        "fee_bps": float(args.fee_bps),
        "slippage_bps": float(args.slippage_bps),
        "round_trip_cost_bps": cost_bps,
        "round_trip_cost_return": finite_float(cost_return),
        "metrics": rule_metrics,
        "ml_walk_forward_enabled": bool(args.ml_walk_forward),
        "ml_walk_forward_metrics": ml_metrics,
        "comparison": compare_rule_and_ml(rule_metrics, ml_metrics),
        "judgment": judgment_lines(ml_metrics) if args.ml_walk_forward else [],
        "order_enabled": False,
    }


def build_report(result: Dict[str, Any]) -> str:
    lines = [
        "# Backtest Report",
        "",
        f"- generated_at: {result.get('generated_at')}",
        "- 실제 주문 실행: 비활성화",
        "- order_enabled=false",
        f"- backtest_status: {result.get('backtest_status')}",
        f"- dataset_path: {result.get('dataset_path')}",
        f"- fee_bps: {result.get('fee_bps')}",
        f"- slippage_bps: {result.get('slippage_bps')}",
        f"- round_trip_cost_bps: {result.get('round_trip_cost_bps')}",
        "",
    ]
    if result.get("backtest_status") == "no_op":
        lines.append("## 백테스트 보류")
        lines.append(f"- 사유: {result.get('reason')}")
        lines.append(f"- labeled_rows: {result.get('labeled_rows', 0)}")
        return "\n".join(lines)

    lines.append("## Full-period rule backtest")
    metrics = result.get("metrics", {})
    if metrics.get("status") == "no_op":
        lines.append(f"- no_op: {metrics.get('reason')}")
    else:
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")

    if result.get("ml_walk_forward_enabled"):
        lines.append("")
        lines.append("## Same-window comparison: rule vs ML vs ensemble")
        ml_metrics = result.get("ml_walk_forward_metrics", {})
        if ml_metrics.get("status") != "completed":
            lines.append(f"- status: {ml_metrics.get('status')}")
            lines.append(f"- reason: {ml_metrics.get('reason')}")
            if ml_metrics.get("missing_dependencies"):
                lines.append(f"- missing_dependencies: {ml_metrics.get('missing_dependencies')}")
                lines.append(f"- install_command: {ml_metrics.get('install_command')}")
        else:
            same_rule = ml_metrics.get("same_window_rule_metrics") or {}
            ensemble = ml_metrics.get("same_window_ensemble_metrics") or {}
            lines.append(f"- policy_weights: {ml_metrics.get('policy_weights')}")
            lines.append("| signal | selected_rows | date_count | top_k_net | all_net | excess_net | precision_at_k | hit_rate | drawdown | turnover | stability |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for label, item in [
                ("rule_score", same_rule),
                ("ml_score", ml_metrics),
                ("ensemble_score", ensemble),
            ]:
                lines.append(
                    f"| {label} | {item.get('selected_rows')} | {item.get('date_count')} | "
                    f"{item.get('top_k_mean_net_return')} | {item.get('all_mean_net_return')} | "
                    f"{item.get('excess_net_return')} | {item.get('precision_at_k')} | "
                    f"{item.get('top_k_hit_rate')} | {item.get('top_k_avg_drawdown')} | "
                    f"{item.get('turnover_proxy')} | {item.get('stability_score')} |"
                )
            warnings = [
                item.get("stability_warning")
                for item in [same_rule, ml_metrics, ensemble]
                if item.get("stability_warning")
            ]
            if warnings:
                lines.append(f"- stability/sample warnings: {sorted(set(warnings))}")

        if result.get("comparison"):
            lines.append("")
            lines.append("## Rule vs ML")
            for key, value in result.get("comparison", {}).items():
                lines.append(f"- {key}: {value}")
        if result.get("judgment"):
            lines.append("")
            lines.append("## 판단")
            for item in result.get("judgment", []):
                lines.append(f"- {item}")

    lines.extend([
        "",
        "## 비용/슬리피지 가정",
        "- net_return은 gross future_return_5d에서 round_trip_cost를 단순 차감한 값입니다.",
        "- 이 비용 반영은 단순 추정이며 실제 세금, 수수료, 체결 슬리피지와 다를 수 있습니다.",
        "- 주문 실행 기능은 포함하지 않습니다.",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest rule/ML scores on labeled model_dataset rows.")
    parser.add_argument("--dataset-path")
    parser.add_argument("--use-historical", action="store_true")
    parser.add_argument("--score-column", default="final_score")
    parser.add_argument("--target-column", default="future_return_5d")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-labeled-rows", type=int, default=30)
    parser.add_argument("--ml-walk-forward", action="store_true")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--test-size-days", type=int, default=5)
    parser.add_argument("--gap-days", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--fee-bps", type=float, default=15.0)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--round-trip-cost-bps", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_backtest(args)
    save_json(BACKTEST_METRICS_PATH, result)
    BACKTEST_REPORT_PATH.write_text(build_report(result), encoding="utf-8")
    print(f"Saved backtest metrics: {BACKTEST_METRICS_PATH}")
    print(f"Saved backtest report: {BACKTEST_REPORT_PATH}")
    print(f"backtest_status={result.get('backtest_status')}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
