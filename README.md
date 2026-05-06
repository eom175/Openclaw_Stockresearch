# Policy Research

OpenClaw + Telegram + Kiwoom 모의투자 환경에서 국내주식/국내상장 ETF 중심 리서치, 가격/수급 동기화, 포트폴리오 추천 리포트, 학습 데이터셋 스냅샷을 만드는 프로젝트다.

## 실행 환경

- 프로젝트 경로: `/Users/eomjiyong/policy-research`
- Python: `/Users/eomjiyong/policy-research/.venv/bin/python`
- 자동화: OpenClaw Gateway + Cron
- 개발/디버깅: VS Code launch configurations
- 거래 환경: Kiwoom REST API 모의투자

`.env`에는 Kiwoom/OpenDART/Naver 접속 정보가 들어가므로 열람하거나 출력하지 않는다.

Yahoo Finance 데이터는 `yfinance`를 이용해 내부 연구/교육 목적 compact feature로만 사용한다.

## 전체 파이프라인

```bash
cd /Users/eomjiyong/policy-research
/Users/eomjiyong/policy-research/.venv/bin/python scripts/run_full_pipeline.py
```

`scripts/run_full_pipeline.py`는 아래 순서를 실행하고 `reports/full_pipeline_report.md`에 단계별 성공/실패를 기록한다.

1. `collect_research.py --hours 720 --max-items 50`
2. `collect_dart.py --days 90 --max-stocks 10 --sleep 0.3`
3. `collect_naver_news.py --days 14 --max-stocks 10 --display 20 --sort date --sleep 0.3`
4. `collect_yahoo_finance.py --days 90 --news-count 10 --max-tickers 14 --sleep 0.3`
5. `kiwoom_daily_report.py`
6. `sync_prices.py --max-stocks 10 --sleep 0.7`
7. `sync_flows.py --max-stocks 5 --sleep 1.2`
8. `recommend_portfolio.py`
9. `build_dataset.py`
10. `label_dataset.py`
11. `audit_dataset.py`
12. `train_model.py --min-labeled-rows 100 --min-dates 10 --n-splits 3`
13. `predict_signals.py`
14. `backtest_signals.py`
15. `generate_order_proposals.py --max-buy-candidates 3 --max-sell-candidates 3 --max-order-amount 1000000`
16. `log_paper_trades.py --max-candidates 5 --only-approved-for-review`
17. `evaluate_paper_trades.py --horizon-days 5`

주문 실행 스크립트는 전체 파이프라인에 포함하지 않는다. 주문 후보 생성과 paper trading 단계도 `proposal_only=true`, `paper_only`, `order_enabled=false`이며 키움 주문 API를 호출하지 않는다. DART/뉴스/Yahoo 수집 실패, ML 학습 no-op, ML 예측 no_model, paper trade 라벨 부족은 warning으로 기록하고 파이프라인은 계속 진행한다.

## OpenClaw Cron 예시

```text
cd /Users/eomjiyong/policy-research
/Users/eomjiyong/policy-research/.venv/bin/python scripts/run_full_pipeline.py
```

Telegram 요약에는 `reports/portfolio_recommendation.md`, `reports/model_dataset_snapshot.md`, `reports/label_dataset_report.md`, `reports/full_pipeline_report.md`의 핵심만 전달한다. API key, token, 계좌번호 원문은 포함하지 않는다.

## 주요 산출물

- `data/candidates.compact.json`
- `data/dart_corp_codes.json`
- `data/dart_disclosures.json`
- `data/dart_event_features.json`
- `data/naver_news.json`
- `data/news_event_features.json`
- `data/yahoo_market_data.json`
- `data/yahoo_news.json`
- `data/yahoo_global_features.json`
- `reports/kiwoom_mock_account_summary.json`
- `data/price_features.json`
- `data/flow_features.json`
- `reports/dart_sync_diagnostic.json`
- `reports/dart_disclosures.md`
- `reports/naver_news_sync_diagnostic.json`
- `reports/naver_news_features.md`
- `reports/yahoo_finance_sync_diagnostic.json`
- `reports/yahoo_global_features.md`
- `reports/portfolio_recommendation.md`
- `reports/order_proposals.json`
- `reports/order_proposals.md`
- `reports/order_risk_check.json`
- `reports/order_risk_check.md`
- `data/order_ledger.jsonl`
- `data/model_dataset.jsonl`
- `data/model_dataset.csv`
- `data/backfill/prices_daily_history.json`
- `data/backfill/flows_daily_history.json`
- `data/backfill/dart_events_history.json`
- `data/backfill/naver_news_history.json`
- `data/backfill/yahoo_global_history.json`
- `data/backfill_status.json`
- `data/historical_model_dataset.jsonl`
- `data/historical_model_dataset.csv`
- `reports/backfill_market_history_report.md`
- `reports/historical_dataset_report.md`
- `reports/historical_dataset_audit_report.md`
- `reports/model_training_report.md`
- `reports/model_training_metrics.json`
- `reports/dataset_audit_report.md`
- `reports/dataset_audit.json`
- `reports/model_quality_report.md`
- `reports/model_quality_metrics.json`
- `reports/ml_signals.json`
- `reports/ml_signals.md`
- `reports/backtest_report.md`
- `reports/backtest_metrics.json`
- `reports/label_dataset_report.md`
- `reports/full_pipeline_report.md`

## 구조

- `scripts/`: OpenClaw Cron, VS Code launch, 수동 실행용 entrypoint
- `scripts/diagnostics/`: Kiwoom 진단용 스크립트
- `src/policylink/`: 실제 구현 모듈
- `data/`: 파이프라인 데이터 산출물
- `reports/`: 사람이 읽는 리포트와 진단 결과
- `raw/`: 민감하거나 원천 성격의 파일 보관 영역이며 git 제외 대상

## OpenDART 공시 수집

OpenDART 기업공시 수집을 사용하려면 OpenDART API key를 발급받고 `.env`에 `DART_API_KEY`를 추가한다. key 값은 터미널, 리포트, 문서에 출력하지 않는다.

수동 실행:

```bash
cd /Users/eomjiyong/policy-research
/Users/eomjiyong/policy-research/.venv/bin/python scripts/collect_dart.py --days 90 --max-stocks 10 --sleep 0.3
```

VS Code에서는 `DART - Collect Disclosures` launch 메뉴를 사용한다.

생성 파일:

- `data/dart_corp_codes.json`
- `data/dart_disclosures.json`
- `data/dart_event_features.json`
- `reports/dart_sync_diagnostic.json`
- `reports/dart_disclosures.md`

공시 데이터는 원문 전체가 아니라 접수번호, 제목, 날짜, 회사명, 링크용 URL과 제목 기반 compact event feature로 저장한다.

## Naver 뉴스 수집

네이버 뉴스 검색 피처를 사용하려면 Naver Developers에서 애플리케이션을 등록하고 검색 API 사용 신청을 완료한 뒤 `.env`에 `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`을 추가한다. 두 값은 터미널, 리포트, 문서에 출력하지 않는다.

수동 실행:

```bash
cd /Users/eomjiyong/policy-research
/Users/eomjiyong/policy-research/.venv/bin/python scripts/collect_naver_news.py --days 14 --max-stocks 10 --display 20 --sort date --sleep 0.3
```

VS Code에서는 `News - Collect Naver` launch 메뉴를 사용한다.

생성 파일:

- `data/naver_news.json`
- `data/news_event_features.json`
- `reports/naver_news_sync_diagnostic.json`
- `reports/naver_news_features.md`

뉴스 데이터는 기사 원문이 아니라 네이버 검색 API 응답의 제목, 요약, URL, 날짜와 compact keyword feature만 저장한다. 기사 본문 scraping은 하지 않는다.

## Yahoo Finance 글로벌 Proxy 수집

Yahoo Finance 글로벌 proxy 피처를 사용하려면 `yfinance`와 `pandas`가 필요하다.

```bash
cd /Users/eomjiyong/policy-research
/Users/eomjiyong/policy-research/.venv/bin/pip install yfinance pandas
```

수동 실행:

```bash
cd /Users/eomjiyong/policy-research
/Users/eomjiyong/policy-research/.venv/bin/python scripts/collect_yahoo_finance.py --days 90 --news-count 10 --max-tickers 14 --sleep 0.3
```

VS Code에서는 `Yahoo - Collect Global Features` launch 메뉴를 사용한다.

생성 파일:

- `data/yahoo_market_data.json`
- `data/yahoo_news.json`
- `data/yahoo_global_features.json`
- `reports/yahoo_finance_sync_diagnostic.json`
- `reports/yahoo_global_features.md`

Yahoo Finance/yfinance 데이터는 공식 제휴 API가 아니므로 내부 연구/교육 목적 compact feature로만 사용한다. 기사 원문 전체는 저장하지 않고 제목, 요약, URL, 발행시각, 관련 ticker와 proxy별 점수만 저장한다.

## Historical Backfill과 Point-in-time Dataset

daily dataset은 하루에 한 번 현재 snapshot만 쌓기 때문에 XGBoost 학습용 labeled sample이 천천히 늘어난다. 모델 정확도를 높이려면 과거 가격/수급/공시/뉴스/Yahoo proxy 데이터를 날짜별 snapshot으로 재구성한 `historical_model_dataset`을 별도로 만든다.

historical dataset은 point-in-time 원칙을 지킨다. 각 `snapshot_date`의 feature는 해당 날짜 당일 또는 이전 데이터만 사용하고, `future_*` 컬럼은 target 전용으로만 둔다. 과거 계좌 상태는 없으므로 historical dataset에서는 계좌/보유 feature를 0 또는 null로 처리한다.

Naver News API는 검색 결과 기반 API이며 완전한 과거 뉴스 아카이브가 아니다. 따라서 Naver 백필은 `display`, `start`, `sort=date`로 얻을 수 있는 검색 결과 안에서 compact metadata와 keyword feature만 만든다.

historical backfill은 무거운 작업이므로 daily `run_full_pipeline.py`에 포함하지 않는다. 수동 실행 또는 주 1회 실행을 권장한다.

실행 순서:

```bash
cd /Users/eomjiyong/policy-research
/Users/eomjiyong/policy-research/.venv/bin/python scripts/backfill_market_history.py --days 252 --max-stocks 10 --sleep 0.7
/Users/eomjiyong/policy-research/.venv/bin/python scripts/build_historical_dataset.py --min-history-days 80 --horizon-days 5 --max-snapshot-days 180
/Users/eomjiyong/policy-research/.venv/bin/python scripts/audit_historical_dataset.py
/Users/eomjiyong/policy-research/.venv/bin/python scripts/train_model.py --use-historical --min-labeled-rows 500 --min-dates 60 --n-splits 3
/Users/eomjiyong/policy-research/.venv/bin/python scripts/backtest_signals.py --use-historical --score-column final_score --target-column future_return_5d --top-k 3 --min-labeled-rows 100
```

VS Code에서는 `Backfill - Market History`, `Dataset - Build Historical`, `Dataset - Audit Historical`, `ML - Train Historical`, `ML - Backtest Historical` launch 메뉴를 사용한다.

생성 파일:

- `data/backfill/prices_daily_history.json`
- `data/backfill/flows_daily_history.json`
- `data/backfill/dart_events_history.json`
- `data/backfill/naver_news_history.json`
- `data/backfill/yahoo_global_history.json`
- `data/backfill_status.json`
- `data/historical_model_dataset.jsonl`
- `data/historical_model_dataset.csv`
- `reports/backfill_market_history_report.md`
- `reports/historical_dataset_report.md`
- `reports/historical_dataset_audit_report.md`

## ML 학습과 예측 신호

XGBoost 기반 ML 신호와 모델 품질 검증을 사용하려면 `xgboost`, `scikit-learn`, `pandas`, `numpy`, `joblib`이 필요하다.

```bash
cd /Users/eomjiyong/policy-research
/Users/eomjiyong/policy-research/.venv/bin/pip install xgboost scikit-learn pandas numpy joblib shap
```

macOS에서 XGBoost가 `libomp.dylib` 누락으로 로딩되지 않으면 OpenMP runtime 설치가 필요하다.

```bash
brew install libomp
```

학습 실행:

```bash
cd /Users/eomjiyong/policy-research
/Users/eomjiyong/policy-research/.venv/bin/python scripts/audit_dataset.py --min-rows 50 --min-labeled-rows 30 --min-dates 5
/Users/eomjiyong/policy-research/.venv/bin/python scripts/train_model.py --min-labeled-rows 100 --min-dates 10 --n-splits 3
```

예측 실행:

```bash
cd /Users/eomjiyong/policy-research
/Users/eomjiyong/policy-research/.venv/bin/python scripts/predict_signals.py
/Users/eomjiyong/policy-research/.venv/bin/python scripts/backtest_signals.py --score-column final_score --target-column future_return_5d --top-k 3 --min-labeled-rows 30
```

VS Code에서는 `Dataset - Audit`, `ML - Train Models`, `ML - Predict Signals`, `ML - Backtest Signals` launch 메뉴를 사용한다.

생성 파일:

- `models/xgb_outperform_5d.json`
- `models/xgb_return_5d.json`
- `models/xgb_drawdown_5d.json`, 가능할 때만
- `models/feature_columns.json`
- `models/model_registry.json`
- `reports/dataset_audit_report.md`
- `reports/dataset_audit.json`
- `reports/model_training_report.md`
- `reports/model_training_metrics.json`
- `reports/model_quality_report.md`
- `reports/model_quality_metrics.json`
- `reports/ml_signals.json`
- `reports/ml_signals.md`
- `reports/backtest_report.md`
- `reports/backtest_metrics.json`

`audit_dataset.py`는 row 수, 라벨 상태, class balance, missing ratio, duplicate key, leakage 의심 컬럼, feature freshness를 점검한다. `train_model.py`는 TimeSeriesSplit/walk-forward 방식으로 검증하며 랜덤 train/test split을 쓰지 않는다. labeled row와 snapshot_date가 충분하지 않으면 `train_model.py`는 실패하지 않고 “학습 보류” no-op 리포트를 만든다. 모델 파일이 없으면 `predict_signals.py`도 실패하지 않고 `no_model` 상태의 ML 신호 리포트를 만든다. `backtest_signals.py`는 snapshot_date별 top-k 단순 백테스트를 만들며, 데이터가 부족하면 no-op 리포트를 만든다. XGBoost 모델은 주문 후보 생성의 보조 점수로만 사용하며, 매수/매도 확정이나 주문 실행 기능이 아니다.

`train_model.py`와 `backtest_signals.py`는 `--use-historical` 또는 `--dataset-path` 옵션으로 historical dataset을 입력받을 수 있다.

## 모델 실험 관리

자동매매로 넘어가기 전에는 단일 학습 결과보다 feature group과 모델 설정을 비교하는 실험 관리가 중요하다. `experiment_models.py`는 price, flow, DART, Naver, Yahoo, research, account 그룹 조합별로 walk-forward 검증을 수행해 어떤 데이터가 실제 예측력에 기여하는지 비교한다.

모델 연구 파이프라인은 무거운 작업이므로 daily `run_full_pipeline.py`에 넣지 않는다. 수동 실행 또는 주 1회 실행을 권장한다.

```bash
cd /Users/eomjiyong/policy-research
/Users/eomjiyong/policy-research/.venv/bin/python scripts/run_model_research_pipeline.py
```

개별 실행:

```bash
/Users/eomjiyong/policy-research/.venv/bin/python scripts/experiment_models.py --use-historical --min-labeled-rows 300 --min-dates 30 --n-splits 3
/Users/eomjiyong/policy-research/.venv/bin/python scripts/calibrate_model.py --use-historical --method sigmoid --min-labeled-rows 300
/Users/eomjiyong/policy-research/.venv/bin/python scripts/explain_model.py --use-historical --top-n 30
/Users/eomjiyong/policy-research/.venv/bin/python scripts/promote_model.py --dry-run --min-precision-at-3 0.55 --min-top3-mean-return 0.0 --max-drawdown-top3 -0.05
```

생성 파일:

- `reports/experiment_results.json`
- `reports/experiment_report.md`
- `reports/calibration_report.json`
- `reports/calibration_report.md`
- `reports/model_explainability.json`
- `reports/model_explainability.md`
- `reports/model_promotion_report.md`
- `reports/model_research_pipeline_report.md`
- `models/experiments/`
- `models/active/`, 승격 시

Calibration은 `outperform_prob_5d`가 실제 확률에 가까운지 확인하고 필요하면 sigmoid 또는 isotonic 보정을 적용하는 과정이다. `explain_model.py`는 SHAP을 사용할 수 있으면 SHAP 기반 설명성을 만들고, 불가능하면 permutation importance 또는 모델 내장 importance로 fallback한다. `promote_model.py`는 precision@3, top3 평균 수익률, drawdown, excess return 기준을 통과한 모델만 active 경로로 승격한다. `--dry-run`은 파일 복사 없이 승격 가능성만 검토하는 안전한 기본 운영 방식이다.

active model이 생겨도 ML 신호는 주문 확정이 아니라 order proposal의 보조 점수다. 모든 주문 후보는 계속 `risk_guard.py`를 통과해야 하며 실제 주문 실행은 비활성화 상태다.

## ML 의존성 점검과 Walk-Forward 백테스트

`model_status=no_op_dependency_unavailable`이 나오면 먼저 현재 `.venv`에서 ML 패키지와 macOS runtime을 점검한다.

```bash
cd /Users/eomjiyong/policy-research
/Users/eomjiyong/policy-research/.venv/bin/python scripts/check_ml_dependencies.py
```

생성 파일:

- `reports/ml_dependency_check.json`
- `reports/ml_dependency_check.md`

필요 패키지 설치:

```bash
/Users/eomjiyong/policy-research/.venv/bin/pip install pandas numpy scikit-learn xgboost joblib shap
```

`shap`은 설명성 분석용 optional dependency다. macOS에서 XGBoost가 `libomp.dylib`를 찾지 못하면 `brew install libomp`가 필요할 수 있다.

Historical dataset 기준 학습:

```bash
/Users/eomjiyong/policy-research/.venv/bin/python scripts/train_model.py --use-historical --min-labeled-rows 100 --min-dates 10 --n-splits 3
/Users/eomjiyong/policy-research/.venv/bin/python scripts/predict_signals.py
```

Rule 기반 백테스트와 ML walk-forward 백테스트는 다르다. Rule 백테스트는 저장된 `final_score`로 날짜별 top-k를 고르고, ML walk-forward는 각 validation fold 이전 데이터로 모델을 다시 학습한 뒤 미래 validation 구간에만 예측을 붙인다.

```bash
/Users/eomjiyong/policy-research/.venv/bin/python scripts/backtest_signals.py --use-historical --score-column final_score --target-column future_return_5d --top-k 3 --min-labeled-rows 100 --fee-bps 15 --slippage-bps 10
/Users/eomjiyong/policy-research/.venv/bin/python scripts/backtest_signals.py --use-historical --ml-walk-forward --top-k 3 --min-labeled-rows 100 --fee-bps 15 --slippage-bps 10
```

`top_k_mean_return`은 gross 기준이고, `top_k_mean_net_return`과 `excess_net_return`은 fee/slippage 단순 추정 비용을 차감한 지표다. 실제 세금, 수수료, 체결 슬리피지는 다를 수 있으므로 자동주문 판단에는 net 지표와 리스크 가드 결과를 함께 본다. 백테스트가 좋아도 주문 실행은 계속 비활성화 상태다.

현재 검증 해석은 보수적으로 유지한다. Rule `final_score`는 full-period excess net return이 양수이고 표본이 더 크다. ML walk-forward는 hit rate와 top-k net return은 좋아 보이지만 validation window가 작고, excess net return이 rule baseline을 확실히 압도하지 못하면 primary signal로 쓰지 않는다.

Signal policy 평가:

```bash
/Users/eomjiyong/policy-research/.venv/bin/python scripts/backtest_signals.py --use-historical --ml-walk-forward --top-k 3 --min-labeled-rows 100 --fee-bps 15 --slippage-bps 10
/Users/eomjiyong/policy-research/.venv/bin/python scripts/evaluate_signal_policy.py
```

생성 파일:

- `reports/signal_policy.json`
- `reports/signal_policy_report.md`

기본 정책은 `rule-first, ML modifier`다. fallback 모델이거나 calibration이 없으면 `rule_weight=0.75`, `ml_weight=0.25`를 사용한다. 자동매매 전에는 다음 조건을 통과해야 한다.

- same-window ensemble excess net return이 rule보다 우수
- selected_rows >= 100
- validation dates >= 30
- calibration 완료
- drawdown 기준 통과
- paper-trading 2주 이상 통과

## Calibration과 Paper Trading Evaluation

현재 신호 정책은 `rule_first_ml_modifier`이며 ML은 primary signal이 아니다. ML 확률은 calibration 전에는 과신하지 않고, fallback 또는 uncalibrated 모델이면 `strong_buy_candidate`를 제한하고 `probability_warning`을 남긴다.

Calibration 실행:

```bash
/Users/eomjiyong/policy-research/.venv/bin/python scripts/calibrate_model.py --use-historical --method sigmoid --min-labeled-rows 300 --min-dates 30 --n-bins 10
/Users/eomjiyong/policy-research/.venv/bin/python scripts/predict_signals.py
```

Calibration은 historical labeled data를 시간순으로 train/calibration/test로 나누고, 보정 전후의 `log_loss`, `brier_score`, `roc_auc`, `average_precision`, `precision_at_3`, `top3_mean_future_return_5d`를 비교한다. 개선될 때만 `models/active/calibrated_outperform_5d.joblib`을 저장한다.

Paper trading은 실제 주문 없이 후보를 추적하는 단계다.

```bash
/Users/eomjiyong/policy-research/.venv/bin/python scripts/generate_order_proposals.py --max-buy-candidates 3 --max-sell-candidates 3 --max-order-amount 1000000
/Users/eomjiyong/policy-research/.venv/bin/python scripts/log_paper_trades.py --max-candidates 5 --only-approved-for-review
/Users/eomjiyong/policy-research/.venv/bin/python scripts/evaluate_paper_trades.py --horizon-days 5 --fee-bps 15 --slippage-bps 10
```

생성 파일:

- `data/paper_trades.jsonl`
- `reports/paper_trades_snapshot.md`
- `reports/paper_trading_report.json`
- `reports/paper_trading_report.md`

승인형 모의주문으로 넘어가기 전 최소 조건:

- selected_rows >= 100
- validation date_count >= 30
- calibration improved 또는 ML weight <= 0.25
- paper trades labeled >= 30
- mean_net_return_5d > 0
- hit_rate_5d >= 0.55
- avg_max_drawdown_5d > -0.05

이 조건을 만족해도 자동매매는 바로 켜지지 않는다. 다음 단계는 별도 승인형 모의주문이며, 현재 daily pipeline은 paper trade logging/evaluation까지만 수행한다.

## 주문 후보 생성

주문 후보 생성은 추천 리포트를 바탕으로 매수/매도 검토 후보만 만들며 실제 주문 실행은 비활성화되어 있다. `generate_order_proposals.py`와 `risk_guard.py`는 키움 주문 API 또는 주문 TR을 호출하지 않는다.

수동 실행:

```bash
cd /Users/eomjiyong/policy-research
/Users/eomjiyong/policy-research/.venv/bin/python scripts/generate_order_proposals.py --max-buy-candidates 3 --max-sell-candidates 3 --max-order-amount 1000000 --min-final-score 70 --min-price-score 55 --min-flow-score 55
```

VS Code에서는 `Orders - Generate Proposals` launch 메뉴를 사용한다.

생성 파일:

- `reports/order_proposals.json`
- `reports/order_proposals.md`
- `reports/order_risk_check.json`
- `reports/order_risk_check.md`
- `data/order_ledger.jsonl`

`risk_guard.py`는 mock mode, proposal_only, 하루 신규 진입 한도, 종목/섹터 비중, 미체결/주문대기 금액, 현금, 가격 추세/변동성, 수급, 공시/뉴스/Yahoo 리스크, 보유 여부, 중복 후보를 검사한다. `reports/order_proposals.md`와 `reports/order_risk_check.md`에서 후보와 reject 사유를 확인한다.

다음 단계는 Telegram 승인형 `execute_approved_mock_order.py`를 별도로 만드는 것이며, 그 전까지 Telegram 승인 문구는 예시로만 표시한다.

## 주의사항

- 아래 주문 스크립트는 명시 요청 전까지 실행하지 않는다.
- `scripts/mock_order_test.py`
- `scripts/kiwoom_smart_buy.py`
- `scripts/kiwoom_smart_sell.py`
- `scripts/kiwoom_cancel_pending_order.py`

추천 리포트는 투자 판단 보조 자료이며 주문 실행 지시가 아니다.
