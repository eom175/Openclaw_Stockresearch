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

주문 실행 스크립트는 전체 파이프라인에 포함하지 않는다. 주문 후보 생성 단계도 `proposal_only=true`, `order_enabled=false`이며 키움 주문 API를 호출하지 않는다. DART/뉴스/Yahoo 수집 실패, ML 학습 no-op, ML 예측 no_model은 warning으로 기록하고 파이프라인은 계속 진행한다.

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

## ML 학습과 예측 신호

XGBoost 기반 ML 신호와 모델 품질 검증을 사용하려면 `xgboost`, `scikit-learn`, `pandas`, `numpy`, `joblib`이 필요하다.

```bash
cd /Users/eomjiyong/policy-research
/Users/eomjiyong/policy-research/.venv/bin/pip install xgboost scikit-learn pandas numpy joblib
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
