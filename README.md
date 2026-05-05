# Policy Research

OpenClaw + Telegram + Kiwoom 모의투자 환경에서 국내주식/국내상장 ETF 중심 리서치, 가격/수급 동기화, 포트폴리오 추천 리포트, 학습 데이터셋 스냅샷을 만드는 프로젝트다.

## 실행 환경

- 프로젝트 경로: `/Users/eomjiyong/policy-research`
- Python: `/Users/eomjiyong/policy-research/.venv/bin/python`
- 자동화: OpenClaw Gateway + Cron
- 개발/디버깅: VS Code launch configurations
- 거래 환경: Kiwoom REST API 모의투자

`.env`에는 Kiwoom/OpenDART 접속 정보가 들어가므로 열람하거나 출력하지 않는다.

## 전체 파이프라인

```bash
cd /Users/eomjiyong/policy-research
/Users/eomjiyong/policy-research/.venv/bin/python scripts/run_full_pipeline.py
```

`scripts/run_full_pipeline.py`는 아래 순서를 실행하고 `reports/full_pipeline_report.md`에 단계별 성공/실패를 기록한다.

1. `collect_research.py --hours 720 --max-items 50`
2. `collect_dart.py --days 90 --max-stocks 10 --sleep 0.3`
3. `kiwoom_daily_report.py`
4. `sync_prices.py --max-stocks 10 --sleep 0.7`
5. `sync_flows.py --max-stocks 5 --sleep 1.2`
6. `recommend_portfolio.py`
7. `build_dataset.py`
8. `label_dataset.py`

주문 실행 스크립트는 전체 파이프라인에 포함하지 않는다. DART 수집 실패는 warning으로 기록하고 가격/수급/추천 파이프라인은 계속 진행한다.

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
- `reports/kiwoom_mock_account_summary.json`
- `data/price_features.json`
- `data/flow_features.json`
- `reports/dart_sync_diagnostic.json`
- `reports/dart_disclosures.md`
- `reports/portfolio_recommendation.md`
- `data/model_dataset.jsonl`
- `data/model_dataset.csv`
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

## 주의사항

- 아래 주문 스크립트는 명시 요청 전까지 실행하지 않는다.
- `scripts/mock_order_test.py`
- `scripts/kiwoom_smart_buy.py`
- `scripts/kiwoom_smart_sell.py`
- `scripts/kiwoom_cancel_pending_order.py`

추천 리포트는 투자 판단 보조 자료이며 주문 실행 지시가 아니다.
