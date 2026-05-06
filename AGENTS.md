# Policy Research Agent Rules

이 프로젝트는 OpenClaw + Telegram + Kiwoom 모의투자 + 국내투자 리서치/추천 파이프라인을 운영한다.

## 보안 규칙

- `.env` 파일을 열람하거나 출력하지 않는다.
- API key, secret, access token, authorization header, 계좌번호 원문을 터미널/리포트/문서에 출력하지 않는다.
- `DART_API_KEY`는 절대 출력하지 않는다.
- `NAVER_CLIENT_ID`와 `NAVER_CLIENT_SECRET`은 절대 출력하지 않는다.
- Kiwoom 클라이언트와 파이프라인 리포트는 민감 키를 마스킹해야 한다.
- 현재 환경은 키움증권 모의투자 REST API만 사용한다.
- 공시 원문 전체를 reports에 복사하지 않는다.
- 공시 수집은 compact metadata와 feature만 저장한다.
- 뉴스 원문 전체를 reports에 복사하지 않는다.
- 뉴스 수집은 제목/요약/URL/날짜와 compact feature만 저장한다.
- 기사 본문 scraping은 하지 않는다.
- Yahoo Finance 뉴스 원문 전체를 reports에 복사하지 않는다.
- yfinance 데이터는 내부 연구/교육 목적 compact feature로만 사용한다.
- Yahoo raw news body를 장기 저장하지 않고 title/summary/url/time/ticker 중심으로 저장한다.
- 데이터 사용 권리/약관은 사용자가 별도로 확인해야 하며, 이 코드는 데이터 재배포를 목적으로 하지 않는다.

## 주문 실행 금지

아래 스크립트는 사용자가 명시적으로 요청하기 전까지 실행하지 않는다.

- `scripts/mock_order_test.py`
- `scripts/kiwoom_smart_buy.py`
- `scripts/kiwoom_smart_sell.py`
- `scripts/kiwoom_cancel_pending_order.py`

추천 시스템은 주문을 만들거나 실행하지 않고, recommendation report 생성까지만 담당한다. `scripts/run_full_pipeline.py`에도 주문 스크립트를 포함하지 않는다.

## 주문 후보 생성 규칙

- `scripts/generate_order_proposals.py`와 `scripts/risk_guard.py`는 주문 후보와 리스크 검사만 수행한다.
- 키움 주문 API 및 주문 TR 호출은 금지한다.
- 자동매매는 아직 비활성화 상태다.
- 모든 주문 후보 output은 `order_enabled=false` 또는 `proposal_only=true`를 명시한다.
- `execute_approved_mock_order.py`가 생기기 전까지 Telegram 승인 문구는 예시로만 표시한다.

## ML 신호 규칙

- `scripts/train_model.py`와 `scripts/predict_signals.py`는 주문을 실행하지 않는다.
- `scripts/experiment_models.py`, `scripts/calibrate_model.py`, `scripts/explain_model.py`, `scripts/promote_model.py`는 주문을 실행하지 않는다.
- ML 신호는 매수/매도 확정이 아니라 후보 scoring 보조 지표다.
- ML 모델은 주문 실행자가 아니라 scoring engine이다.
- `scripts/backtest_signals.py`는 주문을 실행하지 않는다.
- `scripts/promote_model.py`는 모델 파일 승격만 담당한다.
- active model이 있어도 `risk_guard.py` 없이 주문 후보를 강화하지 않는다.
- 모델 성능이 낮거나 데이터가 부족하면 주문 후보를 강화하지 않는다.
- leakage 의심 피처는 학습에서 제외한다.
- 모델 실험에서 leakage 의심 피처가 있으면 active model 승격을 금지한다.
- 모델 성능이 기준 미달이면 active model로 승격하지 않는다.
- dependency 진단과 설치 가이드는 가능하지만 `.env` 또는 credential을 읽거나 출력하지 않는다.
- ML walk-forward 백테스트는 모델 검증용이며 주문 실행과 무관하다.
- 백테스트 결과가 좋아도 자동주문을 켜지 않는다.
- 주문 실행은 별도 승인형 단계에서만 다룬다.
- ML 신호가 생성되어도 자동주문을 실행하지 않는다.
- `signal_policy`가 `auto_order_allowed=false`이면 주문 후보만 생성한다.
- model promotion이 rejected 또는 dry-run rejected이면 ML을 primary signal로 쓰지 않는다.
- `calibrated=false`인 확률은 투자 판단에 과신하지 않는다.
- 모델 파일, 리포트, 로그에 credential, access token, 계좌번호 원문을 저장하지 않는다.
- labeled row가 부족한 학습 no-op과 모델이 없는 예측 no_model은 정상 상태로 처리한다.

## Historical Backfill 규칙

- historical dataset 생성 시 미래 데이터 누수를 금지한다.
- 각 `snapshot_date`의 feature에는 `snapshot_date` 이후 데이터를 쓰지 않는다.
- `future_*` 컬럼은 target 전용이며 학습 feature로 사용하지 않는다.
- Naver/Yahoo/DART raw 원문 전체를 장기 저장하지 않는다.
- backfill 과정에서도 credential, access token, 계좌번호 원문을 출력하거나 리포트에 저장하지 않는다.
- historical backfill은 주문 실행과 무관하며 Kiwoom 주문 TR을 호출하지 않는다.

## 운영 구조

- 자동화는 OpenClaw Gateway + Cron이 담당한다.
- VS Code는 개발/수동 디버깅용이다.
- 실행 entrypoint는 `scripts/`에 유지하고, 구현 로직은 `src/policylink/`에 둔다.
- 공통 경로는 `src/policylink/paths.py`, 공통 유틸은 `src/policylink/utils.py`, 투자 universe는 `src/policylink/universe.py`를 우선 사용한다.
