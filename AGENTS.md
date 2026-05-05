# Policy Research Agent Rules

이 프로젝트는 OpenClaw + Telegram + Kiwoom 모의투자 + 국내투자 리서치/추천 파이프라인을 운영한다.

## 보안 규칙

- `.env` 파일을 열람하거나 출력하지 않는다.
- API key, secret, access token, authorization header, 계좌번호 원문을 터미널/리포트/문서에 출력하지 않는다.
- `DART_API_KEY`는 절대 출력하지 않는다.
- Kiwoom 클라이언트와 파이프라인 리포트는 민감 키를 마스킹해야 한다.
- 현재 환경은 키움증권 모의투자 REST API만 사용한다.
- 공시 원문 전체를 reports에 복사하지 않는다.
- 공시 수집은 compact metadata와 feature만 저장한다.

## 주문 실행 금지

아래 스크립트는 사용자가 명시적으로 요청하기 전까지 실행하지 않는다.

- `scripts/mock_order_test.py`
- `scripts/kiwoom_smart_buy.py`
- `scripts/kiwoom_smart_sell.py`
- `scripts/kiwoom_cancel_pending_order.py`

추천 시스템은 주문을 만들거나 실행하지 않고, recommendation report 생성까지만 담당한다. `scripts/run_full_pipeline.py`에도 주문 스크립트를 포함하지 않는다.

## 운영 구조

- 자동화는 OpenClaw Gateway + Cron이 담당한다.
- VS Code는 개발/수동 디버깅용이다.
- 실행 entrypoint는 `scripts/`에 유지하고, 구현 로직은 `src/policylink/`에 둔다.
- 공통 경로는 `src/policylink/paths.py`, 공통 유틸은 `src/policylink/utils.py`, 투자 universe는 `src/policylink/universe.py`를 우선 사용한다.
