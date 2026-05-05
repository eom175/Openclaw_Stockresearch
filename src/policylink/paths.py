from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
RAW_DIR = PROJECT_ROOT / "raw"

CANDIDATES_PATH = DATA_DIR / "candidates.compact.json"
PRICES_DAILY_PATH = DATA_DIR / "prices_daily.json"
PRICE_FEATURES_PATH = DATA_DIR / "price_features.json"
FLOWS_DAILY_PATH = DATA_DIR / "flows_daily.json"
FLOW_FEATURES_PATH = DATA_DIR / "flow_features.json"
DART_CORP_CODES_PATH = DATA_DIR / "dart_corp_codes.json"
DART_DISCLOSURES_PATH = DATA_DIR / "dart_disclosures.json"
DART_EVENT_FEATURES_PATH = DATA_DIR / "dart_event_features.json"
NAVER_NEWS_PATH = DATA_DIR / "naver_news.json"
NEWS_EVENT_FEATURES_PATH = DATA_DIR / "news_event_features.json"
MODEL_DATASET_JSONL_PATH = DATA_DIR / "model_dataset.jsonl"
MODEL_DATASET_CSV_PATH = DATA_DIR / "model_dataset.csv"

DAILY_FEATURES_PATH = REPORTS_DIR / "daily_features.json"
DAILY_BRIEF_PATH = REPORTS_DIR / "daily_brief.md"
KIWOOM_ACCOUNT_RAW_PATH = REPORTS_DIR / "kiwoom_mock_account_raw.json"
KIWOOM_ACCOUNT_SUMMARY_PATH = REPORTS_DIR / "kiwoom_mock_account_summary.json"
KIWOOM_ACCOUNT_REPORT_PATH = REPORTS_DIR / "kiwoom_mock_account_report.md"
PORTFOLIO_RECOMMENDATION_JSON_PATH = REPORTS_DIR / "portfolio_recommendation.json"
PORTFOLIO_RECOMMENDATION_MD_PATH = REPORTS_DIR / "portfolio_recommendation.md"
MODEL_DATASET_SNAPSHOT_PATH = REPORTS_DIR / "model_dataset_snapshot.md"
LABEL_DATASET_REPORT_PATH = REPORTS_DIR / "label_dataset_report.md"
PRICE_SYNC_DIAGNOSTIC_PATH = REPORTS_DIR / "price_sync_diagnostic.json"
FLOW_SYNC_DIAGNOSTIC_PATH = REPORTS_DIR / "flow_sync_diagnostic.json"
FLOW_FEATURES_REPORT_PATH = REPORTS_DIR / "flow_features.md"
DART_SYNC_DIAGNOSTIC_PATH = REPORTS_DIR / "dart_sync_diagnostic.json"
DART_DISCLOSURES_REPORT_PATH = REPORTS_DIR / "dart_disclosures.md"
NAVER_NEWS_SYNC_DIAGNOSTIC_PATH = REPORTS_DIR / "naver_news_sync_diagnostic.json"
NAVER_NEWS_FEATURES_REPORT_PATH = REPORTS_DIR / "naver_news_features.md"
FULL_PIPELINE_REPORT_PATH = REPORTS_DIR / "full_pipeline_report.md"


def ensure_project_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
