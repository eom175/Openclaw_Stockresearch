import os
from dataclasses import dataclass

from dotenv import load_dotenv

from policylink.paths import PROJECT_ROOT


@dataclass(frozen=True)
class KiwoomSettings:
    base_url: str
    app_key: str
    secret_key: str


@dataclass(frozen=True)
class NaverSettings:
    client_id: str
    client_secret: str


def load_kiwoom_settings() -> KiwoomSettings:
    load_dotenv(PROJECT_ROOT / ".env")

    base_url = os.getenv("KIWOOM_BASE_URL", "https://mockapi.kiwoom.com").rstrip("/")
    app_key = os.getenv("KIWOOM_APP_KEY")
    secret_key = os.getenv("KIWOOM_SECRET_KEY")

    if not app_key or not secret_key:
        raise RuntimeError("KIWOOM_APP_KEY or KIWOOM_SECRET_KEY is not configured.")

    return KiwoomSettings(base_url=base_url, app_key=app_key, secret_key=secret_key)


def load_dart_api_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("DART_API_KEY")
    if not api_key:
        raise RuntimeError("DART_API_KEY is not configured.")

    return api_key


def load_naver_settings() -> NaverSettings:
    load_dotenv(PROJECT_ROOT / ".env")

    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("NAVER_CLIENT_ID or NAVER_CLIENT_SECRET is not configured.")

    return NaverSettings(client_id=client_id, client_secret=client_secret)
