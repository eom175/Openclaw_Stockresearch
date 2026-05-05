#!/usr/bin/env python3
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("KIWOOM_BASE_URL", "https://mockapi.kiwoom.com")
APP_KEY = os.getenv("KIWOOM_APP_KEY")
SECRET_KEY = os.getenv("KIWOOM_SECRET_KEY")


def get_access_token() -> str:
    url = f"{BASE_URL}/oauth2/token"
    payload = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "secretkey": SECRET_KEY,
    }
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
    }

    res = requests.post(url, headers=headers, json=payload, timeout=15)
    res.raise_for_status()

    data = res.json()
    token = data.get("token")
    if not token:
        raise RuntimeError(f"토큰 발급 실패: {data}")

    return token


def get_account_number(token: str):
    url = f"{BASE_URL}/api/dostk/acnt"

    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "api-id": "ka00001",
    }

    # 계좌번호조회 ka00001은 요청 body가 비어 있습니다.
    res = requests.post(url, headers=headers, json={}, timeout=15)
    print("status:", res.status_code)
    print(json.dumps(res.json(), ensure_ascii=False, indent=2))


def main():
    token = get_access_token()
    get_account_number(token)


if __name__ == "__main__":
    main()
