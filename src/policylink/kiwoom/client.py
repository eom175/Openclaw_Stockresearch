import json
from datetime import datetime, timezone

import requests
from policylink.config import load_kiwoom_settings
from policylink.utils import redact_sensitive


class KiwoomRestClient:
    def __init__(self, verbose: bool = False):
        self.settings = load_kiwoom_settings()
        self.base_url = self.settings.base_url
        self.token = None
        self.verbose = verbose

    def get_access_token(self) -> str:
        if self.token:
            return self.token

        url = f"{self.base_url}/oauth2/token"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.settings.app_key,
            "secretkey": self.settings.secret_key,
        }
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
        }

        res = requests.post(url, headers=headers, json=payload, timeout=15)
        self._print_response("TOKEN", res)

        res.raise_for_status()
        data = res.json()

        token = data.get("token")
        if not token:
            raise RuntimeError(f"Token issue failed: {redact_sensitive(data)}")

        self.token = token
        return token

    def post(self, endpoint: str, api_id: str, data: dict, cont_yn: str = "N", next_key: str = "") -> dict:
        token = self.get_access_token()

        url = f"{self.base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {token}",
            "cont-yn": cont_yn,
            "next-key": next_key,
            "api-id": api_id,
        }

        res = requests.post(url, headers=headers, json=data, timeout=15)
        self._print_response(api_id, res)

        try:
            body = res.json()
        except Exception:
            body = {"raw_text": res.text}

        return {
            "api_id": api_id,
            "status_code": res.status_code,
            "headers": {
                "api-id": res.headers.get("api-id"),
                "cont-yn": res.headers.get("cont-yn"),
                "next-key": res.headers.get("next-key"),
            },
            "body": body,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }

    def _print_response(self, label: str, res: requests.Response):
        if not self.verbose:
            return

        print(f"\n===== {label} =====")
        print("status:", res.status_code)
        print("headers:", {
            "api-id": res.headers.get("api-id"),
            "cont-yn": res.headers.get("cont-yn"),
            "next-key": res.headers.get("next-key"),
        })
        try:
            print(json.dumps(redact_sensitive(res.json()), ensure_ascii=False, indent=2)[:3000])
        except Exception:
            print("[non-json response omitted]")
