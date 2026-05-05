#!/usr/bin/env python3
from _bootstrap import add_project_paths

add_project_paths()

from policylink.kiwoom.client import KiwoomRestClient
from policylink.utils import redact_sensitive, safe_body


def main():
    client = KiwoomRestClient()
    result = client.post(
        endpoint="/api/dostk/acnt",
        api_id="ka00001",
        data={},
    )
    body = safe_body(result)
    print("ACCOUNT status:", result.get("status_code"))
    print("return_code:", body.get("return_code"))
    print("return_msg:", body.get("return_msg"))
    print("body_keys:", sorted(redact_sensitive(body).keys()))
    print("account_number_value: [REDACTED]")


if __name__ == "__main__":
    main()
