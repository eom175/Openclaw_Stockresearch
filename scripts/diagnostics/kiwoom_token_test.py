#!/usr/bin/env python3
from _bootstrap import add_project_paths

add_project_paths()

from policylink.kiwoom.client import KiwoomRestClient


def main():
    client = KiwoomRestClient()
    client.get_access_token()
    print("TOKEN status: ok")
    print("token_value: [REDACTED]")


if __name__ == "__main__":
    main()
