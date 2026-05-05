#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from _bootstrap import add_src_to_path

add_src_to_path()

from policylink.paths import ORDER_LEDGER_PATH
from policylink.utils import load_jsonl, redact_sensitive


KST = timezone(timedelta(hours=9))


def today_yyyymmdd() -> str:
    return datetime.now(KST).strftime("%Y%m%d")


def append_ledger_event(event: Dict[str, Any]) -> None:
    safe_event = redact_sensitive(dict(event))
    safe_event.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    safe_event.setdefault("order_enabled", False)
    safe_event.setdefault("execution_status", "not_executed")

    ORDER_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ORDER_LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe_event, ensure_ascii=False) + "\n")


def load_ledger_events() -> List[Dict[str, Any]]:
    return load_jsonl(ORDER_LEDGER_PATH)


def today_existing_proposals(snapshot_date: Optional[str] = None) -> List[Dict[str, Any]]:
    date_text = snapshot_date or today_yyyymmdd()
    events = []
    for event in load_ledger_events():
        if event.get("event_type") != "proposal":
            continue
        proposal_id = str(event.get("proposal_id") or "")
        created_date = str(event.get("snapshot_date") or "")
        if proposal_id.startswith(date_text) or created_date == date_text:
            events.append(event)
    return events


def make_proposal_id(prefix: Optional[str] = None, snapshot_date: Optional[str] = None, sequence: Optional[int] = None) -> str:
    date_text = snapshot_date or today_yyyymmdd()
    if sequence is None:
        sequence = len(today_existing_proposals(date_text)) + 1

    if prefix:
        return f"{date_text}-{prefix}-{sequence:03d}"
    return f"{date_text}-{sequence:03d}"


def main() -> int:
    events = load_ledger_events()
    print(f"ledger_path={ORDER_LEDGER_PATH}")
    print(f"event_count={len(events)}")
    print("order_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
