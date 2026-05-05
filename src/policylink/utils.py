import csv
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SENSITIVE_KEY_RE = re.compile(
    r"(token|secret|app[_-]?key|authorization|auth|password|account|acct|acnt)",
    re.IGNORECASE,
)


def parse_number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace(",", "").replace("%", "")
    if text == "":
        return default

    sign = -1 if text.startswith("-") else 1
    text = text.replace("+", "").replace("-", "")
    cleaned = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    if cleaned == "":
        return default

    try:
        return sign * float(cleaned)
    except ValueError:
        return default


def parse_kiwoom_int(value: Any, default: int = 0) -> int:
    return int(parse_number(value, float(default)))


def parse_kiwoom_float(value: Any, default: float = 0.0) -> float:
    return parse_number(value, default)


def parse_int(value: Any, default: int = 0) -> int:
    return int(parse_number(value, float(default)))


def parse_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(parse_number(value, default if default is not None else 0.0))
    except Exception:
        return default


def normalize_code(value: Any, empty: str = "") -> str:
    if value is None:
        return empty
    text = str(value).strip()
    if text.startswith("A") and len(text) == 7:
        return text[1:]
    return text


def pick_first(body: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    if not isinstance(body, dict):
        return default
    for key in keys:
        if key in body:
            return body.get(key)
    return default


def find_first_list(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict):
            return obj
        return []
    if isinstance(obj, dict):
        for value in obj.values():
            found = find_first_list(value)
            if found:
                return found
    return []


def safe_body(result: Dict[str, Any]) -> Dict[str, Any]:
    body = result.get("body", {}) if isinstance(result, dict) else {}
    return body if isinstance(body, dict) else {}


def get_return_status(result: Dict[str, Any]) -> Dict[str, Any]:
    body = safe_body(result)
    return {
        "status_code": result.get("status_code") if isinstance(result, dict) else None,
        "return_code": body.get("return_code"),
        "return_msg": body.get("return_msg"),
    }


def get_kst_today_yyyymmdd() -> str:
    kst = timezone(timedelta(hours=9))
    return datetime.now(kst).strftime("%Y%m%d")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return default


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if SENSITIVE_KEY_RE.search(str(key)) else redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value
