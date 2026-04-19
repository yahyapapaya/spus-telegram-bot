import csv
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import time
from typing import Dict, List, Optional, Set, Tuple

import requests

SPUS_URL = "https://www.sp-funds.com/wp-content/uploads/data/TidalFG_Holdings_SPUS.csv"
TICKER_COLUMN = "StockTicker"
WEIGHT_COLUMN = "Weightings"
TICKER_RE = re.compile(r"^[A-Z]{1,6}$")

DEFAULT_WEIGHT_THRESHOLD = 0.10
DEFAULT_TOP_MOVERS = 15


def log(message: str) -> None:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {message}")


def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def db_path() -> str:
    return os.path.join(script_dir(), "spus_snapshots.db")


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log(f"Invalid {name}='{raw}'. Falling back to default {default}.")
        return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        log(f"Invalid {name}='{raw}'. Falling back to default {default}.")
        return default
    if value <= 0:
        log(f"Invalid {name}='{raw}'. Falling back to default {default}.")
        return default
    return value


def parse_weight_to_pct(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None

    raw = str(value).strip().replace(",", "")
    if not raw:
        return None

    has_percent = raw.endswith("%")
    if has_percent:
        raw = raw[:-1].strip()

    try:
        numeric = float(raw)
    except ValueError:
        return None

    if has_percent:
        return numeric
    return numeric if numeric > 1 else numeric * 100


def fetch_csv(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.sp-funds.com/",
        "Connection": "keep-alive",
    }

    backoffs = [2, 4, 8]
    last_error: Optional[Exception] = None

    for attempt in range(1, len(backoffs) + 2):
        try:
            log(f"Downloading SPUS holdings CSV (attempt {attempt})")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            if not response.text.strip():
                raise RuntimeError("Downloaded file is empty")
            return response.text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            log(f"Download failed: {exc}")
            if attempt <= len(backoffs):
                time.sleep(backoffs[attempt - 1])

    raise RuntimeError(f"Failed to download CSV after retries: {last_error}")


def parse_holdings(csv_text: str) -> Tuple[Set[str], Dict[str, float]]:
    rows = csv_text.lstrip("\ufeff").splitlines()
    reader = csv.DictReader(rows)

    if not reader.fieldnames:
        raise RuntimeError("CSV appears malformed: missing headers")

    missing = [col for col in (TICKER_COLUMN, WEIGHT_COLUMN) if col not in reader.fieldnames]
    if missing:
        raise RuntimeError(
            f"CSV missing expected columns {missing}. Got headers: {reader.fieldnames}"
        )

    tickers: Set[str] = set()
    weights: Dict[str, float] = {}

    for row in reader:
        ticker = (row.get(TICKER_COLUMN) or "").strip().upper()
        if not ticker or not TICKER_RE.match(ticker):
            continue

        tickers.add(ticker)

        weight_pct = parse_weight_to_pct(row.get(WEIGHT_COLUMN))
        if weight_pct is not None:
            weights[ticker] = weight_pct

    if not tickers:
        raise RuntimeError("Parsed zero tickers from CSV; upstream format may have changed")

    log(f"Parsed {len(tickers)} unique tickers")
    log(f"Parsed {len(weights)} tickers with valid weights")
    return tickers, weights


def db_connect_and_migrate() -> Tuple[sqlite3.Connection, str]:
    conn = sqlite3.connect(db_path())

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshots (
            snap_date TEXT PRIMARY KEY,
            tickers_json TEXT NOT NULL,
            csv_sha256 TEXT NOT NULL,
            weights_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()

    cols = [row[1] for row in conn.execute("PRAGMA table_info(snapshots)").fetchall()]

    hash_col = "csv_sha256"
    if "csv_sha256" not in cols and "csv_hash" in cols:
        hash_col = "csv_hash"
    elif "csv_sha256" not in cols:
        log("Migrating DB: adding csv_sha256")
        conn.execute("ALTER TABLE snapshots ADD COLUMN csv_sha256 TEXT NOT NULL DEFAULT ''")
        conn.commit()

    cols = [row[1] for row in conn.execute("PRAGMA table_info(snapshots)").fetchall()]
    if "weights_json" not in cols:
        log("Migrating DB: adding weights_json")
        conn.execute("ALTER TABLE snapshots ADD COLUMN weights_json TEXT")
        conn.commit()

    return conn, hash_col


def get_latest_snapshot(
    conn: sqlite3.Connection,
    hash_col: str,
) -> Optional[Tuple[str, Set[str], str, Dict[str, float]]]:
    row = conn.execute(
        f"SELECT snap_date, tickers_json, {hash_col}, weights_json FROM snapshots ORDER BY snap_date DESC LIMIT 1"
    ).fetchone()

    if not row:
        return None

    snap_date = row[0]
    tickers = set(json.loads(row[1]))
    csv_hash = row[2] or ""
    weights_json = row[3]
    weights: Dict[str, float] = json.loads(weights_json) if weights_json else {}
    return snap_date, tickers, csv_hash, {k: float(v) for k, v in weights.items()}


def save_snapshot(
    conn: sqlite3.Connection,
    hash_col: str,
    snap_date: str,
    tickers: Set[str],
    csv_hash: str,
    weights: Dict[str, float],
) -> None:
    conn.execute(
        f"""
        INSERT OR REPLACE INTO snapshots (snap_date, tickers_json, {hash_col}, weights_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            snap_date,
            json.dumps(sorted(tickers)),
            csv_hash,
            json.dumps(weights, sort_keys=True),
            utc_now_iso(),
        ),
    )
    conn.commit()


def fmt_pct(delta: float) -> str:
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.2f}%"


def build_message(
    date: str,
    weight_threshold: float,
    added: Set[str],
    removed: Set[str],
    movers: List[Tuple[str, float]],
) -> str:
    lines = [f"SPUS Holdings Update ({date})", ""]

    lines.append("Stocks Removed from Holdings:")
    lines.extend(sorted(removed) if removed else ["(none)"])
    lines.append("")

    lines.append("Stocks Loaded into Holdings:")
    lines.extend(sorted(added) if added else ["(none)"])
    lines.append("")

    lines.append(f"Weight Changes (>|{weight_threshold:.2f}%|):")
    if movers:
        for ticker, delta in movers:
            direction = "increased" if delta > 0 else "decreased"
            lines.append(f"{ticker} weight {direction} {fmt_pct(delta)}")
    else:
        lines.append("(none)")

    return "\n".join(lines)


def send_telegram(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()


def main() -> None:
    token = os.getenv("TG_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TG_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("TG_BOT_TOKEN and TG_CHAT_ID must be set")

    weight_threshold = env_float("WEIGHT_CHANGE_THRESHOLD_PCT", DEFAULT_WEIGHT_THRESHOLD)
    top_movers = env_int("TOP_WEIGHT_MOVERS", DEFAULT_TOP_MOVERS)

    today = dt.date.today().isoformat()

    csv_text = fetch_csv(SPUS_URL)
    current_hash = sha256(csv_text)
    tickers_today, weights_today = parse_holdings(csv_text)

    conn, hash_col = db_connect_and_migrate()
    latest = get_latest_snapshot(conn, hash_col)

    save_snapshot(conn, hash_col, today, tickers_today, current_hash, weights_today)

    if latest is None:
        message = f"SPUS bot initialized ({today}). Saved {len(tickers_today)} tickers."
        log(message)
        send_telegram(token, chat_id, message)
        return

    prev_date, prev_tickers, prev_hash, prev_weights = latest

    if prev_hash and prev_hash == current_hash:
        log("No CSV content change detected. Skipping alert.")
        return

    added = tickers_today - prev_tickers
    removed = prev_tickers - tickers_today

    movers: List[Tuple[str, float]] = []
    for ticker in tickers_today & prev_tickers:
        if ticker in weights_today and ticker in prev_weights:
            delta = weights_today[ticker] - prev_weights[ticker]
            if abs(delta) >= weight_threshold:
                movers.append((ticker, delta))

    movers.sort(key=lambda item: abs(item[1]), reverse=True)
    movers = movers[:top_movers]

    if not added and not removed and not movers:
        log("No added/removed holdings and no significant weight changes.")
        return

    message = build_message(today, weight_threshold, added, removed, movers)
    log(
        f"Alerting vs {prev_date}: +{len(added)} added, "
        f"-{len(removed)} removed, {len(movers)} weight movers"
    )
    send_telegram(token, chat_id, message)
    log("Telegram alert sent")


if __name__ == "__main__":
    main()
