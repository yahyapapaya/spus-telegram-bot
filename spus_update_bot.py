import os
import csv
import json
import sqlite3
import datetime as dt
import hashlib
import re
import time
import requests
from typing import Dict, Set, Optional, Tuple, List

# =====================
# CONFIG
# =====================
SPUS_URL = "https://www.sp-funds.com/wp-content/uploads/data/TidalFG_Holdings_SPUS.csv"

TICKER_COLUMN = "StockTicker"     # confirmed by your screenshot
WEIGHT_COLUMN = "Weightings"      # confirmed by your screenshot

# Tickers like AAPL, MSFT, NVDA. (SPUS appears to use plain tickers without dots)
TICKER_RE = re.compile(r"^[A-Z]{1,6}$")

# Alert tuning (can be overridden by env vars)
WEIGHT_CHANGE_THRESHOLD_PCT = float(os.getenv("WEIGHT_CHANGE_THRESHOLD_PCT", "0.10"))  # 0.10% default
TOP_WEIGHT_MOVERS = int(os.getenv("TOP_WEIGHT_MOVERS", "15"))  # show top 15 movers

# =====================
# UTILITIES
# =====================
def log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def db_path() -> str:
    return os.path.join(script_dir(), "spus_snapshots.db")

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def utc_now_iso() -> str:
    # timezone-aware UTC timestamp (no utcnow deprecation warning)
    return dt.datetime.now(dt.UTC).isoformat()

def parse_weight_to_pct(s: str) -> Optional[float]:
    """
    Converts any of these into PERCENT points float:
      "14.18%" -> 14.18
      "14.18"  -> 14.18
      "0.1418" -> 14.18  (fraction form)
      ""/None  -> None
    Also handles commas.
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None

    s = s.replace(",", "")
    has_pct = s.endswith("%")
    if has_pct:
        s = s[:-1].strip()

    try:
        v = float(s)
    except Exception:
        return None

    if has_pct:
        return v

    # If no % sign: values > 1 likely already percent; <=1 likely fraction
    return v if v > 1.0 else v * 100.0

def fmt_pct(delta: float) -> str:
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.2f}%"

# =====================
# FETCH CSV
# =====================
def fetch_csv() -> str:
    """
    SP Funds sometimes blocks plain requests (403).
    We mimic a browser with headers and retry.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.sp-funds.com/",
        "Connection": "keep-alive",
    }

    last_err = None
    for attempt in range(1, 4):  # 3 attempts
        try:
            log(f"Downloading SPUS holdings CSV... (attempt {attempt})")
            r = requests.get(SPUS_URL, headers=headers, timeout=30)

            if r.status_code == 403:
                raise requests.exceptions.HTTPError("403 Forbidden", response=r)

            r.raise_for_status()
            return r.text

        except Exception as e:
            last_err = e
            log(f"Download failed: {e}")
            if attempt < 3:
                time.sleep(2)

    raise last_err

# =====================
# PARSE HOLDINGS
# =====================
def parse_holdings(csv_text: str) -> Tuple[Set[str], Dict[str, float]]:
    """
    Returns:
      tickers_set
      weights_pct_by_ticker (percentage points float, e.g. 14.18 means 14.18%)
    """
    reader = csv.DictReader(csv_text.splitlines())

    if reader.fieldnames is None:
        raise RuntimeError("CSV has no headers. Possibly blocked or not CSV content.")

    if TICKER_COLUMN not in reader.fieldnames:
        raise RuntimeError(f"Ticker column '{TICKER_COLUMN}' not found. Headers: {reader.fieldnames}")

    if WEIGHT_COLUMN not in reader.fieldnames:
        raise RuntimeError(f"Weight column '{WEIGHT_COLUMN}' not found. Headers: {reader.fieldnames}")

    tickers: Set[str] = set()
    weights: Dict[str, float] = {}

    for row in reader:
        ticker = (row.get(TICKER_COLUMN) or "").strip().upper()
        if not ticker or not TICKER_RE.match(ticker):
            continue

        tickers.add(ticker)

        w = parse_weight_to_pct(row.get(WEIGHT_COLUMN))
        if w is not None:
            weights[ticker] = w

    log(f"Parsed {len(tickers)} unique tickers")
    log(f"Parsed {len(weights)} tickers with weights")
    return tickers, weights

# =====================
# DATABASE (schema-proof + migration)
# =====================
def db_connect_and_migrate() -> Tuple[sqlite3.Connection, str]:
    """
    Returns (conn, hash_column_name) with automatic migration.
    Handles older DB schemas (csv_sha256 vs csv_hash, and missing weights_json).
    """
    conn = sqlite3.connect(db_path())

    # Ensure base table exists with an explicit hash column (csv_sha256)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            snap_date TEXT PRIMARY KEY,
            tickers_json TEXT NOT NULL,
            csv_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()

    cols = [r[1] for r in conn.execute("PRAGMA table_info(snapshots)").fetchall()]

    # Determine hash column to use
    if "csv_sha256" in cols:
        hash_col = "csv_sha256"
    elif "csv_hash" in cols:
        hash_col = "csv_hash"
    else:
        log("DB migration: adding missing column 'csv_sha256'...")
        conn.execute("ALTER TABLE snapshots ADD COLUMN csv_sha256 TEXT NOT NULL DEFAULT ''")
        conn.commit()
        hash_col = "csv_sha256"

    # Ensure weights_json exists
    cols = [r[1] for r in conn.execute("PRAGMA table_info(snapshots)").fetchall()]
    if "weights_json" not in cols:
        log("DB migration: adding missing column 'weights_json'...")
        conn.execute("ALTER TABLE snapshots ADD COLUMN weights_json TEXT")
        conn.commit()

    return conn, hash_col

def get_latest_snapshot(conn: sqlite3.Connection, hash_col: str) -> Optional[Tuple[str, Set[str], str, Dict[str, float]]]:
    row = conn.execute(
        f"SELECT snap_date, tickers_json, {hash_col}, weights_json FROM snapshots ORDER BY snap_date DESC LIMIT 1"
    ).fetchone()

    if not row:
        return None

    snap_date = row[0]
    tickers = set(json.loads(row[1]))
    h = row[2] or ""
    weights_json = row[3]
    weights = json.loads(weights_json) if weights_json else {}

    # weights values are already percent floats
    return snap_date, tickers, h, {k: float(v) for k, v in weights.items()}

def save_snapshot(conn: sqlite3.Connection, hash_col: str, date: str, tickers: Set[str], h: str, weights: Dict[str, float]) -> None:
    conn.execute(
        f"""
        INSERT OR REPLACE INTO snapshots (snap_date, tickers_json, {hash_col}, weights_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (date, json.dumps(sorted(tickers)), h, json.dumps(weights), utc_now_iso())
    )
    conn.commit()

# =====================
# TELEGRAM
# =====================
def send_telegram(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()

def build_message(date: str,
                  added: Set[str],
                  removed: Set[str],
                  movers: List[Tuple[str, float]]) -> str:
    lines = [f"SPUS Holdings Update ({date})"]

    lines.append("")
    lines.append("Stocks Removed from Holdings:")
    lines.extend(sorted(removed) if removed else ["(none)"])

    lines.append("")
    lines.append("Stocks Loaded into Holdings:")
    lines.extend(sorted(added) if added else ["(none)"])

    lines.append("")
    lines.append(f"Weight Changes (>|{WEIGHT_CHANGE_THRESHOLD_PCT:.2f}%|):")
    if movers:
        for t, d in movers:
            direction = "increased" if d > 0 else "decreased"
            lines.append(f"{t} weight {direction} {fmt_pct(d)}")
    else:
        lines.append("(none)")

    return "\n".join(lines)

# =====================
# MAIN
# =====================
def main() -> None:
    token = os.getenv("TG_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TG_CHAT_ID", "").strip()

    if not token or not chat_id:
        raise RuntimeError("TG_BOT_TOKEN or TG_CHAT_ID not set")

    today = dt.date.today().isoformat()

    csv_text = fetch_csv()
    current_hash = sha256(csv_text)
    tickers_today, weights_today = parse_holdings(csv_text)

    conn, hash_col = db_connect_and_migrate()
    latest = get_latest_snapshot(conn, hash_col)

    # Save today's snapshot first
    save_snapshot(conn, hash_col, today, tickers_today, current_hash, weights_today)

    if latest is None:
        msg = f"SPUS bot initialized ({today}). Saved {len(tickers_today)} tickers."
        log(msg)
        send_telegram(token, chat_id, msg)
        return

    prev_date, prev_tickers, prev_hash, prev_weights = latest

    # If exact same CSV content, skip entirely (prevents spam)
    if prev_hash and prev_hash == current_hash:
        log("No CSV content change detected. Skipping alert.")
        return

    # Added/Removed
    added = tickers_today - prev_tickers
    removed = prev_tickers - tickers_today

    # Weight movers (only tickers existing on both days and having weights on both days)
    common = tickers_today & prev_tickers
    movers_all: List[Tuple[str, float]] = []

    for t in common:
        if t in weights_today and t in prev_weights:
            delta = float(weights_today[t]) - float(prev_weights[t])
            if abs(delta) >= WEIGHT_CHANGE_THRESHOLD_PCT:
                movers_all.append((t, delta))

    # Sort by absolute delta descending and keep top N
    movers_all.sort(key=lambda x: abs(x[1]), reverse=True)
    movers = movers_all[:TOP_WEIGHT_MOVERS]

    # Only alert if there’s something meaningful
    if not added and not removed and not movers:
        log("No added/removed tickers and no significant weight changes. Skipping alert.")
        return

    msg = build_message(today, added, removed, movers)
    log(f"Alerting vs {prev_date}: +{len(added)} added, -{len(removed)} removed, {len(movers)} weight movers")
    send_telegram(token, chat_id, msg)
    log("Telegram alert sent.")

if __name__ == "__main__":
    main()
