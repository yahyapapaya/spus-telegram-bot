# Repository Review: `spus-telegram-bot`

## What this project does

This repository contains a single Python script (`spus_update_bot.py`) that monitors holdings changes in the **SPUS ETF** by downloading SP Funds' published CSV, comparing it with the most recent saved snapshot, and sending a Telegram alert when meaningful differences are detected.

## How it works (high-level)

1. Downloads the SPUS holdings CSV from a fixed URL.
2. Parses ticker symbols and portfolio weights from CSV columns.
3. Stores a daily snapshot in a local SQLite database (`spus_snapshots.db`).
4. Compares today vs latest prior snapshot for:
   - added tickers,
   - removed tickers,
   - significant weight deltas above a threshold.
5. Sends a Telegram message if anything meaningful changed.

## Runtime inputs

Required env vars:
- `TG_BOT_TOKEN`
- `TG_CHAT_ID`

Optional env vars:
- `WEIGHT_CHANGE_THRESHOLD_PCT` (default `0.10`)
- `TOP_WEIGHT_MOVERS` (default `15`)

Dependency list is minimal (`requests`).

## Storage model

SQLite table: `snapshots`
- `snap_date` (PK)
- `tickers_json`
- `csv_sha256`
- `weights_json`
- `created_at`

The script also contains migration logic for older schemas (e.g., old `csv_hash` naming, missing `weights_json`).

## Triggering behavior

The script avoids noisy alerts by:
- skipping when CSV hash is unchanged,
- skipping when no add/remove tickers and no movers above threshold,
- limiting mover rows to top `N` by absolute weight delta.

## Operational notes

- This appears intended to run on a schedule (e.g., cron/GitHub Actions).
- It is stateful because it depends on local SQLite history.
- It currently has no tests and no `README` usage guide.

## Suggested next improvements

- Add a `README.md` with setup + scheduling examples.
- Add unit tests for `parse_weight_to_pct`, `parse_holdings`, and message formatting.
- Add structured logging and graceful retry/backoff strategy for Telegram send failures.
- Consider alert deduping by date+hash if run multiple times daily.

## Automation added

A GitHub Actions workflow has been added at `.github/workflows/spus-update-bot.yml` to run:
- daily at **9:00 AM SGT**
- daily at **9:00 PM SGT**

(These are `01:00` and `13:00` UTC in cron.)

To use it, configure repository secrets:
- `TG_BOT_TOKEN`
- `TG_CHAT_ID`

Optional repository variables:
- `WEIGHT_CHANGE_THRESHOLD_PCT`
- `TOP_WEIGHT_MOVERS`
