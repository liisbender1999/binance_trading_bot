"""SQLite persistence for bot trades: trading_bot.db with trades table and total profit query."""
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent / "trading_bot.db"


def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Create trading_bot.db and the trades table if they do not exist."""
    conn = _get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                realized_pnl REAL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def insert_trade(
    timestamp: str,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    realized_pnl: Optional[float] = None,
) -> None:
    """Insert one trade (buy or sell). For sells, pass realized_pnl to support total profit query."""
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO trades (timestamp, symbol, side, quantity, price, realized_pnl)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, symbol, side, quantity, price, realized_pnl),
        )
        conn.commit()
    finally:
        conn.close()


def get_total_profit() -> float:
    """Return the sum of realized_pnl from all sell trades (total profit)."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) FROM trades WHERE realized_pnl IS NOT NULL"
        ).fetchone()
        return float(row[0]) if row else 0.0
    finally:
        conn.close()
