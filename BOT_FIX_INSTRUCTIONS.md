# Fix for "init_db is not defined" in bot.py

Your bot.py has the db import commented out but still calls `init_db()`, `insert_trade()`, and `get_total_profit()`. Do these **3 edits** in order.

---

## EDIT 1 — Add optional db import (near the top)

**Find this (around line 10–12):**
```python
#from db import init_db, insert_trade, get_total_profit
from config import (
```

**Replace with:**
```python
# Optional: use db.py for trade logging; if missing, bot still runs
def _db_noop(*args, **kwargs):
    pass

def _get_total_profit_default():
    return 0.0

try:
    from db import init_db, insert_trade, get_total_profit
    _db_available = True
except ImportError:
    _db_available = False
    init_db = insert_trade = _db_noop
    get_total_profit = _get_total_profit_default

from config import (
```

---

## EDIT 2 — Guard the sell block (around the "Exiting lot" log)

**Find this:**
```python
                    client.submit_market_order(side="sell", qty=qty_to_sell)
                    realized_pnl = (current_price - float(lot["entry_price"])) * sell_qty
                    insert_trade(
                        datetime.utcnow().isoformat(),
                        client.symbol,
                        "sell",
                        float(sell_qty),
                        current_price,
                        realized_pnl=realized_pnl,
                    )
                    logger.info(
                        "Exiting lot: %s (price=%.2f, qty=%s) | total profit (DB): %.2f",
                        reason, current_price, qty_to_sell, get_total_profit(),
                    )
```

**Replace with:**
```python
                    client.submit_market_order(side="sell", qty=qty_to_sell)
                    if _db_available:
                        realized_pnl = (current_price - float(lot["entry_price"])) * sell_qty
                        insert_trade(
                            datetime.utcnow().isoformat(),
                            client.symbol,
                            "sell",
                            float(sell_qty),
                            current_price,
                            realized_pnl=realized_pnl,
                        )
                        logger.info(
                            "Exiting lot: %s (price=%.2f, qty=%s) | total profit (DB): %.2f",
                            reason, current_price, qty_to_sell, get_total_profit(),
                        )
                    else:
                        logger.info("Exiting lot: %s (price=%.2f, qty=%s)", reason, current_price, qty_to_sell)
```

---

## EDIT 3 — Guard the buy block and run_bot

**Find this (buy block):**
```python
                client.submit_market_order(side="buy", qty=qty)
                insert_trade(
                    datetime.utcnow().isoformat(),
                    client.symbol,
                    "buy",
                    float(qty),
                    entry_price_est,
                )
                new_lot = {
```

**Replace with:**
```python
                client.submit_market_order(side="buy", qty=qty)
                if _db_available:
                    insert_trade(
                        datetime.utcnow().isoformat(),
                        client.symbol,
                        "buy",
                        float(qty),
                        entry_price_est,
                    )
                new_lot = {
```

**Then find this in run_bot():**
```python
def run_bot() -> None:
    """Run the bot loop forever (Ctrl+C to stop)."""
    init_db()
    client = AlpacaClient()
```

**Replace with:**
```python
def run_bot() -> None:
    """Run the bot loop forever (Ctrl+C to stop)."""
    if _db_available:
        init_db()
    client = AlpacaClient()
```

---

After these 3 edits, save bot.py and redeploy. The bot will run even when db.py is missing.
