"""
Alpaca Trading Bot - entry point.

Usage:
  1. Copy .env.example to .env and set ALPACA_API_KEY, ALPACA_SECRET_KEY.
  2. Use paper trading URL in .env (default) for testing.
  3. Run: python main.py

Stop with Ctrl+C.

When PORT is set (e.g. on Heroku/Render), starts a tiny HTTP server so the
platform sees the app as "up"; the bot runs in a background thread.
"""
import os
import threading

from bot import run_bot


def _run_bot_in_thread():
    run_bot()


def _serve_health(port: int):
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
    except ImportError:
        return
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        def log_message(self, format, *args):
            pass
    server = HTTPServer(("", port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    port = os.environ.get("PORT")
    if port:
        port = int(port)
        t = threading.Thread(target=_run_bot_in_thread, daemon=True)
        t.start()
        _serve_health(port)
    else:
        run_bot()
