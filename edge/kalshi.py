"""Minimal Kalshi public-API client (no auth needed for market data)."""
import time, requests
K = "https://api.elections.kalshi.com/trade-api/v2"
S = requests.Session()

def get(path, **params):
    for i in range(6):
        r = S.get(K + path, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(1.5 * (i + 1)); continue
        raise RuntimeError(f"{r.status_code} {path} {r.text[:200]}")
    raise RuntimeError(f"gave up {path}")

def markets(series, historical, max_pages=50, **kw):
    path = "/historical/markets" if historical else "/markets"
    if not historical: kw["status"] = "settled"
    out, cur = [], None
    for _ in range(max_pages):
        p = dict(series_ticker=series, limit=1000, **kw)
        if cur: p["cursor"] = cur
        d = get(path, **p)
        out += d.get("markets", [])
        cur = d.get("cursor")
        if not cur or not d.get("markets"): break
    return out

def candles(series, ticker, start, end, interval, historical):
    path = (f"/historical/markets/{ticker}/candlesticks" if historical
            else f"/series/{series}/markets/{ticker}/candlesticks")
    return get(path, start_ts=int(start), end_ts=int(end), period_interval=interval).get("candlesticks", [])
