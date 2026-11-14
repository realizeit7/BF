"""Kalshi S&P 500 end-of-day contracts vs the options market's implied volatility.

For every trading day: S&P level and VIX1D (options-implied 1-day vol) at 17:30 UTC
from Yahoo hourly bars; the Kalshi bid/ask at that same minute for strikes near the
money in two series: KXINXU ("above X", 5-pt steps) and KXINX (25-pt ranges).
Output: data/spx_snap.csv
"""
import csv, sys, math, datetime as dt, requests
from concurrent.futures import ThreadPoolExecutor
import kalshi
H = {"User-Agent": "Mozilla/5.0"}
CUTOFF = dt.datetime.fromisoformat(kalshi.get("/historical/cutoff")["market_settled_ts"].replace("Z", "+00:00"))

def bars(sym):
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                     params=dict(range="730d", interval="1h"), headers=H, timeout=30).json()["chart"]["result"][0]
    return {t + 3600: c for t, c in zip(r["timestamp"], r["indicators"]["quote"][0]["close"]) if c}  # key = bar END
spx, vix = bars("%5EGSPC"), bars("%5EVIX1D")
days = sorted({dt.datetime.fromtimestamp(t, dt.UTC).date() for t in spx})
fee = {s["ticker"]: s.get("fee_multiplier", 1) for s in kalshi.get("/series")["series"] if s["ticker"] in ("KXINX", "KXINXU")}
ts = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00"))

def day_jobs(d):
    t = int(dt.datetime(d.year, d.month, d.day, 17, 30, tzinfo=dt.UTC).timestamp())
    if t not in spx or t not in vix: return []
    S, V = spx[t], vix[t]
    out = []
    for series in ("KXINXU", "KXINX"):
        ev = f"{series}-{d.strftime('%y%b%d').upper()}H1600"
        try:
            ms = kalshi.get("/markets", event_ticker=ev, limit=200)["markets"] or \
                 kalshi.get("/historical/markets", event_ticker=ev, limit=200)["markets"]
        except Exception: ms = []
        if not ms: continue
        close = ts(ms[0]["close_time"]); h = (close.timestamp() - t) / 3600
        if not (1 < h < 5): continue
        sd = S * V / 100 * math.sqrt(h / 6.5 / 252)
        for m in ms:
            if m.get("result") not in ("yes", "no"): continue
            lo, hi = m.get("floor_strike"), m.get("cap_strike")
            ref = lo if lo is not None else hi
            if m["strike_type"] == "between": ref = (lo + hi) / 2
            if abs(ref - S) > 2.5 * sd: continue
            if series == "KXINXU" and round(lo) % 10: continue            # every other 5-pt strike
            out.append((series, m, t, S, V, h, close))
    return out

def snap(job):
    try: return _snap(job)
    except Exception as e: print("ERR", job[1]["ticker"], e, file=sys.stderr); return None

def _snap(job):
    series, m, t, S, V, h, close = job
    hist = close < CUTOFF
    path = (f"/historical/markets/{m['ticker']}/candlesticks" if hist else f"/series/{series}/markets/{m['ticker']}/candlesticks")
    try:
        cs = [c for c in kalshi.get(path, start_ts=t - 600, end_ts=t, period_interval=1).get("candlesticks", []) if c["end_period_ts"] <= t]
    except Exception as e:
        print("ERR", m["ticker"], e, file=sys.stderr); return None
    if not cs: return None
    g = lambda side: float(cs[-1][side].get("close_dollars", cs[-1][side].get("close")) or "nan")
    bid, ask = g("yes_bid"), g("yes_ask")
    if not (0 <= bid < ask <= 1): return None
    return dict(series=series, event=m["event_ticker"], ticker=m["ticker"], date=close.date().isoformat(),
                type=m["strike_type"], lo=m.get("floor_strike"), hi=m.get("cap_strike"), spx=S, vix1d=V, hours=round(h, 2),
                bid=bid, ask=ask, fee_mult=fee[series], settle=float(str(m.get("expiration_value") or "nan").replace(",", "")),
                yes=int(m["result"] == "yes"))

with ThreadPoolExecutor(4) as ex:
    jobs = [j for js in ex.map(day_jobs, days) for j in js]
    print(len(days), "days", len(jobs), "contracts", file=sys.stderr, flush=True)
    rows = [r for r in ex.map(snap, jobs) if r]
with open("data/spx_snap.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "rows", file=sys.stderr)
