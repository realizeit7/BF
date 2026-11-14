"""Pull a sample of settled Kalshi markets with a mid-life bid/ask snapshot.

Snapshot = last hourly candle ending at or before the midpoint of the market's
life (open -> close), so the quote is one a person could really have traded,
well before the outcome was known. Output: data/kalshi_snap.csv
"""
import csv, json, random, sys, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import kalshi

SERIES = ["KXHIGHNY","KXHIGHLAX","KXHIGHCHI","KXHIGHMIA","KXRAIN",
          "KXINX","KXNASDAQ100","KXBTCD","KXWTI",
          "KXNBAGAME","KXNFLGAME","KXMLBGAME","KXATPMATCH","KXNCAAFGAME",
          "KXFEDDECISION","KXCPIYOY","KXAAAGASW","KXTRUMPAPPROVE",
          "KXTRUMPMENTION","KXNBAMENTION","KXRT","KXTOPMODEL"]
PER_SERIES = 450
random.seed(7)
ts = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
f = lambda c, side, k: float(c[side].get(k + "_dollars", c[side].get(k)) or "nan")
meta = {s["ticker"]: s for s in kalshi.get("/series")["series"]}

def snap(job):
    series, m, hist = job
    try:
        o, c = ts(m["open_time"]), ts(m["close_time"])
        if c - o < 4 * 3600: return None
        mid = (o + c) / 2
        cs = kalshi.candles(series, m["ticker"], mid - 12 * 3600, mid, 60, hist)
        cs = [x for x in cs if x["end_period_ts"] <= mid]
        if not cs: return None
        x = cs[-1]
        bid, ask = f(x, "yes_bid", "close"), f(x, "yes_ask", "close")
        if not (0 < bid < ask < 1): return None
        return dict(series=series, category=meta[series]["category"],
                    fee_mult=meta[series].get("fee_multiplier", 1),
                    event=m["event_ticker"], ticker=m["ticker"], close=m["close_time"][:10],
                    life_h=round((c - o) / 3600, 1), lead_h=round((c - x["end_period_ts"]) / 3600, 1),
                    bid=bid, ask=ask, vol=float(m.get("volume_fp") or 0),
                    yes=int(m["result"] == "yes"))
    except Exception as e:
        print("ERR", m["ticker"], e, file=sys.stderr); return None

jobs = []
for s in SERIES:
    ms = [(m, True) for m in kalshi.markets(s, True, max_pages=6)] + \
         [(m, False) for m in kalshi.markets(s, False, max_pages=6)]
    ms = [(m, h) for m, h in ms if m.get("result") in ("yes", "no") and float(m.get("volume_fp") or 0) >= 500]
    random.shuffle(ms)
    jobs += [(s, m, h) for m, h in ms[:PER_SERIES]]
    print(s, len(ms), file=sys.stderr, flush=True)
with ThreadPoolExecutor(4) as ex:
    rows = [r for r in ex.map(snap, jobs) if r]
with open("data/kalshi_snap.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(jobs), "requested", len(rows), "usable")
