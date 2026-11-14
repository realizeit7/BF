"""Pull daily adjusted prices (Yahoo) for current S&P 500 members + SPY, and
earnings dates (Nasdaq calendar). Outputs data/px.parquet|csv, data/earn.csv

Caveat baked into this dataset: membership is *today's* index, so companies
that fell out of the index (often after big drops) are missing -> survivorship
bias that flatters any "buy the dip" result.
"""
import csv, io, sys, time, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests
H = {"User-Agent": "Mozilla/5.0"}
START = "2018-01-01"

def yahoo(sym):
    p1 = int(dt.datetime.fromisoformat(START).timestamp()); p2 = int(time.time())
    for i in range(4):
        try:
            r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                             params=dict(period1=p1, period2=p2, interval="1d", events="div,split"),
                             headers=H, timeout=30)
            if r.status_code != 200: time.sleep(2 * (i + 1)); continue
            res = r.json()["chart"]["result"][0]
            idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
            q = res["indicators"]["quote"][0]
            return sym, pd.DataFrame({"adj": res["indicators"]["adjclose"][0]["adjclose"],
                                      "close": q["close"], "vol": q["volume"]}, index=idx)
        except Exception as e:
            time.sleep(2 * (i + 1))
    print("FAIL", sym, file=sys.stderr); return sym, None

def earn(day):
    for i in range(4):
        try:
            r = requests.get("https://api.nasdaq.com/api/calendar/earnings", params={"date": day}, headers=H, timeout=30)
            rows = (r.json().get("data") or {}).get("rows") or []
            return [(day, x["symbol"], x.get("time", ""), x.get("surprise", "")) for x in rows]
        except Exception:
            time.sleep(2 * (i + 1))
    print("FAIL", day, file=sys.stderr); return []

if __name__ == "__main__":
    src = requests.get("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv", timeout=30).text
    syms = [s.replace(".", "-") for s in pd.read_csv(io.StringIO(src))["Symbol"]] + ["SPY"]
    with ThreadPoolExecutor(6) as ex:
        got = {s: d for s, d in ex.map(yahoo, syms) if d is not None}
    px = pd.concat(got, names=["sym", "date"]).reset_index()
    px.to_csv("data/px.csv.gz", index=False)
    print(len(got), "symbols", len(px), "rows", flush=True)
    days = [d.strftime("%Y-%m-%d") for d in got["SPY"].index if d >= pd.Timestamp("2018-06-01")]
    with ThreadPoolExecutor(5) as ex:
        rows = [r for rs in ex.map(earn, days) for r in rs]
    keep = set(syms)
    with open("data/earn.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["date", "sym", "time", "surprise"])
        w.writerows([r for r in rows if r[1].replace(".", "-") in keep])
    print(len(rows), "earnings rows total")
