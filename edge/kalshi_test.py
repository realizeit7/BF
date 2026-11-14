"""Are Kalshi prices calibrated, and does anything survive the spread + fee?

Uses data/kalshi_snap.csv (mid-life bid/ask of settled markets).
For each price bucket: how often YES actually happened vs the quoted price, and
the average profit per $1 staked for a *taker* who buys YES at the ask or NO at
(1 - bid), after Kalshi's fee of 0.07 * fee_mult * p * (1-p) per contract.
Standard errors are clustered by event (strikes within an event are not independent).
"""
import numpy as np, pandas as pd
d = pd.read_csv("data/kalshi_snap.csv")
d = d[(d.ask - d.bid) <= 0.10].copy()
d["mid"] = (d.bid + d.ask) / 2
fee = lambda p: 0.07 * d.fee_mult * p * (1 - p)
d["yes_ret"] = (d.yes - d.ask - fee(d.ask)) / d.ask               # per $ staked
no_px = 1 - d.bid
d["no_ret"] = ((1 - d.yes) - no_px - fee(no_px)) / no_px
d["err"] = d.yes - d.mid

def cl(g, col):
    """mean and event-clustered standard error"""
    e = g.groupby("event")[col].agg(["sum", "count"])
    n = len(g); mu = g[col].mean()
    se = np.sqrt(((e["sum"] - mu * e["count"]) ** 2).sum()) / n * np.sqrt(len(e) / max(len(e) - 1, 1))
    return mu, se

def table(g, by):
    rows = []
    for k, x in g.groupby(by, observed=True):
        if len(x) < 30: continue
        e, ese = cl(x, "err"); y, yse = cl(x, "yes_ret"); n_, nse = cl(x, "no_ret")
        rows.append({by: k, "n": len(x), "events": x.event.nunique(), "avg_price": round(x.mid.mean(), 3),
                     "yes_rate": round(x.yes.mean(), 3), "miscal": round(e, 3), "miscal_t": round(e / ese, 1),
                     "buyYES_ret%": round(100 * y, 1), "t": round(y / yse, 1),
                     "buyNO_ret%": round(100 * n_, 1), "t ": round(n_ / nse, 1)})
    return pd.DataFrame(rows)

bins = [0, .05, .10, .20, .35, .50, .65, .80, .90, .95, 1]
d["bucket"] = pd.cut(d.mid, bins)
print(f"{len(d)} markets, {d.event.nunique()} events, median spread {100*(d.ask-d.bid).median():.0f}c, median lead {d.lead_h.median():.0f}h\n")
print("ALL SERIES by price bucket"); t = table(d, "bucket"); print(t.to_string(index=False)); t.to_csv("data/result_kalshi_bucket.csv", index=False)
d["zone"] = pd.cut(d.mid, [0, .15, .85, 1], labels=["longshot <15c", "middle", "favorite >85c"])
for cat, g in d.groupby("category"):
    print(f"\n{cat}"); print(table(g, "zone").to_string(index=False))
print("\nBY SERIES (all prices): taker returns")
print(table(d, "series").to_string(index=False))
# out-of-sample check of the one obvious rule: fade longshots (buy NO when YES mid < 15c)
ls = d[d.mid < .15].sort_values("close"); h = len(ls) // 2
for name, g in (("first half (by date)", ls.iloc[:h]), ("second half", ls.iloc[h:])):
    m, se = cl(g, "no_ret"); print(f"\nfade longshots, {name}: n={len(g)} ret {100*m:+.2f}% per $ staked, t={m/se:.1f}")
