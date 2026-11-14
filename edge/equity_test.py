"""Two cheap event studies on current S&P 500 members (2018-2026):
 A. Overreaction: after a 1-day market-relative drop, does the stock bounce?
 B. Earnings drift: does the 2-day earnings reaction continue for 20/60 days?
Returns are market-relative (stock minus SPY), entered at the close of the
event day, so they are tradable in principle. t-stats are computed on monthly
averages of events (events bunch in time, so per-event t-stats overstate).
"""
import numpy as np, pandas as pd
px = pd.read_csv("data/px.csv.gz", parse_dates=["date"])
adj = px.pivot(index="date", columns="sym", values="adj").sort_index()
ret = adj.pct_change(fill_method=None)
ab = ret.sub(ret["SPY"], axis=0).drop(columns="SPY")
logab = np.log1p(ret).sub(np.log1p(ret["SPY"]), axis=0).drop(columns="SPY")
cum = logab.fillna(0).cumsum()
fwd = {h: np.expm1(cum.shift(-h) - cum).where(adj[ab.columns].shift(-h).notna()) for h in (1, 5, 20, 60)}

earn = pd.read_csv("data/earn.csv", parse_dates=["date"])
earn["sym"] = earn["sym"].str.replace(".", "-", regex=False)
dates = ab.index
pos = {d: i for i, d in enumerate(dates)}
is_earn = pd.DataFrame(False, index=dates, columns=ab.columns)
for d, s in zip(earn["date"], earn["sym"]):
    if d in pos and s in is_earn.columns:
        i = pos[d]
        is_earn.iloc[i:i + 2, is_earn.columns.get_loc(s)] = True   # day d (pre-open) or d+1 (after close)

def summarize(ev, label):
    """ev: DataFrame with columns date, f1,f5,f20,f60"""
    out = {"set": label, "n": len(ev)}
    ev = ev.assign(month=ev["date"].dt.to_period("M"))
    for h in (1, 5, 20, 60):
        x = ev[f"f{h}"].dropna()
        m = ev.groupby("month")[f"f{h}"].mean().dropna()
        out[f"mean{h}"] = round(100 * x.mean(), 2)
        out[f"med{h}"] = round(100 * x.median(), 2)
        out[f"t{h}"] = round(m.mean() / (m.std(ddof=1) / np.sqrt(len(m))), 1) if len(m) > 3 else np.nan
    return out

def events(mask):
    s = mask.stack(); s = s[s]
    ev = pd.DataFrame({"date": s.index.get_level_values(0), "sym": s.index.get_level_values(1)})
    for h, f in fwd.items():
        ev[f"f{h}"] = f.stack(future_stack=True).reindex(s.index).values
    return ev

rows = []
print("A. OVERREACTION (market-relative forward return %, entered at close of drop day)")
for lo, hi in ((-0.08, -0.05), (-0.12, -0.08), (-1, -0.12)):
    band = (ab > lo) & (ab <= hi)
    rows.append(summarize(events(band & ~is_earn), f"drop {int(hi*100)}..{int(lo*100)}% non-earnings"))
    rows.append(summarize(events(band & is_earn), f"drop {int(hi*100)}..{int(lo*100)}% earnings"))
big = (ab <= -0.08) & ~is_earn
for a, b in (("2018", "2020"), ("2021", "2023"), ("2024", "2026")):
    e = events(big); e = e[(e["date"] >= a) & (e["date"] <= b + "-12-31")]
    rows.append(summarize(e, f"drop <=-8% non-earn {a}-{b}"))
# mirror image: spikes (control: is it just volatility / survivorship?)
rows.append(summarize(events((ab >= 0.08) & ~is_earn), "RISE >=+8% non-earnings (control)"))
A = pd.DataFrame(rows); print(A.to_string(index=False)); A.to_csv("data/result_overreaction.csv", index=False)

print("\nB. EARNINGS DRIFT (2-day reaction quintile -> forward market-relative return %, entered at close of day d+1)")
r2 = np.expm1(cum - cum.shift(2))          # reaction over days d and d+1, measured at close of d+1
rows, ev_all = [], []
for d, s in zip(earn["date"], earn["sym"]):
    if d in pos and s in ab.columns and pos[d] + 1 < len(dates):
        t = dates[pos[d] + 1]
        ev_all.append((t, s, r2.at[t, s], fwd[20].at[t, s], fwd[60].at[t, s], fwd[1].at[t, s], fwd[5].at[t, s]))
E2 = pd.DataFrame(ev_all, columns=["date", "sym", "react", "f20", "f60", "f1", "f5"]).dropna(subset=["react"])
E2["q"] = E2.groupby(E2["date"].dt.to_period("Q"))["react"].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))
for q in range(5):
    e = E2[E2["q"] == q]
    o = summarize(e, f"reaction quintile {q+1} (avg react {100*e['react'].mean():+.1f}%)"); rows.append(o)
m = E2.assign(month=E2["date"].dt.to_period("Q")).pivot_table(index="month", columns="q", values=["f20", "f60"], aggfunc="mean")
for h in ("f20", "f60"):
    ls = (m[h][4] - m[h][0]).dropna()
    print(f"long top - short bottom quintile, {h}: mean {100*ls.mean():+.2f}% per quarter-cohort, t={ls.mean()/(ls.std(ddof=1)/np.sqrt(len(ls))):.1f}, positive in {int((ls>0).sum())}/{len(ls)} quarters")
    for a, b in (("2018", "2021"), ("2022", "2026")):
        x = ls[(ls.index >= a + "Q1") & (ls.index <= b + "Q4")]
        print(f"    {a}-{b}: {100*x.mean():+.2f}%  (n={len(x)} quarters)")
B = pd.DataFrame(rows); print(B.to_string(index=False)); B.to_csv("data/result_pead.csv", index=False)
