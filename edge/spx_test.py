"""Does the options market (VIX1D) price Kalshi S&P contracts better than Kalshi does,
by enough to pay the spread + fee?

Model: P(S&P close beats strike) from a normal distribution centred on the current
level with sd = k * level * VIX1D * sqrt(hours left / 6.5 / 252). k is fitted on the
first half of days only (VIX overstates realised vol on average, so k < 1 is expected);
everything is then judged on the second half. Standard errors cluster by day.
"""
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar
d = pd.read_csv("data/spx_snap.csv").sort_values("date")
d = d[(d.ask - d.bid) <= 0.10].copy()
d["mid"] = (d.bid + d.ask) / 2

def p_model(x, k):
    sd = k * x.spx * x.vix1d / 100 * np.sqrt(x.hours / 6.5 / 252)
    up = lambda K: norm.sf((K - x.spx) / sd)               # P(close > K)
    p = np.where(x.type.isin(["greater", "greater_or_equal"]), up(x.lo),
        np.where(x.type == "less", 1 - up(x.hi), up(x.lo) - up(x.hi)))
    return np.clip(p, 1e-4, 1 - 1e-4)

ll = lambda p, y: -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
days = np.sort(d.date.unique()); split = days[len(days) // 2]
tr, te = d[d.date < split], d[d.date >= split].copy()
k = minimize_scalar(lambda k: ll(p_model(tr, k), tr.yes), bounds=(0.3, 2), method="bounded").x
te["pm"] = p_model(te, k)
print(f"{len(d)} contracts on {len(days)} days; train < {split} <= test; fitted k = {k:.2f}")
print(f"median spread {100*(d.ask-d.bid).median():.1f}c, fee multiplier {d.fee_mult.iloc[0]}")
mid = te["mid"].clip(.01, .99)
print(f"\nTEST log-loss (lower = better forecast): Kalshi mid {ll(mid, te.yes):.4f}   options model {ll(te.pm, te.yes):.4f}   50/50 blend {ll((mid+te.pm)/2, te.yes):.4f}")
print(f"TEST mean |Kalshi mid - model| = {100*np.abs(mid-te.pm).mean():.1f}c")

fee = lambda p: 0.07 * te.fee_mult * p * (1 - p)
e_yes = te.pm - te.ask - fee(te.ask)                       # expected profit per contract, model's view
e_no = (1 - te.pm) - (1 - te.bid) - fee(1 - te.bid)
def result(mask, side):
    px = te.ask if side == "YES" else 1 - te.bid
    win = te.yes if side == "YES" else 1 - te.yes
    pnl = (win - px - fee(px))[mask]                       # realised profit per contract
    g = pnl.groupby(te.date[mask]).sum(); n = mask.sum()
    if n < 10: return f"{side}: n={n}"
    se = np.sqrt((g ** 2).sum() - g.sum() ** 2 / len(g)) / n * np.sqrt(len(g) / (len(g) - 1)) if len(g) > 1 else np.nan
    m = pnl.mean()
    return f"{side}: n={n:4d} on {len(g):3d} days, avg cost {px[mask].mean():.2f}, profit {100*m:+.2f}c/contract, t={m/se:.1f}, model expected {100*(e_yes if side=='YES' else e_no)[mask].mean():+.2f}c"
print("\nTEST trading rule: take the side the options model says is cheap, if edge > margin after fee")
for margin in (0.0, 0.02, 0.05, 0.10):
    print(f"  margin {margin:.2f}  " + result(e_yes > margin, "YES") + "\n               " + result(e_no > margin, "NO"))
print("\nBy series (margin 0.02, both sides combined):")
for s in te.series.unique():
    m = te.series == s
    for side, e in (("YES", e_yes), ("NO", e_no)):
        print(f"  {s:7s} " + result(m & (e > 0.02), side))
te.to_csv("data/result_spx_test.csv", index=False)
