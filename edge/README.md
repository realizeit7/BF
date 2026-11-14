# Edge pipeline

A repeatable way to look for investments where research gives a measurable
advantage, kept separate from the core long-term holdings (401(k), SCHG).

**Rule of the pipeline:** an idea is judged by *expected profit at the price you
can actually trade*, after spread and fees — not by whether the story is true.

## The four steps

1. **Discover** – add any idea to the register below with one line on who is on
   the other side and why they would misprice it.
2. **Explain the edge** – it must have a reason to persist: too small for
   professionals, awkward to access, a structural/forced seller, or a bias of
   the crowd on that venue. "I'm smarter" is not a reason.
3. **Test cheaply** – one script per idea, real quotes (bid/ask, not last
   trade), real fees, a control group, and standard errors that respect
   clustering. Then log paper trades in `journal.csv` *before* outcomes are known.
4. **Decide** – Reject / Track (paper trades only) / Investigate (more data) /
   Fund (small real money). Promotion to Fund needs: positive after costs in
   history, positive in a held-out period, and >= 30 journal entries that agree.

## Using derivatives (options, futures)

Options and futures are **signals by default**: their prices are the professional
market's estimate of probabilities and expected moves, and we use them to judge
whether something else (a Kalshi contract, a stock after an event) is cheap or
expensive. Trading them ourselves is the exception: only for an event-driven
situation most option traders don't model (tender offer, deal deadline,
scheduled release), with a defined maximum loss, and only after it clears the
same evidence bar as everything else. Rare or never is acceptable.

## Register (first pass, 2026-11-14)

| Idea | Possible edge | Cheap test result | Verdict |
|---|---|---|---|
| Kalshi: fade longshots generally | Retail overpays for lottery tickets | Bias is real (2.9c contracts win 1.6% of the time, t=-3.0) but buying NO nets +0.2% after spread and fee; +0.7% / -0.2% in the two halves | **Reject** as a taker strategy |
| Kalshi: fade longshots in economic-data markets (gas, CPI, Fed) | Thin, retail-heavy, outcome range is forecastable | 127 contracts priced ~6c, 1 paid out; buying NO +3.6% per $ staked, t=4.1 (78 events). One cell out of ~25 examined, so partly luck | **Investigate** – widen to all economics series, hold out the last 6 months |
| Kalshi: weather / entertainment longshots | same | +1.7% (t=2.6) and +3.6% (t=2.4) | **Track** |
| Kalshi: take either side in mid-priced sports/mentions | none | Loses 4-10% per $ both ways; prices are calibrated, cost is the spread + fee | **Reject** |
| Kalshi: provide liquidity (rest limit orders) | Takers lose on both sides, so makers collect it | Not testable from quotes alone (adverse selection) | **Investigate** – needs trade-level data |
| Kalshi vs Polymarket same event | Venues are segmented | Dec-2026 Fed ">25bp hike": 10c bid on Kalshi vs 8.3c ask on Polymarket, ~1c net on 98c for 4 weeks | **Track** (journal) – small, needs both accounts |
| Stocks: buy large-cap 1-day drops >=8% with no earnings | Forced/emotional selling overshoots | +1.0% vs market over 5 days relative to equal-sized *rises* (t=2.6); 55% win rate; absent in 2021-23; sample is today's index members, which flatters it | **Track** – paper trade every signal |
| Stocks: buy earnings-day drops | same | No bounce; -8..-12% earnings drops lose another 1.0% over 20 days (t=-2.5) | **Reject** |
| Stocks: post-earnings drift in large caps | Slow reaction to news | Top-minus-bottom reaction quintile +0.6% over 20 days, t=0.9 | **Reject** in large caps (small caps untested) |
| Odd-lot tender offers | Holders of <100 shares skip proration; too small for funds | 13 listed offers in 12 months; only 1 (Yext) traded below the price floor after announcement (+2.5% minimum in ~4 weeks). Capped at 99 shares | **Track** – real but worth tens of dollars each |
| Untested backlog | | | merger spreads, closed-end fund discounts, index additions, spin-offs, Kalshi S&P ranges vs option prices |

## Files

- `kalshi_pull.py`, `kalshi_test.py` – calibration of settled Kalshi markets using the bid/ask at the midpoint of each market's life.
- `equity_pull.py`, `equity_test.py` – overreaction and earnings-drift event studies.
- `tender_scan.py` – finds issuer tender offers with odd-lot priority in SEC filings. Price fields are regex guesses; read the filing before acting.
- `journal.csv` – prospective paper trades. This is the real test.
- `data/` – cached inputs and result tables (price file is not committed; rerun `equity_pull.py`).

## Known weaknesses

- Stock sample = current S&P 500 members (survivorship). The "rises" control removes much of it, not all.
- Kalshi sample: 22 series, markets with volume >= 500 and spread <= 10c, one snapshot per market. ~25 category/zone cells were examined, so a lone t of 2-3 is weak evidence.
- Quotes are top-of-book closes of hourly bars; size available at that price is unknown.
