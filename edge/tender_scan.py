"""Scan EDGAR for issuer tender offers (SC TO-I) that give odd-lot holders
(<100 shares) priority, i.e. no proration. Usage: tender_scan.py 2026Q3 2026Q4
Output: data/tenders.csv  (one row per filing whose main documents mention odd lots)
"""
import csv, re, sys, time, html, requests
H = {"User-Agent": "research-script admin@example.com"}
def get(u):
    for i in range(4):
        r = requests.get(u, headers=H, timeout=40)
        if r.status_code == 200: return r.text
        time.sleep(1 + 2 * i)
    return ""

rows = []
for q in sys.argv[1:]:
    y, n = q.split("Q")
    idx = get(f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{n}/form.idx")
    for line in idx.splitlines():
        if not line.startswith("SC TO-I "): continue          # original filings only, not /A
        m = re.match(r"SC TO-I\s+(.*?)\s+(\d+)\s+(\d{4}-\d\d-\d\d)\s+(edgar/\S+)", line)
        if not m: continue
        name, cik, date, path = m.groups()
        base = "https://www.sec.gov/Archives/" + path.replace("-", "").replace(".txt", "") + "/"
        txt = get("https://www.sec.gov/Archives/" + path); time.sleep(0.15)
        t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", txt[:3_000_000])))
        if not re.search(r"odd lot", t, re.I): continue
        prio = bool(re.search(r"odd lot[^.]{0,200}(priority|without proration|not be subject to proration)|first[^.]{0,120}odd lot", t, re.I))
        px = re.search(r"not (?:greater|more) than \$\s?([\d,.]+)[^$]{0,60}(?:nor|or|and not) less than \$\s?([\d,.]+)", t)
        if not px:
            p2 = re.search(r"not less than \$\s?([\d,.]+)[^$]{0,60}not (?:more|greater) than \$\s?([\d,.]+)", t)
            px = p2 and re.match(r"(.*)\|(.*)", p2.group(2) + "|" + p2.group(1))
        fx = re.search(r"at a (?:purchase )?price of \$\s?([\d,.]+) per share", t)
        exp = re.search(r"expire[sd]? at[^.]{0,80}?on ([A-Z][a-z]+ \d{1,2}, \d{4})", t)
        rows.append(dict(filed=date, company=name, cik=cik, odd_lot_priority=prio,
                         price_high=px.group(1).rstrip(".,") if px else (fx.group(1).rstrip(".,") if fx else ""),
                         price_low=px.group(2).rstrip(".,") if px else "",
                         expires=exp.group(1) if exp else "", url=base))
        print(rows[-1], flush=True)
with open("data/tenders.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
