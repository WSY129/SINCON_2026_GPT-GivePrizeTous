"""data.go.kr 검색 결과를 키워드별로 전부 긁어 CSV로 저장 (파일데이터 + 오픈API)."""
import csv, html, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

OUT = Path(sys.argv[1])
KEYWORDS = sys.argv[2].split("|")
TYPES = sys.argv[3].split(",") if len(sys.argv) > 3 else ["FILE", "API"]
MAXPAGE = int(sys.argv[4]) if len(sys.argv) > 4 else 400

ITEM_RE = re.compile(r'<div class="apply-result-item">(.*?)(?=<div class="apply-result-item">|<nav|class="pagination)', re.S)


def clean(s):
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(re.sub(r"\s+", " ", s)).strip()


def fetch(kw, dtype, page):
    import subprocess
    q = urllib.parse.urlencode({"keyword": kw, "dType": dtype, "perPage": 40, "currentPage": page})
    url = "https://www.data.go.kr/tcs/dss/selectDataSetList.do?" + q
    for i in range(3):
        r = subprocess.run(["curl", "-s", "-m", "40", "-A", "Mozilla/5.0", url], capture_output=True)
        if r.returncode == 0 and len(r.stdout) > 10000:
            return r.stdout.decode("utf-8", "ignore")
        time.sleep(2 + i * 3)
    return ""


def parse(page_html):
    rows = []
    for b in page_html.split('<div class="apply-result-item">')[1:]:
        b = b[:6000]
        a = re.search(r'href="/data/(\d+)/(fileData|openapi|standard)\.do"[^>]*>(.*?)</a>', b, re.S)
        if not a:
            continue
        cats = [clean(x) for x in re.findall(r'<span class="krds-badge small[^"]*">(.*?)</span>', b, re.S)]
        fmts = [clean(x) for x in re.findall(r'<span class="krds-badge (?:file-ext[^"]*|bg-light-primary)"[^>]*>(.*?)</span>', b, re.S)]
        summ = re.search(r'<span class="apply-result-summary">(.*?)</span>', b, re.S)
        org = re.search(r'<strong>제공기관</strong>(.*?)</li>', b, re.S)
        upd = re.search(r'<strong>수정일</strong>(.*?)</li>', b, re.S)
        rows.append({
            "id": a.group(1), "kind": a.group(2), "title": clean(a.group(3)),
            "org": clean(org.group(1)) if org else "", "updated": clean(upd.group(1)) if upd else "",
            "category": "/".join(cats), "formats": "/".join(fmts),
            "summary": clean(summ.group(1))[:400] if summ else "",
        })
    return rows


def total(page_html, dtype):
    lab = {"FILE": "파일데이터", "API": "오픈 API"}[dtype]
    m = re.search(lab + r" <span>([0-9,]+)건", page_html)
    return int(m.group(1).replace(",", "")) if m else 0


seen = {}
if OUT.exists():
    for r in csv.DictReader(OUT.open(encoding="utf-8")):
        seen[r["id"]] = r
for kw in KEYWORDS:
    for dt in TYPES:
        p1 = fetch(kw, dt, 1)
        n = total(p1, dt)
        pages = min(MAXPAGE, (n + 39) // 40)
        got = 0
        for pg in range(1, pages + 1):
            h = p1 if pg == 1 else fetch(kw, dt, pg)
            rs = parse(h)
            if not rs:
                break
            for r in rs:
                r["kw"] = kw
                if r["id"] in seen:
                    if kw not in seen[r["id"]]["kw"].split(";"):
                        seen[r["id"]]["kw"] += ";" + kw
                else:
                    seen[r["id"]] = r
            got += len(rs)
            time.sleep(0.3)
        print(f"{kw}\t{dt}\ttotal={n}\tgot={got}\tuniq={len(seen)}", flush=True)
cols = ["id", "kind", "title", "org", "updated", "category", "formats", "kw", "summary"]
with OUT.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, cols)
    w.writeheader()
    for r in seen.values():
        w.writerow({c: r.get(c, "") for c in cols})
