"""인천문화예술회관 공연 일정 (역: 예술회관 3125) — 인천시 문화예술회관 누리집 '공연일정' 목록 (키 필요 없음).
https://www.incheon.go.kr/art/ART010101/index?srchYear=YYYY&srchMonth=MM&curPage=N
Output: data/external/artcenter_incheon_performances.csv (공연 1건 = 1행: start_date, end_date, title, hall)
"""
import re, subprocess, html
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/raw/artcenter"
RAW.mkdir(parents=True, exist_ok=True)
URL = "https://www.incheon.go.kr/art/ART010101/index?srchYear={}&srchMonth={:02d}&curPage={}"
ROW = re.compile(r"<tr>\s*<td[^>]*>(.*?)</td>\s*<td[^>]*>\s*(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})\s*</td>\s*<td[^>]*>(.*?)</td>", re.S)


def get(y, m, p):
    f = RAW / f"{y}{m:02d}_p{p}.html"
    if not f.exists() or f.stat().st_size < 5000:
        out = subprocess.run(["curl", "-s", "-L", "-m", "60", "-A", "Mozilla/5.0", URL.format(y, m, p)],
                             capture_output=True, text=True).stdout
        f.write_text(out)
    return f.read_text()


rows = []
for y in range(2015, 2027):
    for m in range(1, 13):
        if (y, m) > (2026, 12):
            break
        h = get(y, m, 1)
        pages = max([int(x) for x in re.findall(r"curPage=(\d+)", h)] or [1])
        for p in range(1, pages + 1):
            hp = h if p == 1 else get(y, m, p)
            for t, a, b, hall in ROW.findall(hp):
                clean = lambda s: html.unescape(re.sub(r"<[^>]+>|\s+", " ", s)).strip()
                rows.append({"start_date": a, "end_date": b, "title": clean(t), "hall": clean(hall), "list_month": f"{y}-{m:02d}"})
d = pd.DataFrame(rows).drop_duplicates(["start_date", "end_date", "title", "hall"]).sort_values("start_date")
d.drop(columns="list_month").to_csv(ROOT / "data/external/artcenter_incheon_performances.csv", index=False)
print(len(d), d.start_date.min(), d.start_date.max())
print(d.groupby(d.start_date.str[:4]).size().to_dict())
print(d.hall.value_counts().head(8).to_dict())
