"""인천문화예술회관 공연 상세 페이지에서 시작 시각·관람(러닝)시간 받기 (2025-01 ~ 2026-12 공연).
목록 페이지(raw/artcenter/*.html)의 상세 링크(progrmSn)를 따라가요. 키 필요 없음.
Output: data/external/artcenter_incheon_performances.csv 에 start_time, running_min 열을 더해요.
  start_time: 상세 페이지 '시간' 칸 그대로 (여러 회차면 '14:00, 19:00' 같이 적혀 있어요)
  start_hour_first: 그중 첫 시각의 시(정수), running_min: '관람시간/러닝타임 N분'이 있으면 N
"""
import html, re, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/raw/artcenter"
DET = RAW / "detail"
DET.mkdir(exist_ok=True)
ROW = re.compile(r'<a href="/art/ART010101/view\?[^"]*progrmSn=(\d+)[^"]*">(.*?)</a></td>\s*<td[^>]*>\s*(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})\s*</td>\s*<td[^>]*>(.*?)</td>', re.S)
clean = lambda s: html.unescape(re.sub(r"<[^>]+>|\s+", " ", s)).strip()

rows = []
for f in sorted(RAW.glob("20*_p*.html")):
    if f.name[:4] not in ("2025", "2026"):
        continue
    for sn, t, a, b, hall in ROW.findall(f.read_text(errors="ignore")):
        rows.append({"progrmSn": sn, "title": clean(t), "start_date": a, "end_date": b, "hall": clean(hall)})
lst = pd.DataFrame(rows).drop_duplicates("progrmSn")


def get(sn):
    p = DET / f"{sn}.html"
    if not p.exists() or p.stat().st_size < 3000:
        url = f"https://www.incheon.go.kr/art/ART010101/view?resveProgrmSeCode=C&resveGroupSn=5&progrmSn={sn}"
        p.write_text(subprocess.run(["curl", "-s", "-m", "40", "-A", "Mozilla/5.0", url], capture_output=True, text=True).stdout)
    t = re.sub(r"\s+", " ", re.sub(r"<script.*?</script>|<style.*?</style>|<[^>]+>", " ", p.read_text(errors="ignore"), flags=re.S))
    m = re.search(r"시간 (.{1,60}?) 장소", t)
    tm = m.group(1).strip() if m else ""
    r = re.search(r"(?:관람시간|공연시간|러닝타임|소요시간)\s*[:：]?\s*(?:약\s*)?(\d{2,3})\s*분", t)
    return sn, tm, int(r.group(1)) if r else None


with ThreadPoolExecutor(8) as ex:
    got = list(ex.map(get, lst.progrmSn))
det = pd.DataFrame(got, columns=["progrmSn", "start_time", "running_min"])
lst = lst.merge(det, on="progrmSn")
lst["start_hour_first"] = lst.start_time.str.extract(r"(\d{1,2})\s*[:시]")[0].astype("Int64")

full = pd.read_csv(ROOT / "data/external/artcenter_incheon_performances.csv")
full = full.drop(columns=[c for c in ("start_time", "running_min", "start_hour_first") if c in full.columns])
full = full.merge(lst[["start_date", "end_date", "title", "hall", "start_time", "start_hour_first", "running_min"]],
                  on=["start_date", "end_date", "title", "hall"], how="left")
full.to_csv(ROOT / "data/external/artcenter_incheon_performances.csv", index=False)
x = full[full.start_date >= "2025-01-01"]
print(len(x), "공연(2025~26) 중 시작 시각", x.start_time.notna().sum(), "/ 러닝타임", x.running_min.notna().sum())
print(x[x.hall == "대공연장"].start_hour_first.value_counts().sort_index().to_dict())
print(x[(x.hall == "대공연장") & (x.start_date >= "2026-07-01") & (x.start_date <= "2026-10-31")][["start_date", "title", "start_time", "running_min"]].to_string(index=False))
