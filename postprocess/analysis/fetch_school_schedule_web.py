"""학교 홈페이지(인천 icehs.kr / icems.kr 통합 홈페이지)의 '월간 일정'에서 학사일정을 받는다.
NEIS에 시험 날짜를 안 올리는 학교가 있어서 (예: 서인천고 — 검바위역 143 m) 홈페이지에서 직접 받는다.
대상: data/external/schools_near_exam_stations.csv 의 학교 (시험 날 신호가 있는 8개 역 700 m 안)
기간: 2024학년도 2학기 ~ 2026학년도 2학기 (2025-01 ~ 2027-02)
Output:
  data/external/raw/school_web/<학교코드>_<연도>_<학기>.html   원본
  data/external/school_schedule_web.csv   학교 × 날짜 × 행사 (2025 · 2026, 1·2학기)
사용: .venv/bin/python -I analysis/fetch_school_schedule_web.py
"""
import html, re, subprocess, time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data/external"
RAW = EXT / "raw/school_web"
RAW.mkdir(parents=True, exist_ok=True)
UA = "Mozilla/5.0"


def get(url, form=None):
    cmd = ["curl", "-s", "-L", "-m", "30", "-A", UA]
    for k, v in (form or {}).items():
        cmd += ["-F", f"{k}={v}"]
    return subprocess.run(cmd + [url], capture_output=True, text=True, errors="ignore").stdout


def text_of(t):
    t = re.sub(r"<script.*?</script>", "", t, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)))


def parse(txt):
    """'2026년 7월의 달력 테이블입니다. ... 행사내용 01일~07일 : 2회고사 08일 : ...' → (날짜, 행사) 목록"""
    rows = []
    blocks = re.finditer(r"(\d{4})년 (\d{1,2})월의 달력 테이블입니다\..*?행사내용(.*?)(?=\d{4}년 \d{1,2}월 \d{4}년 \d{1,2}월의 달력|행사일정은|$)", txt)
    for b in blocks:
        y, m, body = int(b.group(1)), int(b.group(2)), b.group(3)
        for e in re.finditer(r"(\d{2})일(?:~(?:(\d{2})월)?(\d{2})일)? : (.*?)(?= \d{2}일(?:~(?:\d{2}월)?\d{2}일)? : |$)", body):
            d1 = pd.Timestamp(y, m, int(e.group(1)))
            if e.group(3):
                m2 = int(e.group(2)) if e.group(2) else m
                y2 = y + 1 if m2 < m else y
                d2 = pd.Timestamp(y2, m2, int(e.group(3)))
            else:
                d2 = d1
            if (d2 - d1).days > (80 if "방학" in e.group(4) else 40):  # 너무 긴 기간은 첫날만 (방학은 80일까지 펼침)
                d2 = d1
            for d in pd.date_range(d1, d2):
                rows.append((d.strftime("%Y-%m-%d"), e.group(4).strip()))
    return rows


sc = pd.read_csv(EXT / "schools_near_exam_stations.csv")
neis = pd.read_csv(EXT / "raw/neis/schools.csv", dtype=str)[["SCHUL_NM", "SD_SCHUL_CODE", "HMPG_ADRES"]]
sc = sc.merge(neis, left_on="school", right_on="SCHUL_NM", how="left")
out = []
for r in sc.drop_duplicates("school").itertuples():
    host = re.sub(r"^https?://", "", str(r.HMPG_ADRES)).split("/")[0]
    main = get(f"https://{host}/main.do") or get(f"http://{host}/main.do")
    link = re.search(r"/schdList\.do\?m=([0-9A-Za-z]+)&(?:amp;)?s=([0-9A-Za-z_]+)", main)
    if not link:
        print("월간 일정 링크 없음:", r.school, host)
        continue
    m, s = link.groups()
    n = 0
    for y, sec in ((2024, 2), (2025, 1), (2025, 2), (2026, 1), (2026, 2)):  # 2024-2학기 = 2025년 1 ~ 2월
        if True:
            f = RAW / f"{r.SD_SCHUL_CODE}_{y}_{sec}.html"
            if not f.exists():
                f.write_text(get(f"https://{host}/schdList.do?section={sec}&m={m}&s={s}",
                                 {"section": sec, "year": y, "workType": "H"}))
                time.sleep(0.5)
            for d, ev in parse(text_of(f.read_text(errors="ignore"))):
                out.append(dict(station_id=r.station_id, station_name=r.station_name, school=r.school,
                                school_code=r.SD_SCHUL_CODE, dist_m=r.dist_m, date=d, event=ev,
                                source=f"https://{host}/schdList.do?m={m}&s={s} (월간 일정)"))
                n += 1
    print(r.school, host, n, "줄")
df = pd.DataFrame(out).drop_duplicates(["school", "date", "event"]).sort_values(["station_id", "school", "date"])
df.to_csv(EXT / "school_schedule_web.csv", index=False)
print("저장:", (EXT / "school_schedule_web.csv").relative_to(ROOT), len(df), "줄")
