"""문화체육관광부 '지역축제' 목록(인천 전체 + 경기 부천)을 받아 CSV로 저장.

출처: https://www.mcst.go.kr/site/s_culture/festival/festivalList.jsp (키 필요 없음)
출력: data/external/raw/mcst_festivals/festivals_list.csv
  seq, title, period_text, start_date, end_date, time_text, place(시·구까지만 나옴), url

사용: python analysis/fetch_mcst_festivals.py [시작 YYYY.MM.DD.] [끝 YYYY.MM.DD.]
"""
import csv
import html
import re
import sys
import time
import subprocess
from pathlib import Path

BASE = "https://www.mcst.go.kr/site/s_culture/festival/"
OUT = Path(__file__).resolve().parents[1] / "data/external/raw/mcst_festivals"
UA = {"User-Agent": "Mozilla/5.0"}


def get(url):
    # 이 사이트는 파이썬 기본 인증서 검증에 실패해서 curl로 받아요.
    for i in range(3):
        r = subprocess.run(["curl", "-sL", "-m", "30", "-A", UA["User-Agent"], url], capture_output=True)
        if r.returncode == 0 and r.stdout:
            return r.stdout.decode("utf-8", "ignore")
        time.sleep(2 + i * 2)
    return ""


def clean(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def parse_period(txt):
    """'2026. 10. 9. ~ 10. 10. | 12:00 ~ 21:00' → (2026-10-09, 2026-10-10, '12:00 ~ 21:00')"""
    date_part, _, time_part = txt.partition("|")
    nums = re.findall(r"(?:(\d{4})\.\s*)?(\d{1,2})\.\s*(\d{1,2})\.?", date_part)
    if not nums:
        return "", "", time_part.strip()
    y = None
    out = []
    for yy, m, d in nums[:2]:
        y = int(yy) if yy else y
        if y is None:
            return "", "", time_part.strip()
        out.append(f"{y:04d}-{int(m):02d}-{int(d):02d}")
    if len(out) == 1:
        out.append(out[0])
    return out[0], out[1], time_part.strip()


def fetch_list(sido, from_dt, to_dt):
    rows, page = [], 1
    while True:
        t = get(f"{BASE}festivalList.jsp?pSido={sido}&fromDt={from_dt}&toDt={to_dt}&pCurrentPage={page}")
        items = re.findall(r'<a href="festivalView\.jsp\?pSeq=(\d+)&amp;pRo=(\d+)".*?</a>', t, re.S)
        blocks = re.findall(r'(<a href="festivalView\.jsp\?pSeq=\d+&amp;pRo=\d+".*?</a>)', t, re.S)
        if not blocks:
            break
        for b in blocks:
            seq = re.search(r"pSeq=(\d+)", b).group(1)
            title = clean(re.search(r'class="sub-tit">(.*?)</p>', b, re.S).group(1))
            period = clean(re.search(r"기간</strong>\s*:(.*?)</li>", b, re.S).group(1))
            place = clean(re.search(r"장소</strong>\s*:(.*?)</li>", b, re.S).group(1))
            s, e, tm = parse_period(period)
            rows.append(dict(seq=seq, title=title, period_text=period, start_date=s, end_date=e,
                             time_text=tm, place=place,
                             url=f"{BASE}festivalView.jsp?pSeq={seq}"))
        if len(blocks) < 5 or page > 80 or not items:
            break
        page += 1
        time.sleep(0.3)
    return rows


def main():
    from_dt = sys.argv[1] if len(sys.argv) > 1 else "2025.01.01."
    to_dt = sys.argv[2] if len(sys.argv) > 2 else "2026.10.31."
    OUT.mkdir(parents=True, exist_ok=True)
    rows = fetch_list("04", from_dt, to_dt)
    rows += [r for r in fetch_list("10", from_dt, to_dt) if "부천" in r["place"]]
    seen, uniq = set(), []
    for r in rows:
        if r["seq"] not in seen:
            seen.add(r["seq"])
            uniq.append(r)
    uniq.sort(key=lambda r: (r["start_date"], r["title"]))
    with open(OUT / "festivals_list.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(uniq[0].keys()))
        w.writeheader()
        w.writerows(uniq)
    print(len(uniq), "festivals →", OUT / "festivals_list.csv")


if __name__ == "__main__":
    main()
