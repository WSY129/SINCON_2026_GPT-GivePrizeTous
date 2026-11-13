"""인천·부천 중학교·고등학교 학사일정 (방학식·개학식 등) — NEIS 교육정보 개방 포털.
인증키 없이 부르면 한 번에 5줄만 줘서, 방학이 걸치는 기간을 '1주 단위'로 나눠 부르고
5줄이 꽉 차면 하루 단위로 다시 불러요. 인증키(open.neis.go.kr, 무료·즉시 발급)가 생기면 NEIS_KEY 에 넣으면 돼요.
Output:
  data/external/raw/neis/schools.csv            학교 목록 (주소 포함)
  data/external/school_events_2025_2026.csv     학교 × 날짜 × 행사
  data/external/school_vacation_daily.csv       날짜 × 구별 '방학 중인 학교 비율'
"""
import json, os, subprocess, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/raw/neis"
RAW.mkdir(parents=True, exist_ok=True)
KEY = os.environ.get("NEIS_KEY", "")
BASE = "https://open.neis.go.kr/hub/"


def call(svc, **p):
    q = {"Type": "json", "pIndex": 1, "pSize": 1000 if KEY else 5, **p}
    if KEY:
        q["KEY"] = KEY
    url = BASE + svc + "?" + urllib.parse.urlencode(q)
    for _ in range(3):
        try:  # 이 맥의 파이썬 인증서 저장소 문제로 urllib 대신 curl 사용 (다른 수집 프로그램과 같은 방식)
            d = json.loads(subprocess.run(["curl", "-s", "-m", "30", url], capture_output=True, text=True).stdout)
            if svc not in d:
                return 0, []
            return d[svc][0]["head"][0]["list_total_count"], d[svc][1]["row"]
        except Exception:
            time.sleep(1)
    return -1, []


# 1) 학교 목록: 학교 종류·지역을 잘게 나눠서 5줄 제한을 피하기 어려우므로 학교정보는 시·구 단위 반복 대신
#    학교명 앞글자 없이 받는다 → 키 없으면 5개뿐이라, 학교 목록만은 페이지별 단건 조회가 필요.
def school_list():
    f = RAW / "schools.csv"
    if f.exists():
        return pd.read_csv(f, dtype=str)
    rows = []
    for office in ["E10", "J10"]:
        for kind in ["중학교", "고등학교"]:
            if KEY:
                n, r = call("schoolInfo", ATPT_OFCDC_SC_CODE=office, SCHUL_KND_SC_NM=kind)
                rows += r
                continue
            # 키 없이: 학교 코드 범위를 하나씩 조회해요 (코드 1개 = 1줄이라 5줄 제한에 안 걸려요)
            # 인천: 고 7310xxx, 중·초 73[2-7]1xxx / 부천: 중 7581xxx, 경기 고 7530xxx~7531xxx (주소로 부천만 남김)
            if kind == "고등학교":
                codes = range(7310000, 7311000) if office == "E10" else range(7530000, 7531300)
            else:
                codes = ([c for b in (7321, 7331, 7341, 7351, 7361, 7371) for c in range(b * 1000, b * 1000 + 500)]
                         if office == "E10" else range(7581000, 7581300))
            with ThreadPoolExecutor(16) as ex:
                res = ex.map(lambda c: call("schoolInfo", ATPT_OFCDC_SC_CODE=office, SD_SCHUL_CODE=str(c))[1], codes)
            rows += [x for r in res for x in r if x.get("SCHUL_KND_SC_NM") == kind]
    df = pd.DataFrame(rows).drop_duplicates("SD_SCHUL_CODE")
    df = df[(df.ATPT_OFCDC_SC_CODE == "E10") | df.ORG_RDNMA.str.contains("부천", na=False)]
    df.to_csv(f, index=False)
    return df


def weeks(a, b):
    d = a
    while d <= b:
        e = min(d + timedelta(days=6), b)
        yield d, e
        d = e + timedelta(days=1)


WINDOWS = [(date(2025, 7, 7), date(2025, 9, 7)), (date(2025, 12, 15), date(2026, 3, 8)),
           (date(2026, 7, 6), date(2026, 9, 6)),
           (date(2026, 9, 7), date(2026, 10, 31))]  # 2학기: 추석 전후·10월 재량휴업일, 개교기념일, 시험


def school_events(row, windows=None):
    out = []
    o, s = row["ATPT_OFCDC_SC_CODE"], row["SD_SCHUL_CODE"]
    if KEY:
        _, r = call("SchoolSchedule", ATPT_OFCDC_SC_CODE=o, SD_SCHUL_CODE=s, AA_FROM_YMD="20250101",
                    AA_TO_YMD="20261031")
        return r
    for a, b in (windows or WINDOWS):
        for x, y in weeks(a, b):
            n, r = call("SchoolSchedule", ATPT_OFCDC_SC_CODE=o, SD_SCHUL_CODE=s,
                        AA_FROM_YMD=x.strftime("%Y%m%d"), AA_TO_YMD=y.strftime("%Y%m%d"))
            if n <= 5:
                out += r
                continue
            d = x
            while d <= y:
                _, r = call("SchoolSchedule", ATPT_OFCDC_SC_CODE=o, SD_SCHUL_CODE=s,
                            AA_FROM_YMD=d.strftime("%Y%m%d"), AA_TO_YMD=d.strftime("%Y%m%d"))
                out += r
                d += timedelta(days=1)
    return out


if __name__ == "__main__":
    sch = school_list()
    print("schools", len(sch), sch.SCHUL_KND_SC_NM.value_counts().to_dict())
    ev_f = ROOT / "data/external/school_events_2025_2026.csv"
    # 이미 받은 파일이 있으면, 아직 한 줄도 없는 기간만 새로 받아서 이어 붙여요 (키 없을 때)
    old = pd.read_csv(ev_f, dtype=str) if ev_f.exists() and not KEY else None
    todo = WINDOWS if old is None else [
        (a, b) for a, b in WINDOWS if not old.date.between(a.isoformat(), b.isoformat()).any()]
    print("windows to fetch", todo)
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda r: school_events(r, todo), sch.to_dict("records")))
    ev = pd.DataFrame([x for r in res for x in r])
    keep = ["SD_SCHUL_CODE", "SCHUL_NM", "SCHUL_CRSE_SC_NM", "AA_YMD", "EVENT_NM", "SBTR_DD_SC_NM"]
    ev = ev[keep].drop_duplicates()
    ev["date"] = pd.to_datetime(ev.AA_YMD).dt.strftime("%Y-%m-%d")
    ev = ev.merge(sch[["SD_SCHUL_CODE", "ORG_RDNMA", "ORG_RDNDA"]], on="SD_SCHUL_CODE", how="left")
    ev = ev.drop(columns="AA_YMD")
    if old is not None:
        ev = pd.concat([old, ev]).drop_duplicates().sort_values(["SD_SCHUL_CODE", "date"])
    ev.to_csv(ev_f, index=False)
    print("events", len(ev))
