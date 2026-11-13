"""KOPIS(공연예술통합전산망) 공연통계 → 공연시설별: 공연장 × 하루 티켓 판매수.

출처 화면: https://kopis.or.kr/por/stats/perfo/perfoStatsPerfoByFct.do?menuId=MNU_00033 ('일간')
화면이 쓰는 GraphQL(perfoStatsPerfoByFctList)을 그대로 부른다. 키 필요 없음.
날짜는 YYYYMMDD, 한 쪽에 20줄, 티켓 판매수 내림차순. 공연일자 기준. 2019-06-25 전송 의무화 이후가 온전함.

단계
  python analysis/fetch_kopis_venue_daily.py venues            # 2019-07~2026-10 전체 공연장 목록(기간 합계)
  python analysis/fetch_kopis_venue_daily.py daily 2025-01 2026-10   # 고른 공연장만 월 → 일 순서로 조회

응답은 data/external/raw/kopis/ 에 저장해서 다시 돌리면 이어서 받는다. 요청 사이 0.3초 쉰다.
"""
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/raw/kopis"
RAW.mkdir(parents=True, exist_ok=True)
URL = "https://kopis.or.kr:9001/api/prs/graphql"
Q = """query GetPerfoStatsFacilityList($startDate: String!, $endDate: String!, $PRFR_FCLTY_ID: String!, $curPage: Int!) {
 perfoStatsPerfoByFctList(startDate: $startDate, endDate: $endDate, PRFR_FCLTY_ID: $PRFR_FCLTY_ID, curPage: $curPage) {
  result { data1 data2 data3 data4 data5 data6 data8 data9 totalCnt }
  resultSub { data1 data2 data3 data4 data5 data6 data7 data8 data9 data14 data15 }
  curDate postDate searchDate } }"""
SLEEP = 0.3


def call(start, end, fid="", page=1):
    """start/end: YYYYMMDD. 같은 요청은 파일로 저장해 두고 다시 쓰지 않는다."""
    f = RAW / f"{fid or 'ALL'}_{start}_{end}_p{page}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except json.JSONDecodeError:  # 중간에 끊겨 반쯤 쓰인 파일이면 다시 받기
            f.unlink()
    body = json.dumps({"operationName": "GetPerfoStatsFacilityList",
                       "variables": {"startDate": start, "endDate": end, "PRFR_FCLTY_ID": fid, "curPage": page},
                       "query": Q})
    cmd = ["curl", "-s", "-m", "60", "-X", "POST", URL, "-A", "Mozilla/5.0",
           "-H", "Content-Type: application/json", "-H", "Origin: https://kopis.or.kr",
           "-H", "Referer: https://kopis.or.kr/", "--data-binary", body]
    for attempt in range(3):
        try:
            d = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
            r = d["data"]["perfoStatsPerfoByFctList"]
            if r is None:
                raise RuntimeError(d.get("errors"))
            f.write_text(json.dumps(r, ensure_ascii=False))
            time.sleep(SLEEP)
            return r
        except Exception as e:  # 잠깐 쉬고 다시
            print("retry", start, end, fid, page, e, file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed {start} {end} {fid} {page}")


def all_pages(start, end, fid=""):
    r = call(start, end, fid, 1)
    rows = list(r["result"] or [])
    subs = list(r["resultSub"] or [])
    tot = rows[0]["totalCnt"] if rows else 0
    p = 2
    while len(rows) < tot:
        rr = call(start, end, fid, p)
        if not rr["result"]:
            break
        rows += rr["result"]
        subs += rr["resultSub"] or []
        p += 1
    return rows, subs


def active(r):
    """그 기간에 공연(상연) 또는 티켓이 하나라도 있으면 True."""
    res = r["result"] or []
    return bool(res) and (int(res[0]["data5"] or 0) > 0 or int(res[0]["data6"] or 0) > 0)


def venue_days(fid, months):
    """월 → 열흘 → 하루 순서로 좁혀서, 공연이 있는 날만 하루 단위로 받는다."""
    import calendar
    out = []
    for ym in months:
        y, m = int(ym[:4]), int(ym[5:7])
        last = calendar.monthrange(y, m)[1]
        if not active(call(f"{y}{m:02d}01", f"{y}{m:02d}{last:02d}", fid)):
            continue
        for a, b in ((1, 10), (11, 20), (21, last)):
            if not active(call(f"{y}{m:02d}{a:02d}", f"{y}{m:02d}{b:02d}", fid)):
                continue
            for d in range(a, b + 1):
                ds = f"{y}{m:02d}{d:02d}"
                r = call(ds, ds, fid)
                if active(r):
                    out.append((f"{y}-{m:02d}-{d:02d}", r))
    return out


def month_list(a, b):
    import pandas as pd
    return [p.strftime("%Y-%m") for p in pd.period_range(a, b, freq="M")]


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "venues":
        rows, subs = all_pages("20190701", "20261031")
        (ROOT / "data/external/raw/kopis_venues_2019_2026.json").write_text(
            json.dumps({"result": rows, "resultSub": subs}, ensure_ascii=False))
        print("venues", len(rows), "halls", len(subs))
    elif step == "daily":
        import pandas as pd
        SLEEP = 0.15  # 공연장 하나·짧은 기간 조회는 0.05초 정도로 가벼움
        months = month_list(sys.argv[2], sys.argv[3])
        vmap = pd.read_csv(ROOT / "data/external/kopis_venue_station_map.csv", dtype=str)
        if len(sys.argv) > 4:  # 나눠서 동시에: daily 2025-01 2026-10 0/3
            k, n = map(int, sys.argv[4].split("/"))
            vmap = vmap.iloc[k::n]
        for _, v in vmap.iterrows():
            t0 = time.time()
            days = venue_days(v.fclty_id, months)
            print(v.venue, len(days), "days", f"{time.time() - t0:.0f}s", flush=True)
    elif step == "build":
        # 저장된 하루 단위 응답만 모아서 CSV 두 개로 만든다
        import glob, re
        import pandas as pd
        vmap = pd.read_csv(ROOT / "data/external/kopis_venue_station_map.csv", dtype=str).set_index("fclty_id")
        vrows, hrows = [], []
        for f in glob.glob(str(RAW / "FC*_*_*_p1.json")):
            fid, s, e = re.match(r".*/(FC\d+)_(\d{8})_(\d{8})_p1\.json", f).groups()
            if s != e or fid not in vmap.index:
                continue
            r = json.loads(Path(f).read_text())
            if not active(r):
                continue
            x = r["result"][0]
            date = f"{s[:4]}-{s[4:6]}-{s[6:]}"
            vrows.append({"date": date, "fclty_id": fid, "venue": x["data2"],
                          "station_ids": vmap.at[fid, "station_ids"], "group": vmap.at[fid, "group"],
                          "seats": int(x["data3"] or 0), "perf_cnt": int(x["data9"] or 0),
                          "open_cnt": int(x["data4"] or 0), "shows": int(x["data5"] or 0),
                          "tickets": int(x["data6"] or 0)})
            for h in r["resultSub"] or []:
                if int(h["data7"] or 0) == 0 and int(h["data8"] or 0) == 0:
                    continue
                hrows.append({"date": date, "fclty_id": fid, "venue": x["data2"], "hall_id": h["data2"],
                              "hall": h["data14"] or h["data3"], "hall_seats": int(h["data15"] or 0),
                              "perf_cnt": int(h["data9"] or 0), "shows": int(h["data7"] or 0),
                              "tickets": int(h["data8"] or 0)})
        v = pd.DataFrame(vrows).sort_values(["date", "fclty_id"])
        h = pd.DataFrame(hrows).sort_values(["date", "fclty_id", "hall_id"])
        v.to_csv(ROOT / "data/external/kopis_venue_daily.csv", index=False)
        h.to_csv(ROOT / "data/external/kopis_hall_daily.csv", index=False)
        print(len(v), "venue-days", v.date.min(), v.date.max(), "|", len(h), "hall-days")
        print(v.groupby("venue").agg(days=("date", "size"), tickets=("tickets", "sum"),
                                     max_day=("tickets", "max")).sort_values("tickets", ascending=False))
