"""행정동별 월말 주민등록 인구 2015-01 ~ 2026-10 (인천 전체 + 부천).
Source: 행정안전부 주민등록 인구통계 (jumin.mois.go.kr) 월간현황 > 전체읍면동현황 CSV. 키 필요 없음.
Output: data/external/population_dong_monthly_2015_2026.csv (ym, adm_code, sido, sigungu, dong, pop, households)
"""
import io, re, subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/raw/population"
RAW.mkdir(parents=True, exist_ok=True)
UA = "Mozilla/5.0"
CJ = RAW / "cookies.txt"


def fetch(year, m1, m2):
    f = RAW / f"jumin_{year}{m1:02d}_{m2:02d}.csv"
    if f.exists() and f.stat().st_size > 10000:
        return f
    subprocess.run(["curl", "-s", "-m", "60", "-A", UA, "-c", str(CJ), "-b", str(CJ),
                    "https://jumin.mois.go.kr/statMonth.do", "-o", "/dev/null"], check=True)
    data = (f"sltOrgType=1&sltOrgLvl1=A&sltOrgLvl2=&gender=gender&genderPer=genderPer&generation=generation"
            f"&sltUndefType=&searchYearStart={year}&searchMonthStart={m1:02d}&searchYearEnd={year}"
            f"&searchMonthEnd={m2:02d}&sltOrderType=1&sltOrderValue=ASC&category=month&state=3")
    subprocess.run(["curl", "-s", "-m", "300", "-A", UA, "-b", str(CJ), "-e", "https://jumin.mois.go.kr/statMonth.do",
                    "https://jumin.mois.go.kr/downloadCsv.do?searchYearMonth=month&xlsStats=3",
                    "--data", data, "-o", str(f)], check=True)
    return f


def parse(f):
    df = pd.read_csv(f, encoding="cp949", dtype=str)
    df = df[df["행정구역"].str.startswith(("인천광역시", "경기도 부천시"))]
    rows = []
    for c in df.columns[1:]:
        m = re.match(r"(\d{4})년(\d{2})월_(총인구수|세대수)$", c)
        if not m:
            continue
        rows.append(pd.DataFrame({"ym": f"{m[1]}-{m[2]}", "name": df["행정구역"], "var": m[3],
                                  "val": pd.to_numeric(df[c].str.replace(",", "").str.strip(), errors="coerce")}))
    return pd.concat(rows)


frames = []
for y in range(2015, 2027):
    last = 10 if y == 2026 else 12
    frames.append(parse(fetch(y, 1, 6)))
    frames.append(parse(fetch(y, 7, last)))
long = pd.concat(frames)
w = long.pivot_table(index=["ym", "name"], columns="var", values="val", aggfunc="first").reset_index()
w = w.rename(columns={"총인구수": "pop", "세대수": "households"})
w["adm_code"] = w["name"].str.extract(r"\((\d{10})\)")
parts = w["name"].str.replace(r"\(\d{10}\)", "", regex=True).str.split()
w["sido"] = parts.str[0]
# 부천은 "경기도 부천시 [원미구] 동" 형태가 있어 마지막 토큰을 동으로 둔다
w["level"] = w["adm_code"].map(lambda c: "dong" if c[5:] != "00000" else ("sigungu" if c[2:5] != "000" else "sido"))
# 시군구 행은 이름 전체를 시군구로 ("경기도 부천시 원미구" → "부천시 원미구"), 동 행은 마지막 토큰이 동
w["sigungu"] = [(" ".join(p[1:]) if lv == "sigungu" else " ".join(p[1:-1]) if lv == "dong" else "")
                for p, lv in zip(parts, w["level"])]
w["dong"] = [p[-1] if lv == "dong" else "" for p, lv in zip(parts, w["level"])]
w = w[w["pop"] > 0]  # 폐지된 동은 0으로 나와요
w = w[["ym", "adm_code", "level", "sido", "sigungu", "dong", "pop", "households"]].sort_values(["adm_code", "ym"])
w.to_csv(ROOT / "data/external/population_dong_monthly_2015_2026.csv", index=False)
print(w.groupby("level").size(), w.ym.min(), w.ym.max())
