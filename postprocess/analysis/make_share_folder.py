"""팀 공유 폴더 만들기: README + 모은 자료 + 분석 프로그램·결과를 한 폴더(zip)로.
- 넣는 것: 데이터_공유_README.md, analysis/(프로그램·결과·보고서, deep/), outputs/analysis/(분석 표·그림·웹용 표),
          data/external/*.csv (모은 자료 전부), data/external/raw/admdong·osm (역↔행정동 다시 만들 때 필요한 작은 원본)
- 빼는 것: 대회 파일(train·test 등 — 팀이 이미 가짐), 나머지 raw 원본(약 290MB, 프로그램으로 다시 받을 수 있음), src/(모델 담당 코드), 캐시
- v5 (2026-11-13 밤) 추가: 공공데이터_추가후보_20261113.md, analysis/portal_scan_20261113/ (두 포털 조사),
  data/external/raw/datagokr_past/ (과거 일별 원본 42개 — 포털에서 내려갈 수 있어 원본도 넣음),
  analysis/bus_query_0926/ (9/26 버스 판정), outputs/v20e_0926v10/ (9/26·7월 조정 제출 파일)
"""
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "SINCON_데이터공유_v5"
DST = ROOT / NAME
if DST.exists():
    shutil.rmtree(DST)
DST.mkdir()

copied = []


def put(src, rel=None):
    src = ROOT / src
    rel = Path(rel) if rel else src.relative_to(ROOT)
    out = DST / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    copied.append(rel)


put("데이터_공유_README.md")
for f in sorted(ROOT.glob("DATA_REQUESTS_답변_*.md")):  # 팀 자료 요청에 대한 답변 (README 6-3절 (17))
    put(f.relative_to(ROOT))
for f in sorted((ROOT / "analysis").glob("*")):
    if f.is_file() and f.suffix in (".py", ".csv", ".md"):
        put(f.relative_to(ROOT))
for f in sorted((ROOT / "analysis/deep").glob("*.py")):
    put(f.relative_to(ROOT))
for f in sorted((ROOT / "outputs/analysis").rglob("*")):
    if f.is_file() and f.suffix in (".csv", ".png"):
        put(f.relative_to(ROOT))
for f in sorted((ROOT / "data/external").glob("*.csv")):
    put(f.relative_to(ROOT))
for f in sorted((ROOT / "data/external").glob("*.csv")):  # 규칙: 2026-07 이후 지하철 승하차 금지
    if f.name.startswith(("ictr_", "seoul_", "train_vs")) and f.name != "ictr_hourly_posts_index.csv":  # 게시물 목록(숫자 없음, 7월 이후는 '금지' 표시뿐)
        import pandas as pd
        cols = pd.read_csv(f, nrows=0).columns
        dc = next((c for c in ("date", "data_month", "ym") if c in cols), None)
        if dc:
            mx = pd.read_csv(f, usecols=[dc])[dc].astype(str).max()
            assert mx < "2026-07", (f.name, mx)
put("data/external/raw/ictr_other/_index.csv")  # README 11절이 가리키는 공사 게시물 목록
put("data/external/raw/mcst_festivals/festivals_list.csv")  # README 6-3절 (16)의 문체부 지역축제 목록
put("src/common.py")  # analysis/deep 프로그램이 쓰는 달력·공휴일 도우미 (모델 코드 중 이 파일만)
put("공공데이터_추가후보_20261113.md")
SKIP = {"__pycache__", ".DS_Store", ".cookies"}
for folder in ("analysis/portal_scan_20261113", "analysis/bus_query_0926", "data/external/raw/datagokr_past",
               "outputs/v20e_0926v10"):
    for f in sorted((ROOT / folder).rglob("*")):
        if f.is_file() and not (SKIP & set(f.relative_to(ROOT).parts)):
            put(f.relative_to(ROOT))
for sub in ("admdong", "osm"):
    for f in sorted((ROOT / "data/external/raw" / sub).glob("*")):
        if f.is_file():
            put(f.relative_to(ROOT))

zpath = ROOT / f"{NAME}.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for rel in copied:
        z.write(DST / rel, f"{NAME}/{rel}")
size = sum((DST / r).stat().st_size for r in copied)
print(f"{len(copied)} files, {size / 1e6:.1f} MB → {DST.name}/ , zip {zpath.stat().st_size / 1e6:.1f} MB")
