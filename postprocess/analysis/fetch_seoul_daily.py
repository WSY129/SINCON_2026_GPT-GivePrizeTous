"""서울 열린데이터광장 OA-12914 '지하철호선별 역별 승하차 인원'(일별, 교통카드) 파일 받기.
2015~2022는 연도 파일, 2023-01부터는 월 파일. ⚠️ 2026-07 이후 파일은 대회 규칙상 받지 않아요 (LAST = 2026-06).
Output: data/external/raw/seoul_daily/CARD_SUBWAY_MONTH_*.csv (cp949 원본)
"""
import re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/raw/seoul_daily"
RAW.mkdir(parents=True, exist_ok=True)
PAGE = "https://data.seoul.go.kr/dataList/OA-12914/S/1/datasetView.do"
LAST = "202606"

html = subprocess.run(["curl", "-s", "-m", "60", "-A", "Mozilla/5.0", PAGE], capture_output=True, text=True).stdout
files = dict((name, seq) for seq, name in re.findall(r"downloadFile\('(\d+)'\);\">(CARD_SUBWAY_MONTH_\d+)\.csv", html))
for name in sorted(files):
    tag = name.rsplit("_", 1)[1]
    if len(tag) == 6 and tag > LAST:
        print("skip (금지 기간)", name)
        continue
    f = RAW / f"{name}.csv"
    if f.exists() and f.stat().st_size > 100_000:
        continue
    subprocess.run(["curl", "-s", "-m", "300", "-A", "Mozilla/5.0", "-e", PAGE, "-d",
                    f"infId=OA-12914&seqNo=&seq={files[name]}&infSeq=3",
                    "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false", "-o", str(f)])
    print(name, f.stat().st_size)
