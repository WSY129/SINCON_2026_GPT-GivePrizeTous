"""인천교통공사 사전정보 공개자료실 '지하철 시간대별 통행량' 과거 자료 다운로드.

목록: data/external/ictr_hourly_posts_index.csv (status == '과거 자료(사용 가능)' 만 받는다)
저장: data/external/raw/ictr_hourly/{YYYY-MM}_{msg_seq}{확장자}

대회 규칙: 2026-07 ~ 2026-10 지하철 승하차는 금지 → 이 스크립트는 2026-06 이전만 받는다.
"""
import csv
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data/external/ictr_hourly_posts_index.csv"
OUT = ROOT / "data/external/raw/ictr_hourly"
BANNED = ("2026-07", "2026-08", "2026-09", "2026-10")


def filename_from_headers(headers):
    cd = {k.lower(): v for k, v in headers.items()}.get("content-disposition", "")
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd)
    if m:
        return urllib.parse.unquote(m.group(1))
    m = re.search(r'filename="?([^";]+)"?', cd)
    if m:
        raw = m.group(1)
        for enc in ("utf-8", "euc-kr"):
            try:
                return urllib.parse.unquote(raw.encode("latin-1").decode(enc))
            except (UnicodeDecodeError, UnicodeEncodeError):
                continue
        return urllib.parse.unquote(raw)
    return ""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(INDEX, encoding="utf-8-sig")))
    # 기본: 과거 자료만. --overlap 이면 train 기간(2025-01~2026-06)과 겹치는 18개도 받는다 (train 검증용).
    ok = {"과거 자료(사용 가능)"} | ({"train 기간과 겹침"} if "--overlap" in sys.argv else set())
    targets = [r for r in rows if r["status"] in ok]
    log = []
    for r in targets:
        ym, seq, url = r["data_month"], r["msg_seq"], r["file_url"]
        assert ym <= "2026-06" and ym not in BANNED, ym  # test 기간 지하철 승하차 금지
        if list(OUT.glob(f"{ym}_{seq}.*")):
            continue
        # python.org 파이썬은 인증서 묶음이 없어 SSL 오류 → 시스템 curl 사용
        tmp = OUT / "_tmp.bin"
        hdr = subprocess.run(
            ["curl", "-sS", "-f", "-A", "Mozilla/5.0", "-D", "-", "-o", str(tmp), url],
            capture_output=True, check=True, timeout=120,
        ).stdout.decode("latin-1")
        headers = dict(
            line.split(":", 1) for line in hdr.splitlines() if ":" in line
        )
        headers = {k.strip(): v.strip() for k, v in headers.items()}
        orig = filename_from_headers(headers)
        data = tmp.read_bytes()
        tmp.unlink()
        ext = Path(orig).suffix.lower() or ".bin"
        (OUT / f"{ym}_{seq}{ext}").write_bytes(data)
        log.append([ym, seq, orig, len(data)])
        print(ym, seq, orig, len(data), flush=True)
        time.sleep(0.5)
    with open(OUT / "_download_log.csv", "a", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(log)
    print("done", len(log), file=sys.stderr)


if __name__ == "__main__":
    main()
