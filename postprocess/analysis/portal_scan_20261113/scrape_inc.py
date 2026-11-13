"""인천데이터포털(data.incheon.go.kr) KONAN 검색 API로 테이블 전체를 JSON 줄로 저장."""
import json, subprocess, sys, time
from pathlib import Path

OUT = Path(sys.argv[1])
TABLES = sys.argv[2].split(",")
ORDER = sys.argv[3] if len(sys.argv) > 3 else ""
PL = 200

with OUT.open("w", encoding="utf-8") as f:
    for tb in TABLES:
        off, total = 0, None
        while True:
            q = f"select=*&from={tb}&pagelength={PL}&offset={off}"
            if ORDER:
                q = f"select=*&from={tb}&{ORDER}&pagelength={PL}&offset={off}"
            body = json.dumps({"queryString": q})
            r = subprocess.run(["curl", "-s", "-m", "60", "-A", "Mozilla/5.0", "-X", "POST",
                                "-H", "Content-Type: application/json;charset=UTF-8", "-H", "Accept: application/json",
                                "https://data.incheon.go.kr/api/board/search", "--data", body], capture_output=True)
            try:
                d = json.loads(r.stdout)
            except Exception:
                print("bad", tb, off, r.stdout[:200]); break
            res = d.get("result") or {}
            total = res.get("totalCount", 0) if total is None else total
            rows = res.get("rows") or []
            for row in rows:
                fl = row.get("fields", {})
                fl["_table"] = tb
                f.write(json.dumps(fl, ensure_ascii=False) + "\n")
            off += PL
            print(tb, off, total, len(rows), flush=True)
            if not rows or off >= int(total):
                break
            time.sleep(0.3)
