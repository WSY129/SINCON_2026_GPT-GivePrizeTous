"""공공데이터포털 인천교통공사 승하차 자료의 '주기성 과거 데이터' 파일을 모두 받는다.

입력: analysis/portal_scan_20261113/past_versions_subway.csv (과거판 목록) + 각 데이터셋의 현재판
출력: data/external/raw/datagokr_past/<데이터셋ID>/<파일>, 받은 목록 _download_log.csv
로그인·인증키 필요 없음. 2026-07 이후 자료는 없다(모두 2026-06-30 이전 기준 파일).
사용: python3 -I analysis/fetch_datagokr_past.py
"""
import csv, json, re, subprocess, time, urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "analysis/portal_scan_20261113/past_versions_subway.csv"
OUT = ROOT / "data/external/raw/datagokr_past"
JAR = OUT / ".cookies"
BASE = "https://www.data.go.kr"
DATASETS = ["15060369", "15060370", "15102674", "15004329", "15060441"]


def curl(url, data=None, out=None):
    cmd = ["curl", "-s", "-L", "-m", "300", "-A", "Mozilla/5.0", "-c", str(JAR), "-b", str(JAR)]
    if data is not None:
        cmd += ["-X", "POST", "--data", data]
    if out:
        cmd += ["-D", str(out) + ".hdr", "-o", str(out)]
    cmd.append(url)
    r = subprocess.run(cmd, capture_output=True)
    return r.stdout.decode("utf-8", "ignore")


def download(did, detail_pk, atch, sn, label):
    info = curl(BASE + "/tcs/dss/selectFileDataDownload.do", urllib.parse.urlencode(
        {"publicDataDetailPk": detail_pk, "publicDataPk": did, "atchFileId": atch,
         "fileDetailSn": sn, "publicDataTyCode": "PR0051"}))
    try:
        d = json.loads(info)
    except Exception:
        return None, "selectFileDataDownload 실패"
    if not d.get("status"):
        return None, str(d.get("error"))
    atch, sn = d["atchFileId"], d["fileDetailSn"]
    lim = curl(BASE + "/cmm/cmm/check-limit.json", f"atchFileId={atch}&fileDetailSn={sn}")
    if "true" in lim:
        return None, "captcha 필요"
    tmp = OUT / did / f"_tmp_{sn}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    curl(f"{BASE}/cmm/cmm/fileDownload.do?atchFileId={atch}&fileDetailSn={sn}&dataNm="
         + urllib.parse.quote(label), out=tmp)
    hdr = Path(str(tmp) + ".hdr").read_text(errors="ignore")
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";\r\n]+)', hdr, re.I)
    name = urllib.parse.unquote(m.group(1)) if m else label + ".bin"
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    dest = OUT / did / name
    tmp.rename(dest)
    Path(str(tmp) + ".hdr").unlink()
    return dest, "ok"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    # 현재판
    for did in DATASETS:
        h = curl(f"{BASE}/data/{did}/fileData.do")
        pk = re.search(r'id="publicDataDetailPk"[^>]*value="([^"]+)"', h).group(1)
        name = re.search(r'<strong class="key">파일데이터명</strong>\s*<div class="value">([^<]+)</div>', h).group(1).strip()
        jobs.append((did, name, pk, "", "1"))
    # 과거판: 팝업에서 첨부파일 ID를 읽는다
    for r in csv.DictReader(open(LIST, encoding="utf-8-sig")):
        pop = curl(BASE + "/tcs/dss/selectDpkDetailInfo.do", urllib.parse.urlencode(
            {"publicDataDetailPk": r["publicDataDetailPk"], "publicDataHistSn": r["publicDataHistSn"]}))
        m = re.search(r"fn_fileDataDown\('(\d+)', '([^']+)', '([^']*)', '(\d+)'", pop)
        if not m:
            jobs.append((r["dataset_id"], r["version_name"], r["publicDataDetailPk"], None, None))
            continue
        jobs.append((r["dataset_id"], r["version_name"], m.group(2), m.group(3), m.group(4)))
        time.sleep(0.3)
    log = []
    for did, label, pk, atch, sn in jobs:
        if atch is None:
            log.append([did, label, "", "", "다운로드 버튼 없음"]); print(did, label, "버튼 없음"); continue
        dest, st = download(did, pk, atch, sn, label)
        size = dest.stat().st_size if dest else 0
        log.append([did, label, dest.name if dest else "", size, st])
        print(did, label, "->", dest.name if dest else "-", size, st, flush=True)
        time.sleep(1)
    with open(OUT / "_download_log.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset_id", "version_name", "file", "bytes", "status"])
        w.writerows(log)


if __name__ == "__main__":
    main()
