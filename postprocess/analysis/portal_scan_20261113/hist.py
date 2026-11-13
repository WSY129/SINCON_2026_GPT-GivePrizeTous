"""data.go.kr 파일데이터의 설명·행수·주기성 과거 데이터 목록을 뽑는다. 사용: hist.py out.jsonl id1 id2 ..."""
import html, json, re, subprocess, sys, time


def get(url, data=None):
    cmd = ["curl", "-s", "-m", "40", "-A", "Mozilla/5.0", url]
    if data:
        cmd += ["-X", "POST", "--data", data]
    for i in range(3):
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0 and r.stdout:
            return r.stdout.decode("utf-8", "ignore")
        time.sleep(2)
    return ""


def c(s):
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s))).strip()


out = open(sys.argv[1], "a", encoding="utf-8")
for did in sys.argv[2:]:
    h = get(f"https://www.data.go.kr/data/{did}/fileData.do")
    meta = {}
    for k, v in re.findall(r'<strong class="key">(.*?)</strong>\s*<div class="value">(.*?)</div>', h, re.S):
        k = c(k)
        if k not in meta:
            meta[k] = c(v)[:700]
    pk = re.search(r'id="publicDataDetailPk"[^>]*value="([^"]+)"', h)
    hist = []
    if pk:
        hh = get("https://www.data.go.kr/tcs/dss/selectHistAndCsvData.do",
                 f"publicDataPk={did}&publicDataDetailPk={pk.group(1)}")
        for dpk, sn, name, dt in re.findall(
                r'data-public-pk="([^"]+)"\s*data-public-detail-sn="(\d+)">\s*(.*?)\s*</a>.*?(\d{4}-\d{2}-\d{2})', hh, re.S):
            hist.append({"name": c(name), "reg": dt, "pk": dpk, "sn": sn})
    rec = {"id": did, "name": meta.get("파일데이터명", ""), "desc": meta.get("설명", ""),
           "rows": meta.get("전체 행", ""), "cycle": meta.get("업데이트 주기", ""),
           "updated": meta.get("수정일", ""), "ext": meta.get("확장자", ""), "hist": hist}
    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.flush()
    print(did, rec["name"], "| rows", rec["rows"], "| hist", len(hist), [x["name"].split("_")[-1] for x in hist])
    time.sleep(0.3)
