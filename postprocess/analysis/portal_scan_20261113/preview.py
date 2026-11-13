"""data.go.kr 파일데이터 미리보기 표(포털이 보여 주는 앞부분)만 읽는다. 사용: preview.py id [id ...]"""
import html, re, subprocess, sys


def get(url, data=None):
    cmd = ["curl", "-s", "-m", "40", "-A", "Mozilla/5.0", url] + (["-X", "POST", "--data", data] if data else [])
    return subprocess.run(cmd, capture_output=True).stdout.decode("utf-8", "ignore")


def c(s):
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s))).strip()


for did in sys.argv[1:]:
    h = get(f"https://www.data.go.kr/data/{did}/fileData.do")
    pk = re.search(r'id="publicDataDetailPk"[^>]*value="([^"]+)"', h)
    if not pk:
        print("##", did, "no pk"); continue
    hh = get("https://www.data.go.kr/tcs/dss/selectHistAndCsvData.do", f"publicDataPk={did}&publicDataDetailPk={pk.group(1)}")
    tb = re.search(r"<table.*?</table>", hh, re.S)
    print("##", did)
    if not tb:
        print("   (no preview)"); continue
    head = [c(x) for x in re.findall(r"<th[^>]*>(.*?)</th>", tb.group(0), re.S)]
    print("   cols:", head[:30])
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tb.group(0), re.S)[1:9]:
        print("   ", [c(x)[:25] for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)][:14])
