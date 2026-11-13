"""Parses a dump of the in-page query results (stcis.go.kr, 노선·정류장 지표 > 정류장별 이용량, bus stops only) into a long CSV.
Each query result row is: stop name, ARS number, the dates of the window, then 48 numbers per day = hours 04..03 (next day),
alternating boardings (승차) and alightings (하차).
    python outputs/handoff_for_gemini/scripts/parse_stcis_dump.py <dump.txt> <out.csv>"""
import sys, json, re
import pandas as pd

def load_dump(path):
    raw = open(path, encoding='utf-8').read()
    try:                                                        # tool-result file: [{type, text}] with the JS string inside text
        arr = json.loads(raw); text = ''.join(x.get('text', '') for x in arr)
    except json.JSONDecodeError:
        text = raw
    s, _ = json.JSONDecoder().raw_decode(text[text.index('"'):]) if not text.lstrip().startswith('{') else (text, 0)
    obj = json.loads(s) if isinstance(s, str) else s
    return obj.get('data', obj)

def parse(data):
    rows, bad = [], []
    for key, v in data.items():
        ars, f, t = key.split('|'); txt = v.get('txt', '')
        dates = re.findall(r'(\d{4}-\d{2}-\d{2})\((.)\)', txt)
        if not dates:
            bad.append(dict(ars=ars, query_from=f, query_to=t, result=txt[:60])); continue
        nums = txt[txt.rindex(')') + 1:].split()
        nums = [int(x.replace(',', '')) for x in nums]
        assert len(nums) == 48 * len(dates), (key, len(nums), len(dates))
        name = txt[:txt.index(' ' + ars + ' ')].strip()
        for i, (d, wd) in enumerate(dates):
            r = nums[i * 48:(i + 1) * 48]
            for j in range(24):
                h = 4 + j
                rows.append(dict(ars=ars, stop_name=name, date=d, weekday=wd, hour=h % 24, next_day=h >= 24, ride=r[2 * j], alight=r[2 * j + 1],
                                 query_from=f, query_to=t, queried_at_utc=v.get('at', ''), site_label=v.get('label', '')))
    return pd.DataFrame(rows), pd.DataFrame(bad)

if __name__ == '__main__':
    df, bad = parse(load_dump(sys.argv[1]))
    df = df.drop_duplicates(['ars', 'date', 'hour', 'next_day'])
    df.to_csv(sys.argv[2], index=False, encoding='utf-8-sig')
    print(len(df), 'rows', df.ars.nunique(), 'stops', df.date.nunique(), 'dates'); print('failed queries:', bad.to_dict('records'))
