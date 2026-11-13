"""Find abnormal days per station: daily boardings+alightings vs same-weekday median of surrounding +-4 weeks."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from common import load, calendar, ROOT

tr, _, meta = load()
tr["tot"] = tr.boardings + tr.alightings
d = tr.groupby(["station_id", "date"]).agg(tot=("tot", "sum"),
        night=("tot", lambda s: 0)).reset_index()
# night = 20-23h boardings (post-event exits)
nb = tr[tr.hour >= 20].groupby(["station_id", "date"]).boardings.sum().rename("night_b")
d = d.drop(columns="night").join(nb, on=["station_id", "date"])
d["dow"] = d.date.dt.day_of_week
cal = calendar()[["date", "is_holiday"]]
d = d.merge(cal, on="date")
out = []
for (s, w), g in d.groupby(["station_id", "dow"]):
    g = g.sort_values("date").set_index("date")
    for col in ["tot", "night_b"]:
        base = g[col].where(g.is_holiday == 0).rolling(9, center=True, min_periods=4).median()
        g[col + "_ratio"] = g[col] / base
    out.append(g.reset_index())
d = pd.concat(out)
d = d.merge(meta[["station_id", "station_name", "canonical_line"]], on="station_id")
d.to_csv(ROOT / "analysis" / "daily_ratio.csv", index=False)
sp = d[(d.tot_ratio >= 1.25) | (d.night_b_ratio >= 2.0)].sort_values(["station_id", "date"])
sp = sp[["date", "station_id", "station_name", "dow", "is_holiday", "tot", "tot_ratio", "night_b", "night_b_ratio"]]
sp.round(2).to_csv(ROOT / "analysis" / "spike_days.csv", index=False)
print("spike rows", len(sp))
print(sp.groupby(["station_id", "station_name"]).size().sort_values(ascending=False).head(20).to_string())
