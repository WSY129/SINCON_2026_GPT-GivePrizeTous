"""1. 기본 구조: 시간대·요일 모양, 역 유형, 월별 추세."""
import numpy as np
import pandas as pd
from _load import OUT, cal, meta, plt, train

t = train().merge(cal()[["date", "dow", "daytype", "is_holiday", "is_semi_holiday", "bridge"]], on="date")
m = meta()
t = t.merge(m[["station_id", "name", "canonical_line"]], on="station_id")
reg = t[(t.is_holiday == 0) & (t.is_semi_holiday == 0) & (t.bridge == 0)]

# 1-1 노선×요일유형 시간대 모양 (하루 합 대비 비율)
prof = reg.groupby(["canonical_line", "daytype", "date", "hour"]).boardings.sum().groupby(["canonical_line", "daytype", "hour"]).mean().unstack("hour")
share = prof.div(prof.sum(axis=1), axis=0).round(4)
share.to_csv(OUT / "a1_hour_profile_line_daytype.csv")
fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for i, (dt, lab) in enumerate([(0, "평일"), (1, "토요일"), (2, "일요일·공휴일")]):
    for ln in (1, 2, 7):
        ax[i].plot(share.columns, share.loc[(ln, dt)] * 100, marker="o", ms=3, label=f"{ln}호선")
    ax[i].set_title(lab); ax[i].set_xlabel("시"); ax[i].grid(alpha=.3)
ax[0].set_ylabel("하루 승차 중 비중(%)"); ax[0].legend()
fig.tight_layout(); fig.savefig(OUT / "a1_hour_profile.png", dpi=130); plt.close()

# 1-2 역 유형: 평일 아침(7~8시) 승차 비중, 저녁(18~19시) 승차 비중, 주말/평일 비
wd = reg[reg.daytype == 0].groupby(["station_id", "date", "hour"]).boardings.sum().groupby(["station_id", "hour"]).mean().unstack()
we = reg[reg.daytype == 2].groupby(["station_id", "date"]).boardings.sum().groupby("station_id").mean()
st = pd.DataFrame({"weekday_daily": wd.sum(axis=1)})
st["am_share"] = wd[[7, 8]].sum(axis=1) / st.weekday_daily
st["pm_share"] = wd[[18, 19]].sum(axis=1) / st.weekday_daily
st["night_share"] = wd[[21, 22, 23]].sum(axis=1) / st.weekday_daily
st["sun_to_weekday"] = we / st.weekday_daily
st["am_pm_ratio"] = st.am_share / st.pm_share


def kind(r):
    if r.sun_to_weekday >= 0.85:
        return "여가·관광형"
    if r.am_pm_ratio >= 1.6:
        return "주거형(아침 승차↑)"
    if r.am_pm_ratio <= 0.7:
        return "업무·산업형(저녁 승차↑)"
    return "혼합형"


st["type"] = st.apply(kind, axis=1)
st = st.join(m.set_index("station_id")[["name", "canonical_line"]]).round(3)
st.sort_values("am_pm_ratio").to_csv(OUT / "a1_station_types.csv")
print(st.type.value_counts().to_dict())
for k_, g in st.groupby("type"):
    print(k_, ", ".join(g.sort_values("weekday_daily", ascending=False).name.head(8)))

fig, ax = plt.subplots(figsize=(9, 7))
for k_, g in st.groupby("type"):
    ax.scatter(g.am_pm_ratio, g.sun_to_weekday, s=g.weekday_daily / 150, alpha=.6, label=k_)
for r in st.itertuples():
    if r.weekday_daily > 20000 or r.sun_to_weekday > 0.85 or r.am_pm_ratio < 0.6 or r.am_pm_ratio > 3:
        ax.annotate(r.name, (r.am_pm_ratio, r.sun_to_weekday), fontsize=7)
ax.set_xscale("log"); ax.set_xticks([0.2, 0.3, 0.5, 0.7, 1, 1.6, 2, 3, 5, 7]); ax.set_xticklabels(["0.2", "0.3", "0.5", "0.7", "1", "1.6", "2", "3", "5", "7"]); ax.minorticks_off();ax.set_xlabel("평일 아침(7~8시) ÷ 저녁(18~19시) 승차 비중 (로그)"); ax.set_ylabel("일요일 ÷ 평일 하루 승차")
ax.axhline(0.85, ls="--", c="gray", lw=.8); ax.axvline(1.6, ls=":", c="gray"); ax.axvline(0.7, ls=":", c="gray")
ax.legend(); ax.set_title("역 유형 (점 크기 = 평일 하루 승차)")
fig.tight_layout(); fig.savefig(OUT / "a1_station_types.png", dpi=130); plt.close()

# 1-3 월별 추세 (평일 하루 승차, 노선별) + 같은 달 전년 대비
mon = reg[reg.daytype == 0].groupby(["canonical_line", pd.Grouper(key="date", freq="MS"), "date"]).boardings.sum() \
    .groupby(["canonical_line", "date"]).mean().unstack(0)
mon.index = mon.index.to_period("M")
mon.round(0).to_csv(OUT / "a1_monthly_weekday_by_line.csv")
yoy = (mon.loc["2026-01":"2026-06"].values / mon.loc["2025-01":"2025-06"].values - 1)
print("2026 상반기 vs 2025 상반기 평일 승차 증감(노선 1,2,7):", np.round(yoy.mean(axis=0) * 100, 1))
fig, ax = plt.subplots(figsize=(10, 4))
for c in mon.columns:
    ax.plot(mon.index.to_timestamp(), mon[c] / 1000, marker="o", label=f"{c}호선")
ax.axvline(pd.Timestamp("2025-06-28"), c="red", ls="--", lw=.8); ax.text(pd.Timestamp("2025-07-01"), ax.get_ylim()[1] * .95, "검단연장·요금인상", fontsize=8, color="red")
ax.set_ylabel("평일 하루 승차 (천 명)"); ax.legend(); ax.grid(alpha=.3); ax.set_title("월별 평일 하루 승차")
fig.tight_layout(); fig.savefig(OUT / "a1_monthly.png", dpi=130); plt.close()

# 1-4 요일별 (평일 중 월~금 차이)
dowp = reg[reg.dow < 5].groupby(["dow", "date"]).boardings.sum().groupby("dow").mean()
print("요일별 평일 하루 승차 (월=1):", (dowp / dowp.iloc[0]).round(3).to_dict())
