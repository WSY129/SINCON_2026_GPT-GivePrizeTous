# External tables of v20e (queried 2026-11-13 on the transport-card statistics site https://www.stcis.go.kr, menu 노선·정류장 지표 > 정류장별 이용량)

Hourly **bus** boardings of single bus stops.  Only rows of kind `버스` were selected; the subway station that the stop search also lists was never queried.
- `stcis_bus_boardings_stop_37144.csv`  stop 문학경기장(야구장), ARS 37144: fifteen days of the training period (2026-05-30 .. 06-13) and six event days of the evaluation period, boardings at 16-23h
- `bus_to_subway_calibration.csv`  the eight event / game days of the training period in that file against the training ridership of station 3127: boardings leaving by subway = 8.3 - 12.9 x bus boardings (median 11.9)
- `stcis_bus_boardings_seonhak.csv`  stops 선학역 (ARS 38305, 38306): festival days 2026-03-28/29 (training period) and 2026-10-09/10
Also looked at, not used: stops at 가재울역 (volumes of a few dozen a day, no clear signal), 신검단중앙역사거리 and 아라역1번출구 (weekday totals June -> October +16 % and +53 %; new bus routes from 2026-09-19 serve these stops).
