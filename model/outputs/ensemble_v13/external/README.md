# External tables used by ensemble_v13 (copied 2026-11-13; data only, no code)

| file | from | content |
|---|---|---|
| events_venues.csv, events_venues_added.csv | team share v3 (`SINCON_데이터공유_v3.zip`) | concerts / festivals / sports events with start and end times; rows whose confidence starts with `data` (dates inferred from ridership spikes) are not used |
| games_kovo_incheon.csv | team share v3 | V-League home games (only the rows mapped to station 3759 are used) |
| artcenter_incheon_performances.csv, bac_bucheon_performances.csv | team share v3 | performances at the Incheon arts centre (station 3125) and the Bucheon arts centre (3756, 3757) |
| school_vacation_daily.csv | team share v3 (NEIS school calendars) | share of middle and high schools on vacation per day |
| incheon_univ_festival_labels.csv | team share `SINCON_data_weather_univ_bus` | university festival days by station |

None of these holds subway ridership. Ridership of the evaluation period (2026-07-01 .. 2026-10-31) is not used anywhere.
