# External tables of ensemble_v15 (copied 2026-11-13; data only, no code)

| file | from | used for |
|---|---|---|
| `event_corrections.csv` | written for v15 from the two team shares below | the only change of the model inputs against v14: 7 corrections of event labels in the training period, each confirmed by the training ridership |
| `events_venues.csv`, `events_venues_added.csv`, `transport_policy_events.csv` | team share v4 (`SINCON_데이터공유_v4.zip`) | reference for the corrections and for the entries of `configs/ensemble_v15.json`; not read by the code (putting all new labels into the model made it worse) |
| `DATA_REQUESTS_답변_v2.md` | teammate | answers to `outputs/analysis_20261113/DATA_REQUESTS.md` with sources |
| `events_2026_jul_oct_venues.csv` | team share `SINCON_Data_Event` | start / end times and attendance of the events of July-October 2026; reference for the v15e entries |
| `ssg_home_games_2025_2026h1.csv` | team share `SINCON_Data_Event` (`verification/1_ssg_games_2025_2026h1.csv`) | official start / end times and attendance of the home games before the cutoff |
| `exit_split_by_end_minute.csv` | computed here from the file above and the training ridership of station 3127 | share of the leaving crowd in the hour of the end / the next hour, by minute of the end |

Other tables (volleyball, arts centres, university festivals, school calendars, KMA weather, calendar, daily card data) are the copies in
`outputs/ensemble_v13/external` and `outputs/ensemble_v14/external`.

None of the files holds subway ridership of the evaluation period (2026-07-01 .. 2026-10-31).  Columns such as `peak_excess`,
`reference_peak_excess` and the `train:` notes are figures of the training period (to 2026-06-30) worked out by the teammates.
