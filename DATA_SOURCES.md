# Data sources and provenance

This repository packages the **derived analytic tables used by the project**, not a newly created official NBA database.

The multi-season extraction pipeline separates several source layers:

1. **Official NBA injury-report information.** Historical injury designations were based on publicly released NBA injury reports. For 2021-22 through 2023-24, the final extraction notebook used Vaughn Hajra / Stat Surge's public pre-parsed archive of NBA injury-report data. For 2024-25, the pipeline retained the previously validated public parsed season file used in the project and included code paths for validation against official NBA reports.
2. **Player-game and box-score data.** Historical 2021-22 through 2023-24 player-game coverage was reconstructed from ESPN-backed public data; the validated 2024-25 player-log path used the NBA/public mirror pipeline documented in the extraction notebook.
3. **Schedules and game context.** Team schedules, home/away identity, and game dates were harmonized from public NBA/ESPN-backed sources with QC against independent schedule coverage.
4. **NBA live box-score metadata.** Where available, public NBA live-data feeds were used for participation reason, exact game time, arena metadata, and related fields.
5. **Age/player metadata.** Public player-season/player profile information was harmonized and audited; age audit files are included in `data/`.

Key source endpoints/repositories referenced by the extraction notebook include:

- Official NBA injury reports: `https://ak-static.cms.nba.com/referee/injury/`
- NBA official site: `https://official.nba.com/`
- NBA live box scores: `https://cdn.nba.com/static/json/liveData/boxscore/`
- Public NBA data mirror used by the pipeline: `https://github.com/llimllib/nba_data`
- ESPN NBA scoreboard endpoint: `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard`
- 2024-25 public parsed injury-report source used in the pilot: `https://github.com/Professor-Pete/2024-2025-injury_analysis`
- Historical Stat Surge injury archive: public Google-Sheet CSV links documented in the extraction notebook.

## Redistribution note

All source information was publicly accessible when collected. This repository releases **derived research tables and analysis materials for reproducibility**. Underlying source content remains subject to the terms, trademarks, copyrights, and policies of its original providers. No affiliation with or endorsement by the NBA, ESPN, Stat Surge, or other providers is implied.
