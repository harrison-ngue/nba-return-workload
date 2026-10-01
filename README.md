# NBA Return-to-Play Workload After Injury

Open-source data and reproducibility materials for the study **“NBA Players With Lower Early Workload After Injury Had Fewer Subsequent Injury-Related Absences.”**

## Research question

After an NBA player returns from injury, does the amount of normal game workload restored during the first games back relate to subsequent injury-related absence?

The study reconstructs NBA injury and playing-time histories from the **2021-22 through 2024-25** seasons. The final primary longitudinal landmark analysis included **1,230 return episodes from 578 players** and evaluated unplanned injury-related absence during the following 30 days.

## What is in this repository?

### `data/`
Analysis-source tables used to build injury episodes, return-to-play windows, player-game histories, and alternative absence definitions. These are derived from publicly accessible NBA/game/injury-report sources. All included files are individually small enough for normal GitHub hosting.

The large Excel QC workbook from the original local analysis bundle is intentionally omitted because it duplicates the Parquet tables and exceeds GitHub's convenient browser-upload size. No analytic information needed from that workbook is unique: the corresponding Parquet/QC files are included.

### `code/`
- `00_extract_and_validate_multiseason_data.ipynb` — source extraction, harmonization, and QC notebook for 2021-22 through 2024-25.
- `10_comprehensive_analysis_archive.py` — broad exploratory analysis script from the project development archive.
- `20_sloan_figure1.py` — code used to construct the Sloan graphical summary.
- `21_sloan_figure_templates.py` — additional Sloan analysis/figure template.
- `99_export_helper_archive.py` — archive/export helper used during project development.

### `results/`
Frozen result summaries reported in the final manuscript/supplement. They are included as reference targets for reproducibility checks.

### `figures/` and `tables/`
Current Sloan abstract figure and table.

### `docs/`
Current manuscript and supplement, if included in this release.

## Return Workload

For each return, the player's usual preinjury playing time is used as the baseline. Schedule-aware **Return Workload** is the average percentage of that baseline accumulated across the first scheduled games after return, with a planned rest/recovery game contributing zero competitive minutes and each played game's contribution capped at 100% of baseline.

The final manuscript emphasizes the 3- and 5-game windows. See `docs/FINAL_LONGITUDINAL_ANALYSIS_SPEC.md` for the exact cohort and outcome definitions.

## Important reproducibility note

The files in `data/` are the validated four-season **source/derived data layer** that preceded the final longitudinal reconstruction. The final manuscript subsequently ordered player histories continuously across season boundaries so late-season injuries could link to returns in the next season. Therefore, a simple filter of `combined_injury_episodes.parquet` is **not** expected to equal the final 1,230-episode longitudinal landmark cohort. The exact final analysis definitions and cross-season reconstruction logic are documented in `docs/FINAL_LONGITUDINAL_ANALYSIS_SPEC.md`.

The competition requires the underlying data to be open. The complete analysis-source tables are included here; code is also provided to maximize transparency. `results/final_reported_key_results.csv` provides frozen targets from the final longitudinal analysis.

## Data files

- `combined_player_game_master.parquet` — full harmonized player-game table.
- `combined_injury_episodes.parquet` — validated season-level injury/RTP episode table generated before the final cross-season reconstruction.
- `combined_rtp_episode_game_windows.parquet` — game-level windows around injury/RTP episodes.
- `combined_all_absence_episodes.parquet` — all classified absence episodes.
- `DATA_SCHEMA.csv` — column-level schema and completeness inventory.
- `combined_qc_summary.csv`, `combined_qc_gates.csv` — season-level QC outputs.

## Reproduction environment

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

The historical extraction notebook makes network requests to public sources and may be affected by rate limits or source changes. Review the notebook's cache/QC logic before re-fetching data.

## Data provenance

See `DATA_SOURCES.md`. The repository contains derived research tables assembled from public sources; it does not claim ownership of the underlying NBA, ESPN, Stat Surge, or other third-party source data.

## Citation

If this work is accepted/published, replace the placeholder citation in `CITATION.md` with the final paper citation.

## Contact

Add the corresponding author's name and email here before making the repository public.
