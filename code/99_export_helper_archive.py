
"""
NBA RTP FINAL EXPORT HELPER v1.0

Run AFTER:
1) the four primary seasons are PASS,
2) NBA_RTP_AGE_RETROFIT_v2_8 has been loaded, and
3) retrofit_age_existing_outputs_v28(...) has completed successfully.

Creates:
A) NBA_RTP_PERSONAL_FULL_ARCHIVE.zip
   - entire nba_injury_rtp_multiseason project tree (including raw/cache files)
   - cleaned final notebook if available
   - archive README + file manifest

B) NBA_RTP_AI_ANALYSIS_BUNDLE.zip
   - analysis-ready combined Parquets
   - master Excel workbook
   - QC files
   - age audit files
   - cleaned final notebook if available
   - AI README + automatically generated schema inventory

No extraction is performed.
"""

from pathlib import Path
import json
import os
import shutil
import zipfile
import pandas as pd


PRIMARY_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]

# Put the cleaned notebook downloaded from ChatGPT in /content with this name
# if you want it embedded inside both ZIPs.
CLEAN_NOTEBOOK_NAME = "NBA_Injury_RTP_MultiSeason_FINAL_2021_25_v2_9.ipynb"

PERSONAL_ZIP_NAME = "NBA_RTP_PERSONAL_FULL_ARCHIVE.zip"
AI_ZIP_NAME = "NBA_RTP_AI_ANALYSIS_BUNDLE.zip"


def _find_clean_notebook():
    candidates = [
        Path(CLEAN_NOTEBOOK_NAME),
        Path("/content") / CLEAN_NOTEBOOK_NAME,
        Path("/mnt/data") / CLEAN_NOTEBOOK_NAME,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _zip_tree(zip_path, root_dir, arc_prefix=None):
    root_dir = Path(root_dir)
    prefix = arc_prefix or root_dir.name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for p in sorted(root_dir.rglob("*")):
            if p.is_file():
                arc = Path(prefix) / p.relative_to(root_dir)
                z.write(p, arcname=str(arc))


def _write_manifest(root, out_path):
    rows = []
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rows.append({
                "relative_path": str(p.relative_to(root)),
                "bytes": p.stat().st_size,
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return len(rows)


def _make_schema_inventory(ai_dir):
    records = []

    for p in sorted(ai_dir.iterdir()):
        try:
            if p.suffix.lower() == ".parquet":
                df = pd.read_parquet(p)
            elif p.suffix.lower() == ".csv" and p.name != "DATA_SCHEMA.csv":
                df = pd.read_csv(p, low_memory=False)
            else:
                continue
        except Exception:
            continue

        for c in df.columns:
            s = df[c]
            records.append({
                "file": p.name,
                "column": c,
                "dtype": str(s.dtype),
                "nonmissing_n": int(s.notna().sum()),
                "rows_n": int(len(df)),
                "nonmissing_fraction": float(s.notna().mean()) if len(df) else None,
            })

    out = pd.DataFrame(records)
    out.to_csv(ai_dir / "DATA_SCHEMA.csv", index=False)
    return out


def _ai_readme():
    return """# NBA Injury / Return-to-Play AI Analysis Bundle

## Primary cohort
NBA seasons 2021-22, 2022-23, 2023-24, and 2024-25.

Only seasons that passed the pipeline QC gates are intended for the primary
analysis. Review `combined_qc_summary.csv` and `combined_qc_gates.csv` before
modeling.

## Main files

### combined_injury_episodes.parquet
Primary episode-level modeling table. One row per injury/RTP episode.
Use `primary_analysis_eligible == True` for the main cohort unless the analysis
plan specifies otherwise.

Important exposure families include:
- return_minutes_ratio
- post3_mean_minutes_ratio
- post5_mean_minutes_ratio
- ramp_slope_minutes_per_game
- max_positive_minute_step_first5
- ramp_deficit_auc_first5
- games_to_90pct_baseline
- return_team_is_b2b
- return_team_calendar_rest_days
- return_travel_km
- b2b_count_first5_team_games_post
- travel_km_first5_team_games_post
- season_age

Important outcome families include:
- subsequent_any_injury_absence_within_14d / 30d / 60d
- subsequent_same_region_absence_within_*
- subsequent_same_body_part_absence_within_*
- subsequent_same_body_laterality_dx_absence_within_*
- early_same_body_continuation_proxy_30d_lt3_played
- separated_same_body_event_proxy_30d_ge3_played
- separated_same_body_laterality_dx_proxy_60d_ge3_played

Performance fields include pre/post true shooting, points, plus/minus, and some
advanced measures. Check DATA_SCHEMA.csv for coverage before using a field.

### combined_rtp_episode_game_windows.parquet
Game-level rows around each RTP episode. Use this to reconstruct alternative
pre/post windows, minute-ramp definitions, performance trajectories, schedule
burden, or time-varying exposures.

### combined_player_game_master.parquet
Full player-game master. This is the most flexible source for constructing new
game-level variables without re-extracting raw NBA/ESPN data.

### combined_all_absence_episodes.parquet
All classified absence episodes, useful for alternative recurrence/continuation
definitions and sensitivity analyses.

### NBA_Injury_RTP_MultiSeason_QC_and_Episodes.xlsx
Human-readable workbook containing QC, injury episodes, absence episodes, and
RTP windows.

### combined_qc_summary.csv / combined_qc_gates.csv
Season-level quality-control evidence.

### age_enrichment_audit.csv / age_enrichment_unmatched.csv / age_manual_resolution_audit.csv
Audit trail for the season_age retrofit and authoritative fallback resolution.

### NBA_Injury_RTP_MultiSeason_FINAL_2021_25_v2_9.ipynb
Cleaned reproducible extraction/processing notebook, if included.

## Statistical cautions
- Injury episodes are clustered within players; do not assume episode-level
  independence. Use player-clustered SEs, mixed effects, or an appropriate
  recurrent-event framework.
- Confounding by indication is substantial: injury severity/type, games missed,
  surgery/recovery wording, age, baseline role/minutes, team, season, and
  schedule may affect both RTP ramp and subsequent injury.
- Do not equate every subsequent injury-report absence with a true reinjury.
  The separated-event proxies are intended to help distinguish continuation
  from a later event.
- Handle end-of-season censoring explicitly for 14/30/60-day outcomes.
- Exact rest-hour fields are incomplete historically; calendar-rest/B2B fields
  are more consistently available.
"""


def _personal_readme():
    return """# NBA RTP Personal Full Archive

This archive is the preservation copy intended to prevent future re-extraction.

It contains the ENTIRE `nba_injury_rtp_multiseason` project directory available
in the Colab runtime at export time, including season-level processed outputs,
raw/cache material, injury-report downloads, diagnostics, combined datasets, and
QC artifacts.

Keep this ZIP in at least two durable locations.

For routine statistical analysis, use the smaller AI analysis bundle rather
than unpacking the full raw archive.

Important:
- Primary validated cohort: 2021-22 through 2024-25.
- Experimental/failed extension season folders may also be present if they were
  created in the same runtime. Their presence does NOT mean they passed QC.
- `season_manifest.json` and QC files determine whether a season is validated.
"""


def create_final_archives(
    project_root=None,
    download=True,
):
    project_root = Path(project_root or MULTI_ROOT)
    if not project_root.exists():
        raise FileNotFoundError(f"Project root not found: {project_root}")

    # Verify the core analysis files exist before archiving.
    required = [
        "combined_player_game_master.parquet",
        "combined_injury_episodes.parquet",
        "combined_all_absence_episodes.parquet",
        "combined_rtp_episode_game_windows.parquet",
        "combined_qc_summary.csv",
        "combined_qc_gates.csv",
        "NBA_Injury_RTP_MultiSeason_QC_and_Episodes.xlsx",
    ]
    missing = [f for f in required if not (project_root / f).exists()]
    if missing:
        raise RuntimeError(
            "Do not archive yet. Missing combined files:\n  - " + "\n  - ".join(missing)
        )

    clean_nb = _find_clean_notebook()

    # ------------------------------------------------------------------
    # PERSONAL archive staging
    # ------------------------------------------------------------------
    personal_stage = Path("NBA_RTP_PERSONAL_FULL_ARCHIVE")
    if personal_stage.exists():
        shutil.rmtree(personal_stage)
    personal_stage.mkdir()

    # Keep the project tree intact.
    shutil.copytree(project_root, personal_stage / project_root.name)

    (personal_stage / "README_PERSONAL_ARCHIVE.md").write_text(
        _personal_readme(), encoding="utf-8"
    )

    if clean_nb:
        shutil.copy2(clean_nb, personal_stage / clean_nb.name)

    n_manifest = _write_manifest(
        personal_stage,
        personal_stage / "FILE_MANIFEST.csv"
    )

    personal_zip = Path(PERSONAL_ZIP_NAME)
    if personal_zip.exists():
        personal_zip.unlink()
    _zip_tree(personal_zip, personal_stage, arc_prefix=personal_stage.name)

    # ------------------------------------------------------------------
    # AI bundle staging
    # ------------------------------------------------------------------
    ai_stage = Path("NBA_RTP_AI_ANALYSIS_BUNDLE")
    if ai_stage.exists():
        shutil.rmtree(ai_stage)
    ai_stage.mkdir()

    ai_files = required + [
        "age_enrichment_audit.csv",
        "age_enrichment_unmatched.csv",
        "age_manual_resolution_audit.csv",
    ]
    copied = []
    for name in ai_files:
        src = project_root / name
        if src.exists():
            shutil.copy2(src, ai_stage / name)
            copied.append(name)

    if clean_nb:
        shutil.copy2(clean_nb, ai_stage / clean_nb.name)
        copied.append(clean_nb.name)

    (ai_stage / "README_FOR_AI.md").write_text(
        _ai_readme(), encoding="utf-8"
    )
    _make_schema_inventory(ai_stage)

    ai_zip = Path(AI_ZIP_NAME)
    if ai_zip.exists():
        ai_zip.unlink()
    _zip_tree(ai_zip, ai_stage, arc_prefix=ai_stage.name)

    print("\n" + "=" * 78)
    print("FINAL ARCHIVES CREATED")
    print("=" * 78)
    print(f"Personal archive: {personal_zip.resolve()}")
    print(f"  files indexed:  {n_manifest:,}")
    print(f"  size:           {personal_zip.stat().st_size / (1024**2):,.1f} MB")
    print()
    print(f"AI bundle:        {ai_zip.resolve()}")
    print(f"  size:           {ai_zip.stat().st_size / (1024**2):,.1f} MB")
    print(f"  cleaned notebook included: {'YES' if clean_nb else 'NO'}")

    if clean_nb is None:
        print(
            "\nNOTE: The cleaned notebook was not found in /content. "
            "Upload NBA_Injury_RTP_MultiSeason_FINAL_2021_25_v2_9.ipynb to Colab "
            "and rerun this function if you want it embedded in both ZIPs."
        )

    if download:
        try:
            from google.colab import files
            print("\nStarting browser downloads...")
            files.download(str(personal_zip))
            files.download(str(ai_zip))
        except Exception as e:
            print("Automatic Colab download skipped:", e)

    return personal_zip, ai_zip


print("NBA RTP final export helper v1.0 loaded.")
print("Call create_final_archives() only AFTER the age retrofit has completed.")
