# Final longitudinal analysis specification

This file documents the cohort and estimand used in the final manuscript so that the longitudinal reconstruction can be independently reimplemented from the source tables.

## 1. Continuous chronology

Order all available scheduled player-game rows continuously by player across the 2021-22, 2022-23, 2023-24, and 2024-25 seasons. Do not reset injury histories at season boundaries. Late-season injuries may therefore link to returns in the following season; preinjury baselines for early-season injuries may use legitimate played games from the preceding season.

## 2. Index injury episode

An index time-loss episode requires at least **2 consecutive scheduled games not played because of injury**. Illness, personal reasons, suspension, coaching decision, roster status, G League assignment, and other noninjury reasons are not index injuries.

## 3. Meaningful return

Meaningful return to play is the first subsequent appearance of at least **5 minutes**.

Primary eligibility additionally requires:

- at least **5 preceding played games**, and
- a player-specific preinjury baseline of at least **10 minutes/game**.

The preinjury baseline is the **median minutes across up to the 10 preceding played games**. Previous-season games may be used when chronologically appropriate.

## 4. Primary early-workload exposure

For each of the first 3 completed played games after meaningful return:

`workload ratio = minutes played / preinjury baseline minutes`

`deficit = max(0, 1 - workload ratio)`

Primary cumulative workload deficit is the sum of those 3 deficits. Larger values indicate more conservative restoration.

## 5. Primary landmark and follow-up

The primary analysis uses a **three-played-game landmark**. A player must complete the first 3 played games without an intervening unplanned injury-related absence. Outcome follow-up starts only after the third played game. The primary endpoint is the first qualifying **unplanned injury-related absence within the next 30 calendar days**, with adequate remaining observation required.

This estimand therefore applies to players who successfully reach the three-played-game landmark.

## 6. Schedule-aware Return Workload

For the first N scheduled team games after return:

`Return Workload_N = 100/N * sum(min(minutes_i / baseline, 1))`

A planned rest/recovery game contributes **0 minutes**. Each played game is capped at 100% of baseline so playing above baseline does not contribute more than one full game of restored workload.

Three- and five-scheduled-game windows are the main schedule-aware analyses.

## 7. Planned rest/recovery

Explicit management/recovery sits were identified from public report wording indicating management, recovery, reconditioning, return to competition, maintenance, rehabilitation, conditioning, or rest. Planned sits contribute zero competitive workload but are excluded from the unplanned injury outcome.

## 8. Outcome specificity

The first qualifying later injury-related absence is classified relative to the **initial index injury descriptor** as:

- exact same body part/laterality/diagnosis,
- same body part but different laterality or diagnosis,
- different body part within the same broad anatomical region,
- different broad anatomical region, or
- unclassified.

These are public-report proxies, not medical adjudication of tissue-level recurrence.

## 9. Core adjustment set

The expanded models adjust for age at return, baseline minutes, games missed, same-season time out where applicable, prior injury episodes, recent same-body injury history, initial injury region and diagnosis, season, return-game home/B2B status, travel, cross-season return, and baseline provenance. Player-clustered robust standard errors are used because players may contribute multiple episodes.

## 10. Sensitivity analyses

The final study additionally used fine body-part/diagnosis adjustment, within-player conditional logistic models, player-cluster bootstrap resampling, weighting analyses, survival analysis, leave-one-team/season-out analyses, falsification outcomes, prediction validation, ramp-shape analyses, and later performance analyses.

See the manuscript and supplement in `docs/` for the exact reported results and interpretation.
