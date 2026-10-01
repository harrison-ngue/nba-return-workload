# ============================================================
# MIT SLOAN NBA RETURN-TO-PLAY FIGURES
# Basic plotting / analysis template
#
# Produces:
#   Figure 1: What Return Workload is + how NBA teams vary
#   Figure 2: Return Workload vs subsequent injury + robustness
#
# Recommended exact inputs from the final longitudinal analysis:
#   episodes_master_analysis.pkl
#   post_raw.pkl
#   player_game_master_selected.pkl
#   final_authoritative_model_results.csv
#   final_bootstrap_1000_played3_any.csv
#
# Optional:
#   shape_metrics_expanded.csv
#   final_within_player.csv
#
# If you only have the original AI bundle:
#   combined_injury_episodes.parquet
#   combined_rtp_episode_game_windows.parquet
#   combined_player_game_master.parquet
#
# The descriptive Figure 1 can be rebuilt from the AI bundle directly.
# For Figure 2, the corrected longitudinal final-analysis files are
# preferred because they preserve cross-season reconstruction and the
# exact landmark outcome definition used in the manuscript.
# ============================================================

# ----------------------------
# 0. Imports and configuration
# ----------------------------
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ---- EDIT THIS ----
DATA_DIR = Path("/content/NBA_RTP_SLOAN_INPUTS")
OUT_DIR = Path("/content/NBA_RTP_SLOAN_FIGURES")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# If you want to manually choose 4 player episodes for Figure 1,
# paste their episode_ids here. Otherwise leave as None and the
# script will automatically choose contrasting ramp shapes.
EXAMPLE_EPISODE_IDS = None
# Example:
# EXAMPLE_EPISODE_IDS = [
#     "episode_id_for_smooth_progression",
#     "episode_id_for_immediate_return",
#     "episode_id_for_rest_interrupted",
#     "episode_id_for_irregular_return",
# ]

# Final manuscript estimates, used only for labels/forest plot.
# These are NOT used to manufacture the raw data plots.
FINAL_STATS = {
    "primary":      dict(label="Primary adjusted", OR=0.75, lo=0.64, hi=0.88),
    "fine_clinical":dict(label="Fine injury type + severity", OR=0.75, lo=0.64, hi=0.88),
    "within_player":dict(label="Within-player", OR=0.67, lo=0.52, hi=0.85),
    "different_body":dict(label="Different-body second injury", OR=0.74, lo=0.61, hi=0.89),
    "exact_recurrence":dict(label="Exact recurrence", OR=1.00, lo=0.73, hi=1.38),
    "ramp_deviation":dict(label="Ramp Deviation*", OR=1.09, lo=0.95, hi=1.24),
}
FINAL_SCHEDULE_OR = 1.16
FINAL_SCHEDULE_LO = 1.08
FINAL_SCHEDULE_HI = 1.26
FINAL_RISK_LOW = 25.0
FINAL_RISK_HIGH = 43.3
FINAL_RISK_DIFF = -18.3


# -----------------------------------
# 1. Small helpers / canonical loader
# -----------------------------------
def first_existing(df, candidates, required=True):
    """Return the first column in candidates that exists."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"None of these columns were found: {candidates}\n"
            f"Available columns include:\n{list(df.columns)[:100]}"
        )
    return None


def load_inputs(data_dir=DATA_DIR):
    """
    Preferred: corrected longitudinal final-analysis files.

    Fallback: original AI analysis bundle. The fallback is excellent
    for Figure 1, but the final-analysis files should be used for exact
    Figure 2 replication.
    """
    final_ep = data_dir / "episodes_master_analysis.pkl"
    final_post = data_dir / "post_raw.pkl"
    final_pg = data_dir / "player_game_master_selected.pkl"

    if final_ep.exists() and final_post.exists():
        print("Loading corrected longitudinal final-analysis files.")
        ep = pd.read_pickle(final_ep)
        post = pd.read_pickle(final_post)
        pg = pd.read_pickle(final_pg) if final_pg.exists() else None
        source = "final"

    else:
        print("Final-analysis pickle files not found.")
        print("Falling back to the original AI bundle for descriptive plotting.")
        ep = pd.read_parquet(data_dir / "combined_injury_episodes.parquet")
        post = pd.read_parquet(data_dir / "combined_rtp_episode_game_windows.parquet")
        pg_path = data_dir / "combined_player_game_master.parquet"
        pg = pd.read_parquet(pg_path) if pg_path.exists() else None
        source = "bundle"

    return ep, post, pg, source


ep_raw, post_raw, pg_raw, SOURCE = load_inputs()


# ------------------------------------------------
# 2. Convert source-specific columns to one schema
# ------------------------------------------------
def canonicalize_episode_table(ep, source):
    x = ep.copy()

    if source == "final":
        colmap = {
            "episode_id": first_existing(x, ["episode_id"]),
            "player_id": first_existing(x, ["player_id"]),
            "player_name": first_existing(x, ["player_name"], required=False),
            "baseline": first_existing(x, ["baseline_minutes", "pre_median_minutes"]),
            "games_missed": first_existing(x, ["games_missed"]),
            "season": first_existing(x, ["return_season", "season"], required=False),
            "starter": first_existing(x, ["normal_starter"], required=False),
            "injury_region": first_existing(x, ["initial_region", "injury_region"], required=False),
            "injury_body": first_existing(x, ["initial_body_part", "injury_body_part"], required=False),
            "injury_dx": first_existing(x, ["initial_diagnosis_class", "injury_diagnosis_class"], required=False),
        }

        out = pd.DataFrame(index=x.index)
        for new, old in colmap.items():
            out[new] = x[old] if old is not None else np.nan

        # Exact primary landmark cohort from final analysis
        if {"primary_eligible", "played3_complete", "complete30_played3"}.issubset(x.columns):
            out["primary_cohort"] = (
                (x["primary_eligible"] == 1) &
                (x["played3_complete"] == 1) &
                (x["complete30_played3"] == 1)
            )
        else:
            out["primary_cohort"] = True

        out["outcome30"] = (
            x["event30_played3"]
            if "event30_played3" in x.columns
            else np.nan
        )

        if "played3_deficit" in x.columns:
            out["played3_deficit"] = x["played3_deficit"]
        if "rri5" in x.columns:
            out["return_workload_5_stored"] = x["rri5"]

    else:
        colmap = {
            "episode_id": first_existing(x, ["episode_id"]),
            "player_id": first_existing(x, ["player_id"]),
            "player_name": first_existing(x, ["player_name"], required=False),
            "baseline": first_existing(x, ["pre_median_minutes"]),
            "games_missed": first_existing(x, ["games_missed"]),
            "season": first_existing(x, ["season"], required=False),
            "starter": first_existing(x, ["normal_starter", "starter"], required=False),
            "injury_region": first_existing(x, ["injury_region"], required=False),
            "injury_body": first_existing(x, ["injury_body_part"], required=False),
            "injury_dx": first_existing(x, ["injury_diagnosis_class"], required=False),
        }

        out = pd.DataFrame(index=x.index)
        for new, old in colmap.items():
            out[new] = x[old] if old is not None else np.nan

        eligible_col = first_existing(
            x,
            ["primary_analysis_eligible", "primary_eligible"],
            required=False
        )
        out["primary_cohort"] = (
            x[eligible_col].astype(bool) if eligible_col is not None else True
        )

        outcome_col = first_existing(
            x,
            ["subsequent_any_injury_absence_within_30d",
             "any_injury_absence_within_30d"],
            required=False
        )
        out["outcome30"] = x[outcome_col] if outcome_col else np.nan

    out["baseline"] = pd.to_numeric(out["baseline"], errors="coerce")
    out["games_missed"] = pd.to_numeric(out["games_missed"], errors="coerce")
    out["role_group"] = pd.cut(
        out["baseline"],
        bins=[10, 20, 30, np.inf],
        right=False,
        labels=["10–<20 MPG", "20–<30 MPG", "≥30 MPG"]
    )
    out["severity_group"] = pd.cut(
        out["games_missed"],
        bins=[1, 3, 7, 15, np.inf],
        labels=["2–3", "4–7", "8–15", "≥16"]
    )
    return out


def canonicalize_window_table(post, source):
    w = post.copy()

    if source == "final":
        out = pd.DataFrame()
        out["episode_id"] = w[first_existing(w, ["episode_id"])]
        out["scheduled_index"] = pd.to_numeric(
            w[first_existing(w, ["game_index", "relative_game"])],
            errors="coerce"
        )
        out["minutes"] = pd.to_numeric(
            w[first_existing(w, ["minutes"])], errors="coerce"
        ).fillna(0)

        played_col = first_existing(w, ["played", "played_actual"], required=False)
        out["played"] = (
            w[played_col].astype(bool)
            if played_col is not None
            else out["minutes"] > 0
        )

    else:
        out = pd.DataFrame()
        out["episode_id"] = w[first_existing(w, ["episode_id"])]
        out["scheduled_index"] = pd.to_numeric(
            w[first_existing(w, ["relative_game"])], errors="coerce"
        )
        out["minutes"] = pd.to_numeric(
            w[first_existing(w, ["minutes"])], errors="coerce"
        ).fillna(0)
        played_col = first_existing(w, ["played_actual"], required=False)
        out["played"] = (
            w[played_col].astype(bool)
            if played_col is not None
            else out["minutes"] > 0
        )

        # Keep only return game and later scheduled team games.
        out = out[out["scheduled_index"] >= 0].copy()

    return out


episodes = canonicalize_episode_table(ep_raw, SOURCE)
windows = canonicalize_window_table(post_raw, SOURCE)


# -----------------------------------------------------
# 3. Compute schedule-aware Return Workload from games
# -----------------------------------------------------
def build_return_workload(episodes, windows, n_games=3):
    """
    Return Workload = 100 * mean(min(minutes / baseline, 1))
    across first N SCHEDULED team games after return.

    A scheduled non-played game has minutes = 0, so it contributes zero.
    """
    base = episodes.set_index("episode_id")["baseline"]

    z = windows.copy()
    z = z[z["scheduled_index"].between(0, n_games - 1)]
    z["baseline"] = z["episode_id"].map(base)
    z["relative_workload"] = np.where(
        z["baseline"] > 0,
        np.minimum(z["minutes"] / z["baseline"], 1.0),
        np.nan
    )

    # Need all N scheduled games observed.
    summary = z.groupby("episode_id").agg(
        observed_games=("scheduled_index", "nunique"),
        return_workload=("relative_workload", "mean")
    )
    summary = summary[summary["observed_games"] == n_games]
    summary[f"return_workload_{n_games}"] = 100 * summary["return_workload"]
    return summary[[f"return_workload_{n_games}"]]


rw3 = build_return_workload(episodes, windows, 3)
rw5 = build_return_workload(episodes, windows, 5)

episodes = episodes.merge(rw3, left_on="episode_id", right_index=True, how="left")
episodes = episodes.merge(rw5, left_on="episode_id", right_index=True, how="left")

primary = episodes[
    episodes["primary_cohort"] &
    episodes["baseline"].ge(10)
].copy()

print(f"Primary descriptive cohort: {len(primary):,} episodes")
print(f"Players: {primary['player_id'].nunique():,}")
print(primary[["return_workload_3", "return_workload_5"]].describe())


# ---------------------------------
# 4. Ramp-shape helper calculations
# ---------------------------------
def episode_schedule_profile(episode_id, n_games=5):
    row = episodes.loc[episodes["episode_id"] == episode_id].iloc[0]
    z = windows[
        (windows["episode_id"] == episode_id) &
        (windows["scheduled_index"].between(0, n_games - 1))
    ].sort_values("scheduled_index").copy()

    # Reindex so rest/missing scheduled games remain explicit if indices exist.
    z = z.set_index("scheduled_index").reindex(range(n_games))
    z["minutes"] = z["minutes"].fillna(0)
    z["ratio"] = np.minimum(z["minutes"] / row["baseline"], 1.0)

    return row, z.reset_index()


def shape_metrics_from_ratios(r):
    r = np.asarray(r, dtype=float)
    if len(r) < 3:
        return dict(ramp_deviation=np.nan, backtracking=np.nan,
                    monotonic_fraction=np.nan)

    line = np.linspace(r[0], r[-1], len(r))
    ramp_deviation = np.sqrt(np.mean((r - line) ** 2))
    d = np.diff(r)
    backtracking = np.maximum(-d, 0).sum()
    monotonic_fraction = np.mean(d >= 0)

    return dict(
        ramp_deviation=ramp_deviation,
        backtracking=backtracking,
        monotonic_fraction=monotonic_fraction
    )


def make_shape_table(n_games=5):
    rows = []
    for eid in primary["episode_id"].dropna():
        try:
            row, z = episode_schedule_profile(eid, n_games)
        except Exception:
            continue
        if len(z) != n_games:
            continue
        m = shape_metrics_from_ratios(z["ratio"])
        rows.append({
            "episode_id": eid,
            "player_name": row.get("player_name", ""),
            "baseline": row["baseline"],
            "games_missed": row["games_missed"],
            "return_workload": 100 * z["ratio"].mean(),
            "has_zero": (z["minutes"] <= 0).any(),
            **m
        })
    return pd.DataFrame(rows)


shape5 = make_shape_table(5)


# ------------------------------------------------
# 5. Automatically pick 4 contrasting trajectories
# ------------------------------------------------
def choose_contrasting_examples(shape_df):
    s = shape_df.dropna(subset=["ramp_deviation"]).copy()

    # Smooth progression: some restriction early, high finish,
    # low deviation, and not a totally flat sequence.
    smooth_pool = s[
        (s["return_workload"].between(45, 85)) &
        (~s["has_zero"])
    ]
    smooth = (
        smooth_pool.sort_values(["ramp_deviation", "backtracking"])
        .iloc[0]["episode_id"]
        if len(smooth_pool) else s.sort_values("ramp_deviation").iloc[0]["episode_id"]
    )

    # Immediate/full return.
    immediate_pool = s[s["return_workload"] >= 95]
    immediate = (
        immediate_pool.sort_values("ramp_deviation").iloc[0]["episode_id"]
        if len(immediate_pool) else s.sort_values("return_workload", ascending=False).iloc[0]["episode_id"]
    )

    # Interrupted by zero-minute scheduled game.
    rest_pool = s[s["has_zero"]]
    rest = (
        rest_pool.sort_values("return_workload").iloc[len(rest_pool)//2]["episode_id"]
        if len(rest_pool) else s.sort_values("return_workload").iloc[0]["episode_id"]
    )

    # Most irregular/backtracking.
    irregular = s.sort_values(
        ["ramp_deviation", "backtracking"],
        ascending=False
    ).iloc[0]["episode_id"]

    # Remove duplicates while preserving order; refill if needed.
    ids = []
    for eid in [smooth, immediate, rest, irregular]:
        if eid not in ids:
            ids.append(eid)

    for eid in s.sort_values("ramp_deviation", ascending=False)["episode_id"]:
        if len(ids) == 4:
            break
        if eid not in ids:
            ids.append(eid)

    return ids[:4]


if EXAMPLE_EPISODE_IDS is None:
    EXAMPLE_EPISODE_IDS = choose_contrasting_examples(shape5)

print("\nSelected example episode IDs:")
print(EXAMPLE_EPISODE_IDS)


# ----------------------------------------------------
# 6. Shared low-level plotting functions
# ----------------------------------------------------
def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(ax, letter):
    ax.text(
        -0.10, 1.08, letter,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top"
    )


def jitter(n, center, width=0.08):
    return rng.normal(center, width, n)


def plot_episode_ramp(ax, episode_id, n_games=8, title=None):
    row, z = episode_schedule_profile(episode_id, n_games)

    x = np.arange(1, n_games + 1)
    y = z["minutes"].to_numpy(float)
    baseline = float(row["baseline"])

    ax.plot(x, y, marker="o", linewidth=1.8)
    ax.axhline(baseline, linestyle="--", linewidth=1.0)
    ax.set_xlim(0.7, n_games + 0.3)
    ax.set_xticks(x)
    ax.set_xlabel("Scheduled game after return")
    ax.set_ylabel("Minutes")
    clean_axis(ax)

    # Right y-axis = the same raw minute scale expressed as % baseline.
    ax2 = ax.twinx()
    ymin, ymax = ax.get_ylim()
    ax2.set_ylim(100 * ymin / baseline, 100 * ymax / baseline)
    ax2.set_ylabel("% usual minutes")
    ax2.spines["top"].set_visible(False)

    # Mark zero-minute games as rests/sits.
    for i, val in enumerate(y, start=1):
        if val <= 0:
            ax.text(i, 1, "0", ha="center", va="bottom", fontsize=8)

    if title is None:
        player = row.get("player_name", "")
        injury = " ".join([
            str(row.get("injury_body", "")),
            str(row.get("injury_dx", ""))
        ]).strip()
        rw = 100 * np.mean(np.minimum(y / baseline, 1))
        title = f"{player}\n{injury} | Return Workload {rw:.0f}%"

    ax.set_title(title, fontsize=10)


def strip_plot(ax, df, group_col, value_col, order, xlabel, ylabel):
    for i, group in enumerate(order):
        vals = df.loc[df[group_col] == group, value_col].dropna()
        if not len(vals):
            continue
        ax.scatter(
            jitter(len(vals), i),
            vals,
            s=10,
            alpha=0.28
        )
        ax.scatter(
            [i], [vals.median()],
            marker="_",
            s=180,
            linewidth=2.2
        )

    ax.set_xticks(range(len(order)), order)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    clean_axis(ax)


# ============================================================
# FIGURE 1
# NBA teams vary widely in how they restore normal workload
# ============================================================
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(
    3, 4,
    figure=fig,
    height_ratios=[1.05, 0.85, 1.0],
    hspace=0.55,
    wspace=0.45
)

# A-D: four real player return trajectories
for j, eid in enumerate(EXAMPLE_EPISODE_IDS):
    ax = fig.add_subplot(gs[0, j])
    plot_episode_ramp(ax, eid, n_games=8)
    panel_label(ax, chr(ord("A") + j))

# E: graphical definition of Return Workload
ax = fig.add_subplot(gs[1, :2])
ax.axis("off")
panel_label(ax, "E")

# Use one illustrative row from the actual data.
example = primary.dropna(subset=["return_workload_3"]).iloc[0]
row, z = episode_schedule_profile(example["episode_id"], 3)
b = row["baseline"]
mins = z["minutes"].to_numpy(float)
pcts = 100 * np.minimum(mins / b, 1)
rw = pcts.mean()

formula_text = (
    "RETURN WORKLOAD\n\n"
    f"Usual workload = {b:.1f} min/game\n\n"
    f"Game 1: {mins[0]:.0f} min  →  {pcts[0]:.0f}%\n"
    f"Game 2: {mins[1]:.0f} min  →  {pcts[1]:.0f}%\n"
    f"Game 3: {mins[2]:.0f} min  →  {pcts[2]:.0f}%\n\n"
    f"3-game Return Workload = {rw:.0f}%\n\n"
    "Scheduled non-played game = 0% competitive workload\n"
    "Each game is capped at 100% of usual workload"
)
ax.text(0.02, 0.98, formula_text, va="top", fontsize=12)
ax.set_title(
    "A simple player-normalized measure of early competitive workload",
    fontsize=12
)

# F: histogram
ax = fig.add_subplot(gs[1, 2:])
vals = primary["return_workload_5"].dropna()
ax.hist(vals, bins=np.arange(0, 105, 5))
ax.axvline(vals.mean(), linestyle="--", linewidth=1.2)
ax.set_xlabel("5-game Return Workload (%)")
ax.set_ylabel("Return episodes")
ax.set_title(
    f"Return strategies span a wide range\n"
    f"Mean {vals.mean():.1f}% | SD {vals.std(ddof=1):.1f} points"
)
clean_axis(ax)
panel_label(ax, "F")

# G: severity strip
ax = fig.add_subplot(gs[2, :2])
severity_order = ["2–3", "4–7", "8–15", "≥16"]
strip_plot(
    ax,
    primary,
    "severity_group",
    "return_workload_5",
    severity_order,
    "Games missed before return",
    "5-game Return Workload (%)"
)
ax.set_title(
    "Teams generally restore less workload after longer absences,\n"
    "but return strategies still overlap substantially"
)
panel_label(ax, "G")

# H: role strip
ax = fig.add_subplot(gs[2, 2:])
role_order = ["10–<20 MPG", "20–<30 MPG", "≥30 MPG"]
strip_plot(
    ax,
    primary,
    "role_group",
    "return_workload_5",
    role_order,
    "Usual preinjury role",
    "5-game Return Workload (%)"
)
ax.set_title(
    "Player role also shapes return decisions,\n"
    "but similar-role players still receive different workloads"
)
panel_label(ax, "H")

fig.suptitle(
    "Figure 1. NBA teams vary widely in how quickly they restore a player's normal workload after injury",
    fontsize=17,
    y=0.995,
    fontweight="bold"
)
fig.savefig(
    OUT_DIR / "Figure_1_Return_Workload_Variation.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 7. Optional adjusted dose-response for Figure 2 panel A
# ============================================================
def fit_adjusted_return_workload_model(primary):
    """
    Basic adjusted GLM for plotting a standardized risk curve.
    The exact manuscript model may contain additional covariates.

    If using the final longitudinal episode table, this curve should
    be very close to the final analysis. The forest-panel annotations
    below use the frozen final manuscript estimates.
    """
    d = primary.dropna(
        subset=["outcome30", "return_workload_3", "baseline", "games_missed"]
    ).copy()

    d["outcome30"] = pd.to_numeric(d["outcome30"], errors="coerce")
    d = d[d["outcome30"].isin([0, 1])]
    d["rw10"] = d["return_workload_3"] / 10
    d["baseline5"] = d["baseline"] / 5
    d["log_games_missed"] = np.log1p(d["games_missed"])

    # Only use covariates actually available in the canonical table.
    formula = "outcome30 ~ rw10 + baseline5 + log_games_missed"

    if d["injury_region"].notna().sum() > 0:
        formula += " + C(injury_region)"
    if d["injury_dx"].notna().sum() > 0:
        formula += " + C(injury_dx)"
    if d["season"].notna().sum() > 0:
        formula += " + C(season)"

    model = smf.glm(
        formula,
        data=d,
        family=sm.families.Binomial()
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": pd.factorize(d["player_id"])[0]}
    )
    return model, d


def standardized_risk_curve(model, d, grid=np.arange(30, 101, 2)):
    """
    G-computation curve: set Return Workload to each grid value for
    every episode, predict, then average.
    """
    rows = []
    for g in grid:
        new = d.copy()
        new["rw10"] = g / 10
        pred = model.predict(new)
        rows.append({"return_workload": g, "risk": pred.mean()})
    return pd.DataFrame(rows)


dose_model, dose_data = fit_adjusted_return_workload_model(primary)
curve = standardized_risk_curve(dose_model, dose_data)


# ============================================================
# 8. Bootstrap data
# ============================================================
bootstrap_path = DATA_DIR / "final_bootstrap_1000_played3_any.csv"

if bootstrap_path.exists():
    boot = pd.read_csv(bootstrap_path)
    or_candidates = [c for c in boot.columns if "OR" in c.upper()]
    if not or_candidates:
        raise ValueError("Bootstrap file found but no OR column detected.")
    boot_or = pd.to_numeric(boot[or_candidates[0]], errors="coerce").dropna()
else:
    # We deliberately DO NOT fabricate a bootstrap distribution.
    # The panel will show an instruction until the final bootstrap CSV
    # is added to DATA_DIR.
    boot_or = pd.Series(dtype=float)
    print(
        "\nNOTE: final_bootstrap_1000_played3_any.csv not found. "
        "Figure 2D will show a placeholder. "
        "Add the final bootstrap CSV for exact replication."
    )


# ============================================================
# FIGURE 2
# Lower Return Workload is associated with fewer later absences
# ============================================================
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(
    2, 2,
    figure=fig,
    hspace=0.40,
    wspace=0.36
)

# A: adjusted dose-response
ax = fig.add_subplot(gs[0, 0])
ax.plot(curve["return_workload"], 100 * curve["risk"], linewidth=2.2)
ax.set_xlabel("3-game Return Workload (%)")
ax.set_ylabel("Standardized 30-day injury-related absence risk (%)")
ax.set_title(
    "Higher early Return Workload is associated with higher subsequent risk"
)
ax.text(
    0.03, 0.95,
    f"+10 workload points: OR {FINAL_SCHEDULE_OR:.2f} "
    f"({FINAL_SCHEDULE_LO:.2f}–{FINAL_SCHEDULE_HI:.2f})",
    transform=ax.transAxes,
    va="top",
    fontsize=11
)
clean_axis(ax)
panel_label(ax, "A")

# B: illustrative absolute-risk contrast
ax = fig.add_subplot(gs[0, 1])
risk_df = pd.DataFrame({
    "group": ["<80% early workload", "≥80% early workload"],
    "risk": [FINAL_RISK_LOW, FINAL_RISK_HIGH]
})
ax.bar(risk_df["group"], risk_df["risk"])
ax.set_ylabel("Standardized 30-day risk (%)")
ax.set_title(
    "Illustrative absolute-risk contrast\n"
    "80% is not a recommended threshold"
)
for i, v in enumerate(risk_df["risk"]):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
ax.text(
    0.5, 0.04,
    f"Risk difference = {FINAL_RISK_DIFF:.1f} percentage points",
    transform=ax.transAxes,
    ha="center"
)
clean_axis(ax)
panel_label(ax, "B")

# C: compact hypothesis-testing forest
ax = fig.add_subplot(gs[1, 0])
forest_rows = [
    FINAL_STATS["primary"],
    FINAL_STATS["fine_clinical"],
    FINAL_STATS["within_player"],
    FINAL_STATS["different_body"],
    FINAL_STATS["exact_recurrence"],
    FINAL_STATS["ramp_deviation"],
]
f = pd.DataFrame(forest_rows)
y = np.arange(len(f))[::-1]

ax.errorbar(
    f["OR"],
    y,
    xerr=[f["OR"] - f["lo"], f["hi"] - f["OR"]],
    fmt="o",
    capsize=3
)
ax.axvline(1, linestyle="--", linewidth=1)
ax.set_yticks(y, f["label"])
ax.set_xlabel("Odds ratio (95% CI)")
ax.set_title(
    "The association persists under key alternative explanations"
)
clean_axis(ax)
panel_label(ax, "C")

# Footnote because shape estimate uses a different scale.
ax.text(
    0.0, -0.22,
    "*Ramp Deviation is scaled per +10 percentage points and "
    "adjusted for total early workload.",
    transform=ax.transAxes,
    fontsize=8
)

# D: 1,000-player bootstrap
ax = fig.add_subplot(gs[1, 1])
if len(boot_or):
    ax.hist(boot_or, bins=35)
    ax.axvline(1, linestyle="--", linewidth=1)
    ax.axvline(boot_or.median(), linestyle=":", linewidth=1.5)
    ax.set_xlabel("Player-cluster bootstrap OR")
    ax.set_ylabel("Bootstrap replicates")
    pct = 100 * np.mean(boot_or < 1)
    ax.set_title(
        f"Player-level resampling reproduces the association\n"
        f"{pct:.1f}% of bootstrap estimates < 1"
    )
else:
    ax.axis("off")
    ax.text(
        0.5, 0.5,
        "Add final_bootstrap_1000_played3_any.csv\n"
        "to DATA_DIR to populate this panel.",
        ha="center", va="center", fontsize=12
    )
    ax.set_title("Player-level bootstrap")
panel_label(ax, "D")

fig.suptitle(
    "Figure 2. Lower early Return Workload is associated with fewer subsequent injury-related absences",
    fontsize=17,
    y=0.995,
    fontweight="bold"
)
fig.savefig(
    OUT_DIR / "Figure_2_Return_Workload_and_Injury.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("\nSaved:")
print(OUT_DIR / "Figure_1_Return_Workload_Variation.png")
print(OUT_DIR / "Figure_2_Return_Workload_and_Injury.png")
