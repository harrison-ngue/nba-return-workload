#!/usr/bin/env python3
"""
MIT Sloan Figure 1 — Nature-style two-row revision v5

Changes from prior draft:
A. Three players only (Luka Doncic, Jalen Suggs, Jamal Murray).
   - Preinjury phase is a single 100% baseline segment.
   - Injury absence is shown as a same-width compressed zero plateau for every player.
   - The actual number of games missed is written in the direct label.
   - Return ramps show up to 10 played games.
   - NBA median + IQR only in the return phase.
   - Luka's subsequent ankle injury-related absence is marked.

B. Event vs no-event average ramp.
   - Common 100% preinjury baseline shown before a compressed injury gap.
   - Ten scheduled games displayed, with the five-game landmark marked.
   - Bootstrap 95% CI ribbons + error bars and compact risk-set counts.
   - Direct labels, no legend.

C. Outcome specificity.
   - Simpler outcome labels.
   - No title.

D. Average ramp by time out.
   - Common 100% preinjury baseline shown before a compressed injury gap.
   - Mean postreturn ramp over 10 played games.
   - Bootstrap 95% CI ribbons + error bars.
   - Direct labels, no legend.
   - Vertical spacing is tuned for clear separation of the four trajectories/labels.

E. Severity × role heatmap.
   - No panel title.

No subfigure has an internal title. Use the external figure caption generated at the end.

INPUT:
    NBA_RTP_AI_ANALYSIS_BUNDLE.zip
or:
    NBA_Injury_RTP_MultiSeason_QC_and_Episodes.xlsx
"""

from pathlib import Path
import zipfile, shutil, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(".")
OUT = ROOT / "NBA_RTP_SLOAN_SIGNIFICANCE"
PAN = OUT / "panels"
OUT.mkdir(exist_ok=True)
PAN.mkdir(exist_ok=True)

ZIP_NAME = "NBA_RTP_AI_ANALYSIS_BUNDLE.zip"
XLSX_NAME = "NBA_Injury_RTP_MultiSeason_QC_and_Episodes.xlsx"
rng = np.random.default_rng(42)


# ============================================================
# Publication-style appearance (visual changes only)
# ============================================================
# Optional: upload an Arial .ttf/.otf file to the Colab working directory.
# The code will register it automatically; otherwise it falls back to a
# metrically similar sans-serif font without changing any analysis.
def _configure_publication_style():
    font_files = list(ROOT.glob("*.ttf")) + list(ROOT.glob("*.otf"))
    for fp in font_files:
        try:
            fm.fontManager.addfont(str(fp))
        except Exception:
            pass

    available = {f.name for f in fm.fontManager.ttflist}
    if "Arial" in available:
        family = "Arial"
    elif "Liberation Sans" in available:
        family = "Liberation Sans"
    else:
        family = "DejaVu Sans"

    mpl.rcParams.update({
        "font.family": family,
        # Deliberately large type relative to the plotting area, as in many
        # Nature-family multi-panel figures. This keeps labels readable when
        # the whole figure is viewed at manuscript width.
        "font.size": 13.2,
        "axes.labelsize": 14.6,
        "axes.titlesize": 13.6,
        "xtick.labelsize": 12.2,
        "ytick.labelsize": 12.2,
        "legend.fontsize": 11.6,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "lines.linewidth": 1.7,
        "lines.markersize": 4.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return family, font_files

FONT_FAMILY, UPLOADED_FONT_FILES = _configure_publication_style()

# Cohesive NBA-inspired palette. Analytical panels use a consistent semantic:
# gold = lower/shorter/comparison; purple = greater/longer/injury-related.
# Panel A is the one exception because the colors identify specific players and
# therefore follow muted team colors.
INK = "#242424"
NEUTRAL = "#707070"
REFERENCE = "#B9BDC2"
LIGHT_NEUTRAL = "#ECEDEF"

ANALYSIS_GOLD = "#C79A33"
PALE_GOLD = "#E7CF8A"
LAVENDER = "#AE95C8"
ANALYSIS_PURPLE = "#7957A5"
DEEP_PURPLE = "#563A78"

# Muted team-inspired colors for the three illustrative players.
LAKERS_PURPLE = "#6B50A1"
ORLANDO_BLUE = "#3A78A5"
NUGGETS_NAVY = "#356184"

PLAYER_COLORS = {
    "Luka Doncic": LAKERS_PURPLE,
    "Jalen Suggs": ORLANDO_BLUE,
    "Jamal Murray": NUGGETS_NAVY,
}
PLAYER_DISPLAY_NAMES = {
    "Luka Doncic": "Luka Dončić",
    "Jalen Suggs": "Jalen Suggs",
    "Jamal Murray": "Jamal Murray",
}

# Increasing time away from play moves from gold to purple.
TIMEOUT_COLORS = [ANALYSIS_GOLD, PALE_GOLD, LAVENDER, DEEP_PURPLE]
# Adverse later-injury group is purple; comparison group is gold.
EVENT_COLORS = {False: ANALYSIS_GOLD, True: DEEP_PURPLE}
# Low workload = pale gold; high workload = purple.
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "nba_purple_gold",
    ["#FFF8E7", "#E9D7A6", "#C4AFD3", "#7B5AA6", "#51356F"]
)

PLAYER_EXAMPLES = [
    {
        "name": "Luka Doncic",
        "episode_id": "-82448609703109_2024-01-11_38",
        "descriptor": "immediate full restoration",
    },
    {
        "name": "Jalen Suggs",
        # 2022-23 Orlando episode: 19 games missed. Return workload is ~17%
        # of preinjury minutes in the first game, then remains markedly below
        # baseline across the displayed 10-game window.
        "episode_id": "-84975179258134_2022-11-27_19",
        "descriptor": "very low early workload",
    },
    {
        "name": "Jamal Murray",
        "episode_id": "-43110900467630_2024-03-23_68",
        "descriptor": "restricted start, then step-up",
    },
]

FINAL_5GAME_OR = 1.12
FINAL_5GAME_LO = 1.02
FINAL_5GAME_HI = 1.22

FINAL_SPECIFICITY = pd.DataFrame([
    ["Any later injury",      0.75, 0.64, 0.88],
    ["New injury elsewhere",  0.74, 0.61, 0.89],
    ["Same body part",        0.91, 0.74, 1.13],
    ["Exact recurrence",      1.00, 0.73, 1.38],
], columns=["Outcome", "OR", "Low", "High"])


# ============================================================
# Input
# ============================================================
def locate_workbook():
    direct = ROOT / XLSX_NAME
    if direct.exists():
        return direct

    zpath = ROOT / ZIP_NAME
    if not zpath.exists():
        # Colab/browser uploads often append " (1)", " (2)", etc.
        # Accept those automatically rather than requiring a manual rename.
        candidates = sorted(ROOT.glob("NBA_RTP_AI_ANALYSIS_BUNDLE*.zip"))
        if candidates:
            zpath = candidates[0]
        else:
            raise FileNotFoundError(
                f"Place {ZIP_NAME} (or a similarly named copy) or {XLSX_NAME} next to this script."
            )

    extract_dir = ROOT / "_NBA_RTP_TMP"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()

    with zipfile.ZipFile(zpath, "r") as z:
        z.extractall(extract_dir)

    matches = list(extract_dir.rglob(XLSX_NAME))
    if not matches:
        raise FileNotFoundError(f"{XLSX_NAME} not found inside ZIP.")
    return matches[0]


XLSX = locate_workbook()

EP_COLS = [
    "episode_id", "player_id", "player_name", "games_missed",
    "pre_median_minutes", "primary_analysis_eligible",
    "next_subsequent_injury_absence_date",
    "days_to_next_subsequent_injury_absence",
    "played_games_before_next_subsequent_injury_absence",
    "next_injury_body_part", "next_injury_diagnosis_class",
    "subsequent_any_injury_absence_within_30d",
]
W_COLS = [
    "episode_id", "relative_game", "game_date", "minutes",
    "played_actual", "absence_reason_family", "report_reason",
]

ep = pd.read_excel(XLSX, sheet_name="injury_episodes", usecols=EP_COLS, engine="openpyxl")
w = pd.read_excel(XLSX, sheet_name="rtp_game_windows", usecols=W_COLS, engine="openpyxl")

w["game_date"] = pd.to_datetime(w["game_date"], errors="coerce")
ep["next_subsequent_injury_absence_date"] = pd.to_datetime(
    ep["next_subsequent_injury_absence_date"], errors="coerce"
)

eligible = ep[
    (ep["primary_analysis_eligible"] == 1) &
    (ep["pre_median_minutes"] >= 10)
].copy()

eligible_ids = set(eligible["episode_id"])
baseline = eligible.set_index("episode_id")["pre_median_minutes"]


# ============================================================
# Helpers
# ============================================================
def clean(ax):
    """Minimal publication axis: thin spines and outward ticks on left/bottom only."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(
        axis="both", which="both", direction="out",
        top=False, right=False, width=0.8, colors=INK, pad=3
    )
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.grid(False)


def save_panel(name, width=7.0, height=4.6):
    fig = plt.gcf()
    fig.set_size_inches(width, height)
    fig.tight_layout(pad=0.55)
    path = PAN / name
    fig.savefig(path, dpi=450, bbox_inches="tight", pad_inches=0.04)
    # Also preserve a vector copy for manuscript assembly.
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def bootstrap_ci(vals_by_x, B=700, seed=42):
    """Input: list of 1D arrays for each x. Return 2 x N percentile CI."""
    rr = np.random.default_rng(seed)
    lo, hi = [], []
    for vals in vals_by_x:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            lo.append(np.nan); hi.append(np.nan); continue
        boots = np.empty(B)
        for b in range(B):
            boots[b] = rr.choice(vals, size=len(vals), replace=True).mean()
        q = np.percentile(boots, [2.5, 97.5])
        lo.append(q[0]); hi.append(q[1])
    return np.array(lo), np.array(hi)



def star_from_q(q):
    if pd.isna(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def cluster_global_group_p(df, group_col, cluster_col="player_id"):
    """
    Player-clustered omnibus test that all group indicators are zero.
    Used in Panel B, where four time-out groups are compared at each game.
    """
    d = df[[group_col, "workload_pct", cluster_col]].dropna().copy()
    d[group_col] = d[group_col].astype(str)
    X = pd.get_dummies(d[group_col], drop_first=True, dtype=float)
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(
        d["workload_pct"].astype(float), X
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": d[cluster_col]}
    )
    if X.shape[1] <= 1:
        return np.nan
    R = np.zeros((X.shape[1]-1, X.shape[1]))
    for i in range(X.shape[1]-1):
        R[i, i+1] = 1
    return float(model.wald_test(R, scalar=True).pvalue)


def cluster_binary_p(df, cluster_col="player_id"):
    """
    Player-clustered comparison of later-event vs no-later-event groups.
    """
    d = df[["later_event", "workload_pct", cluster_col]].dropna().copy()
    X = sm.add_constant(d["later_event"].astype(float), has_constant="add")
    model = sm.OLS(
        d["workload_pct"].astype(float), X
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": d[cluster_col]}
    )
    return float(model.pvalues["later_event"])


def first_n_played(eid, n=10, truncate_at_event=True):
    row = eligible[eligible["episode_id"] == eid].iloc[0]
    z = w[
        (w["episode_id"] == eid) &
        (w["relative_game"] >= 0) &
        (w["played_actual"] == True)
    ].sort_values("relative_game").copy()

    if truncate_at_event and pd.notna(row["next_subsequent_injury_absence_date"]):
        z = z[z["game_date"] < row["next_subsequent_injury_absence_date"]]

    z = z.head(n).copy()
    z["played_index"] = np.arange(len(z))
    z["workload_pct"] = 100 * z["minutes"] / row["pre_median_minutes"]
    return z


def draw_common_preinjury_and_gap(ax, x_pre=(-2.7, -1.6), x_gap=(-1.6, -0.25)):
    """Common schematic baseline/gap; time in the injury gap is intentionally compressed."""
    ax.plot(x_pre, [100, 100], linewidth=1.25, color=NEUTRAL, solid_capstyle="round")
    ax.plot([x_pre[1], x_pre[1]], [100, 0], linewidth=0.9, color=NEUTRAL)
    ax.plot(x_gap, [0, 0], linewidth=1.05, color=NEUTRAL, solid_capstyle="round")

    # Visual time-break slashes inside the zero plateau.
    xb = (x_gap[0] + x_gap[1]) / 2
    ax.plot([xb - .13, xb - .03], [-1.5, 3.5], linewidth=0.9, color=NEUTRAL)
    ax.plot([xb + .03, xb + .13], [-1.5, 3.5], linewidth=0.9, color=NEUTRAL)

    # Keep the compressed-gap label inside the plotting area so it cannot
    # collide with x tick labels or the x-axis title.
    ax.text(
        xb, 6.0, "Injury",
        ha="center", va="bottom", fontsize=11.1, color=NEUTRAL
    )


def label_line(ax, x, y, text, line, dy=0.0, fontsize=12.0):
    ax.text(
        x, y + dy, text, fontsize=fontsize, va="center", ha="left",
        color=line.get_color(), linespacing=1.02
    )


# ============================================================
# A. THREE REAL PLAYER RAMPS
# ============================================================

# Population median/IQR over first 10 played games.
pop = w[
    w["episode_id"].isin(eligible_ids) &
    (w["relative_game"] >= 0) &
    (w["played_actual"] == True)
].sort_values(["episode_id", "relative_game"]).copy()

pop["played_index"] = pop.groupby("episode_id").cumcount()
pop = pop[pop["played_index"] < 10]
pop["baseline"] = pop["episode_id"].map(baseline)
pop["workload_pct"] = 100 * pop["minutes"] / pop["baseline"]

pop_summary = (
    pop.groupby("played_index")["workload_pct"]
    .agg(
        median="median",
        q25=lambda s: s.quantile(.25),
        q75=lambda s: s.quantile(.75),
    )
    .reindex(range(10))
)

fig, ax = plt.subplots()

# Common preinjury baseline and compressed injury absence.
draw_common_preinjury_and_gap(ax)

# NBA background only during return phase.
rx = np.arange(0, 10)
ax.fill_between(
    rx,
    pop_summary["q25"].to_numpy(),
    pop_summary["q75"].to_numpy(),
    color=NEUTRAL, alpha=0.10, linewidth=0
)
ax.plot(
    rx,
    pop_summary["median"].to_numpy(),
    linestyle=(0, (3, 2)),
    linewidth=1.25, color=NEUTRAL
)
ax.text(
    8.35, pop_summary["median"].iloc[-1] - 7, "NBA median",
    fontsize=11.0, color=NEUTRAL
)

for meta in PLAYER_EXAMPLES:
    row = eligible[eligible["episode_id"] == meta["episode_id"]].iloc[0]
    z = first_n_played(meta["episode_id"], n=10, truncate_at_event=True)

    x = z["played_index"].to_numpy()
    y = z["workload_pct"].to_numpy()

    # Connect from common zero plateau into the player's first return game.
    x_plot = np.concatenate([[-0.25], x])
    y_plot = np.concatenate([[0], y])

    line, = ax.plot(
        x_plot, y_plot, marker="o", markersize=3.9, linewidth=1.8,
        color=PLAYER_COLORS[meta["name"]], markeredgecolor="white",
        markeredgewidth=0.45
    )

    # Direct labels stay intentionally simple; absence duration and qualitative
    # descriptors remain in the caption rather than competing with the data.
    label = PLAYER_DISPLAY_NAMES[meta["name"]]
    player_dy = {
        "Luka Doncic": 2.8,
        "Jalen Suggs": 0.8,
        "Jamal Murray": -2.0,
    }[meta["name"]]
    label_line(ax, x[-1] + 0.18, y[-1], label, line, dy=player_dy, fontsize=12.3)

    if meta["name"] == "Luka Doncic" and int(row["subsequent_any_injury_absence_within_30d"]) == 1:
        played_before = row["played_games_before_next_subsequent_injury_absence"]
        ex = float(played_before) - 0.10 if pd.notna(played_before) else x[-1] + 0.5
        ey = y[-1]
        ax.scatter(
            [ex], [ey], marker="X", s=72, zorder=10,
            color=PLAYER_COLORS[meta["name"]], edgecolor="white", linewidth=0.55
        )
        body = str(row["next_injury_body_part"]).replace("_", " ")
        dx = str(row["next_injury_diagnosis_class"]).replace("_", " ")
        ax.annotate(
            f"Second injury-related absence\n{body} {dx}",
            xy=(ex, ey),
            xytext=(ex + 0.48, min(141, ey + 20)),
            arrowprops=dict(arrowstyle="->", color=NEUTRAL, lw=0.8),
            fontsize=11.0, color=INK
        )

ax.axhline(100, linestyle=(0, (1.5, 1.5)), linewidth=0.9, color=REFERENCE, zorder=0)
ax.axvline(0, linestyle=(0, (1.5, 1.5)), linewidth=0.8, color=REFERENCE, zorder=0)

ax.set_xlim(-3.0, 11.45)
ax.set_ylim(-5, 145)
ax.set_xticks(
    [-2.2] + list(range(0,10)),
    ["Preinjury"] + [str(i) for i in range(1,11)]
)
ax.set_xlabel("Games after injury return", labelpad=8)
ax.set_ylabel("% of usual preinjury minutes played", labelpad=7)
clean(ax)
panel_A = save_panel("A_real_player_ramps_simplified.png", 6.25, 4.25)


# ============================================================
# D. RAMPS BY TIME MISSED — 10 PLAYED GAMES + 95% CI + GLOBAL TESTS
# ============================================================
games_missed_map = eligible.set_index("episode_id")["games_missed"]
player_map = eligible.set_index("episode_id")["player_id"]

pop["games_missed"] = pop["episode_id"].map(games_missed_map)
pop["player_id"] = pop["episode_id"].map(player_map)
pop["time_out"] = pd.cut(
    pop["games_missed"],
    [1,3,7,15,np.inf],
    labels=["2–3 games out", "4–7 games out", "8–15 games out", "≥16 games out"]
)

fig, ax = plt.subplots()
draw_common_preinjury_and_gap(ax)

groups = ["2–3 games out", "4–7 games out", "8–15 games out", "≥16 games out"]

for gi, group in enumerate(groups):
    zz = pop[pop["time_out"].astype(str) == group]

    means = []
    vals_by_x = []
    for g in range(10):
        vals = zz.loc[zz["played_index"] == g, "workload_pct"].dropna().to_numpy()
        vals_by_x.append(vals)
        means.append(np.mean(vals) if len(vals) else np.nan)

    means = np.array(means)
    lo, hi = bootstrap_ci(vals_by_x, B=700, seed=100+gi)

    x = np.arange(0,10)
    color = TIMEOUT_COLORS[gi]
    line, = ax.plot(
        np.concatenate([[-0.25], x]),
        np.concatenate([[0], means]),
        marker="o", markersize=3.5, linewidth=1.55, color=color,
        markeredgecolor="white", markeredgewidth=0.4
    )
    # Show the 95% CI both as a faint ribbon and as compact capped error bars.
    # The ribbon carries the interval continuously; the error bars make the
    # game-specific uncertainty legible when the full figure is viewed small.
    ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0, zorder=1)
    ax.errorbar(
        x, means,
        yerr=np.vstack([means - lo, hi - means]),
        fmt="none", ecolor=color, elinewidth=0.75, capsize=2.0,
        capthick=0.75, alpha=0.72, zorder=2
    )

    # End labels are deliberately spaced rather than sitting directly on nearly
    # overlapping endpoints. Short leader segments preserve the connection.
    label_y = {
        "2–3 games out": 108.5,
        "4–7 games out": 101.0,
        "8–15 games out": 93.5,
        "≥16 games out": 86.0,
    }[group]
    ax.plot([9.0, 9.16], [means[-1], label_y], color=color, linewidth=0.8)
    label_line(ax, 9.20, label_y, group, line, fontsize=11.7)

# Player-clustered omnibus group test at each game.
p_Duration = []
for g in range(10):
    p_Duration.append(
        cluster_global_group_p(
            pop[pop["played_index"] == g],
            group_col="time_out",
            cluster_col="player_id"
        )
    )
_, q_Duration, _, _ = multipletests(p_Duration, alpha=0.05, method="fdr_bh")

for g, q in enumerate(q_Duration):
    star = star_from_q(q)
    if star:
        ax.text(g, 114.0, star, ha="center", va="bottom", fontsize=11.5)

ax.axhline(100, linestyle=(0, (1.5, 1.5)), linewidth=0.9, color=REFERENCE, zorder=0)
ax.axvline(0, linestyle=(0, (1.5, 1.5)), linewidth=0.8, color=REFERENCE, zorder=0)

ax.set_xlim(-3.0, 11.45)
ax.set_ylim(-5, 121)
ax.set_xticks(
    [-2.2] + list(range(0,10)),
    ["Preinjury"] + [str(i) for i in range(1,11)]
)
ax.set_xlabel("Games after injury return", labelpad=8)
ax.set_ylabel("Mean workload (% of usual)", labelpad=7)
clean(ax)
panel_D = save_panel("D_ramps_by_time_out_CI_significance.png", 6.25, 4.20)

pd.DataFrame({
    "played_game": np.arange(1,11),
    "clustered_global_p": p_Duration,
    "BH_FDR_q": q_Duration,
    "significance": [star_from_q(q) for q in q_Duration],
}).to_csv(OUT/"D_gamewise_group_tests.csv", index=False)


# ============================================================
# E. ROLE × TIME-OUT HEATMAP
# ============================================================
planned_terms = (
    r"management|recovery|reconditioning|return to competition|"
    r"maintenance|rehabilitation|conditioning|\brest\b"
)

post5 = w[
    w["episode_id"].isin(eligible_ids) &
    w["relative_game"].between(0,4)
].copy()

post5["is_planned"] = (
    post5["absence_reason_family"].eq("rest") |
    post5["report_reason"].fillna("").str.lower().str.contains(planned_terms, regex=True)
)

post5["unplanned_injury_interrupt"] = (
    (~post5["played_actual"].fillna(False)) &
    post5["absence_reason_family"].eq("injury") &
    (~post5["is_planned"])
)

bad_ids = set(post5.loc[post5["unplanned_injury_interrupt"], "episode_id"])
counts = post5.groupby("episode_id")["relative_game"].nunique()
complete_ids = set(counts[counts == 5].index)
clean5_ids = sorted((eligible_ids & complete_ids) - bad_ids)

sched5 = post5[post5["episode_id"].isin(clean5_ids)].copy()
sched5["baseline"] = sched5["episode_id"].map(baseline)
sched5["workload_pct"] = 100 * np.minimum(
    sched5["minutes"].fillna(0) / sched5["baseline"], 1.0
)

rw5 = sched5.groupby("episode_id")["workload_pct"].mean().rename("return_workload_5")

h = eligible[eligible["episode_id"].isin(clean5_ids)].copy()
h = h.merge(rw5, left_on="episode_id", right_index=True, how="left")

h["time_out"] = pd.cut(
    h["games_missed"], [1,3,7,15,np.inf],
    labels=["2–3", "4–7", "8–15", "≥16"]
)
h["role"] = pd.cut(
    h["pre_median_minutes"], [10,20,30,np.inf],
    right=False,
    labels=["10–<20 MPG", "20–<30 MPG", "≥30 MPG"]
)

heat = h.pivot_table(
    index="time_out", columns="role", values="return_workload_5",
    aggfunc="mean", observed=False
).reindex(
    index=["2–3", "4–7", "8–15", "≥16"],
    columns=["10–<20 MPG", "20–<30 MPG", "≥30 MPG"]
)
heat_n = h.pivot_table(
    index="time_out", columns="role", values="episode_id",
    aggfunc="count", observed=False
).reindex(
    index=["2–3", "4–7", "8–15", "≥16"],
    columns=["10–<20 MPG", "20–<30 MPG", "≥30 MPG"]
)

fig, ax = plt.subplots()
heat_arr = heat.to_numpy(float)
im = ax.imshow(heat_arr, aspect="auto", cmap=HEAT_CMAP, interpolation="nearest")
ax.set_xticks(range(3), ["10–<20", "20–<30", "≥30"])
ax.set_yticks(range(4), ["2–3", "4–7", "8–15", "≥16"])

for i in range(4):
    for j in range(3):
        value = heat.iloc[i, j]
        threshold = np.nanmin(heat_arr) + 0.62 * (np.nanmax(heat_arr) - np.nanmin(heat_arr))
        txt_color = "white" if value >= threshold else INK
        ax.text(
            j, i,
            f"{value:.0f}%\n(n={int(heat_n.iloc[i,j])})",
            ha="center", va="center", fontsize=11.3, color=txt_color, linespacing=0.98
        )

cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
cb.set_label("Mean 5-game workload (%)", fontsize=11.5, labelpad=8)
cb.ax.tick_params(direction="out", width=0.7, length=3.2, labelsize=10.5)
cb.outline.set_linewidth(0.7)
ax.set_xlabel("Usual preinjury playing time\n(minutes per game)", labelpad=7)
ax.set_ylabel("Games missed", labelpad=7)
clean(ax)
panel_E = save_panel("E_role_by_time_heatmap.png", 4.05, 4.20)


# ============================================================
# B. EVENT VS NO EVENT — 10 SCHEDULED GAMES AS A RISK-SET PLOT
# ============================================================
# The exposure is fixed after scheduled game 5.
# Games 6–10 are shown descriptively, censoring an episode at its first
# subsequent unplanned injury-related absence. The risk-set/event counts
# printed below the axis make this survivor attrition explicit.

landmark_date = (
    sched5[sched5["relative_game"] == 4]
    .set_index("episode_id")["game_date"]
)

future = w[
    w["episode_id"].isin(clean5_ids) &
    (w["relative_game"] > 4)
].copy()

future["is_planned"] = (
    future["absence_reason_family"].eq("rest") |
    future["report_reason"].fillna("").str.lower().str.contains(
        planned_terms, regex=True
    )
)
future["unplanned_injury"] = (
    (~future["played_actual"].fillna(False)) &
    future["absence_reason_family"].eq("injury") &
    (~future["is_planned"])
)
future["landmark_date"] = future["episode_id"].map(landmark_date)
future["days_after_landmark"] = (
    future["game_date"] - future["landmark_date"]
).dt.days

# First qualifying event after the 5-game landmark, across the entire 30-day window.
event_rows = future[
    future["unplanned_injury"] &
    future["days_after_landmark"].between(1,30)
].copy()

first_event_rel = (
    event_rows.groupby("episode_id")["relative_game"]
    .min()
)

later_event_ids = set(first_event_rel.index)

# First 10 scheduled games after return.
sched10 = w[
    w["episode_id"].isin(clean5_ids) &
    w["relative_game"].between(0,9)
].copy()

sched10["baseline"] = sched10["episode_id"].map(baseline)
sched10["workload_pct"] = 100 * np.minimum(
    sched10["minutes"].fillna(0) / sched10["baseline"], 1.0
)
sched10["later_event"] = sched10["episode_id"].isin(later_event_ids)
sched10["event_rel"] = sched10["episode_id"].map(first_event_rel)
sched10["player_id"] = sched10["episode_id"].map(
    eligible.set_index("episode_id")["player_id"]
)

# Censor at the event game itself: once an injury absence occurs, that game and
# later games do not contribute workload to the ramp curve.
risk10 = sched10[
    sched10["event_rel"].isna() |
    (sched10["relative_game"] < sched10["event_rel"])
].copy()

fig, ax = plt.subplots()
draw_common_preinjury_and_gap(ax)

curve_stats = {}
for idx, (flag, label) in enumerate([
    (False, "No later injury-related absence"),
    (True,  "Later injury-related absence"),
]):
    zz = risk10[risk10["later_event"] == flag]

    means = []
    vals_by_x = []
    counts = []
    for g in range(10):
        vals = zz.loc[zz["relative_game"] == g, "workload_pct"].dropna().to_numpy()
        vals_by_x.append(vals)
        counts.append(len(vals))
        means.append(np.mean(vals) if len(vals) else np.nan)

    means = np.array(means)
    lo, hi = bootstrap_ci(vals_by_x, B=800, seed=200+idx)

    x = np.arange(0,10)
    color = EVENT_COLORS[flag]
    line, = ax.plot(
        np.concatenate([[-0.25], x]),
        np.concatenate([[0], means]),
        marker="o", markersize=3.7, linewidth=1.65, color=color,
        markeredgecolor="white", markeredgewidth=0.4
    )
    ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0, zorder=1)
    ax.errorbar(
        x, means,
        yerr=np.vstack([means - lo, hi - means]),
        fmt="none", ecolor=color, elinewidth=0.80, capsize=2.1,
        capthick=0.80, alpha=0.75, zorder=2
    )

    short_label = {
        False: "No later injury",
        True: "Later injury",
    }[flag]
    label_y = {False: 80.8, True: 87.4}[flag]
    ax.plot([9.0, 9.12], [means[-1], label_y], color=color, linewidth=0.8)
    label_line(
        ax, 9.16, label_y, short_label, line,
        fontsize=11.7
    )

    curve_stats[flag] = {
        "mean": means,
        "lo": lo,
        "hi": hi,
        "n": np.array(counts),
    }

# Significance only across the PRE-SPECIFIED five-game exposure window.
p_D = []
for g in range(5):
    p_D.append(
        cluster_binary_p(
            risk10[risk10["relative_game"] == g],
            cluster_col="player_id"
        )
    )
_, q_D, _, _ = multipletests(p_D, alpha=0.05, method="fdr_bh")

for g, q in enumerate(q_D):
    star = star_from_q(q)
    if star:
        ax.text(g, 102.2, star, ha="center", va="bottom", fontsize=11.3)

# Mark end of exposure window. The longer explanation lives in the caption,
# keeping the plot itself uncluttered.
ax.axvline(4.5, linestyle=(0, (3, 2)), linewidth=0.9, color=NEUTRAL)
ax.text(
    4.70, 108.0, "5-game\nlandmark",
    ha="left", va="top", fontsize=9.8, color=NEUTRAL, linespacing=0.95
)

# New injury absences at each scheduled game.
event_counts = np.zeros(10, dtype=int)
for rel, n in first_event_rel.value_counts().items():
    if 0 <= int(rel) <= 9:
        event_counts[int(rel)] = int(n)

event_group_remaining = curve_stats[True]["n"]

# Compact risk-set annotation rows. Place them in axes coordinates below the
# tick labels so they read as a tiny table rather than colliding with the data.
trans = ax.get_xaxis_transform()
for g in range(10):
    ax.text(
        g, -0.185, f"{event_group_remaining[g]}",
        transform=trans, ha="center", va="top", fontsize=9.7,
        color=INK, clip_on=False
    )
    ax.text(
        g, -0.275, f"{event_counts[g]}" if event_counts[g] > 0 else "–",
        transform=trans, ha="center", va="top", fontsize=9.7,
        color=INK, clip_on=False
    )

ax.text(
    -0.035, -0.185, "At risk", transform=ax.transAxes,
    ha="right", va="top", fontsize=9.8, color=NEUTRAL, clip_on=False
)
ax.text(
    -0.035, -0.275, "New injury", transform=ax.transAxes,
    ha="right", va="top", fontsize=9.8, color=NEUTRAL, clip_on=False
)
ax.axhline(100, linestyle=(0, (1.5, 1.5)), linewidth=0.9, color=REFERENCE, zorder=0)
ax.axvline(0, linestyle=(0, (1.5, 1.5)), linewidth=0.8, color=REFERENCE, zorder=0)

ax.set_xlim(-3.0, 11.35)
ax.set_ylim(-5, 111)
ax.set_xticks(
    [-2.2] + list(range(0,10)),
    ["Preinjury"] + [str(i) for i in range(1,11)]
)
ax.set_xlabel("Games after injury return", labelpad=61)
ax.set_ylabel("Mean workload (% of usual)", labelpad=7)
clean(ax)
panel_B = save_panel("B_event_vs_none_10game_riskset.png", 6.25, 4.25)

# Save exact pointwise tests and risk-set counts.
b_test = pd.DataFrame({
    "scheduled_game": np.arange(1,6),
    "clustered_group_p": p_D,
    "BH_FDR_q": q_D,
    "significance": [star_from_q(q) for q in q_D],
})
b_test.to_csv(OUT/"B_exposure_window_pointwise_tests.csv", index=False)

b_risk = pd.DataFrame({
    "scheduled_game": np.arange(1,11),
    "later_injury_group_remaining_in_curve": event_group_remaining,
    "new_injury_absences_at_game": event_counts,
    "later_injury_mean_workload": curve_stats[True]["mean"],
    "later_injury_CI_low": curve_stats[True]["lo"],
    "later_injury_CI_high": curve_stats[True]["hi"],
    "no_later_injury_mean_workload": curve_stats[False]["mean"],
    "no_later_injury_CI_low": curve_stats[False]["lo"],
    "no_later_injury_CI_high": curve_stats[False]["hi"],
})
b_risk.to_csv(OUT/"B_10game_riskset_summary.csv", index=False)


# ============================================================
# C. SIMPLIFIED OUTCOME SPECIFICITY
# ============================================================
fig, ax = plt.subplots()
f = FINAL_SPECIFICITY.copy()
y = np.arange(len(f))[::-1]

ax.errorbar(
    f["OR"], y,
    xerr=[f["OR"]-f["Low"], f["High"]-f["OR"]],
    fmt="o", markersize=5.8, color=DEEP_PURPLE, ecolor=DEEP_PURPLE,
    markerfacecolor=DEEP_PURPLE, markeredgecolor="white", markeredgewidth=0.5,
    elinewidth=1.15, capsize=2.5, capthick=0.9
)
ax.axvline(1, linestyle=(0, (3, 2)), linewidth=0.9, color=REFERENCE, zorder=0)
ax.set_yticks(y, f["Outcome"])
ax.set_xlabel("Adjusted odds ratio\n(95% CI)", labelpad=8)
clean(ax)
panel_C = save_panel("C_outcome_specificity_simple.png", 4.55, 4.20)


# ============================================================
# Export supporting data
# ============================================================
pop_summary.reset_index().to_csv(OUT/"A_population_median_IQR.csv", index=False)
heat.to_csv(OUT/"E_heatmap_mean_workload.csv")
heat_n.to_csv(OUT/"E_heatmap_counts.csv")

FINAL_SPECIFICITY.to_csv(OUT/"C_final_outcome_specificity.csv", index=False)


# ============================================================
# Composite: A/B on the top row; C/D/E on the second row.
# Panels within a row are scaled to the SAME DISPLAY HEIGHT. This keeps A/B
# visually aligned and likewise keeps C/D/E aligned, regardless of differences
# in axis labels, risk-set rows, or color bars. Aspect ratios are preserved.
# ============================================================
def _load_rgb(path):
    return Image.open(path).convert("RGB")

images = {
    "a": _load_rgb(panel_A),
    "b": _load_rgb(panel_B),
    "c": _load_rgb(panel_C),
    "d": _load_rgb(panel_D),
    "e": _load_rgb(panel_E),
}

DPI = 450
outer = int(0.18 * DPI)
col_gap = int(0.20 * DPI)
row_gap = int(0.22 * DPI)
label_gutter = int(0.17 * DPI)

# Equal visual height within each row. The bottom row is only slightly shorter
# than the top row so all labels remain readable at manuscript width.
top_target_h = int(4.20 * DPI)
bottom_target_h = int(4.05 * DPI)

def scale_to_height(im, target_h):
    scale = target_h / im.height
    return im.resize(
        (max(1, int(round(im.width * scale))), target_h),
        Image.Resampling.LANCZOS
    )

scaled = {
    "a": scale_to_height(images["a"], top_target_h),
    "b": scale_to_height(images["b"], top_target_h),
    "c": scale_to_height(images["c"], bottom_target_h),
    "d": scale_to_height(images["d"], bottom_target_h),
    "e": scale_to_height(images["e"], bottom_target_h),
}

def row_width(keys):
    return sum(label_gutter + scaled[k].width for k in keys) + col_gap * (len(keys)-1)

top_keys = ["a", "b"]
bottom_keys = ["c", "d", "e"]
top_content_w = row_width(top_keys)
bottom_content_w = row_width(bottom_keys)
content_w = max(top_content_w, bottom_content_w)
W = outer*2 + content_w
H = outer*2 + top_target_h + row_gap + bottom_target_h
canvas = Image.new("RGB", (W, H), "white")

# Nature-style lowercase panel labels live in a dedicated left gutter.
try:
    bold_path = fm.findfont(fm.FontProperties(family=FONT_FAMILY, weight="bold"))
    letter_font = ImageFont.truetype(bold_path, int(0.205 * DPI))
except Exception:
    letter_font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(0.205 * DPI))

draw = ImageDraw.Draw(canvas)

def paste_row(keys, y0, total_w):
    x = outer + (content_w - total_w)//2
    for key in keys:
        draw.text((x + 2, y0 + 1), key, fill=INK, font=letter_font)
        x_img = x + label_gutter
        canvas.paste(scaled[key], (x_img, y0))
        x = x_img + scaled[key].width + col_gap

paste_row(top_keys, outer, top_content_w)
paste_row(bottom_keys, outer + top_target_h + row_gap, bottom_content_w)

composite = OUT/"Figure_1_Sloan_Simplified.png"
canvas.save(composite, dpi=(DPI, DPI))

# External caption text.
caption = """Figure 1. Early competitive workload restoration after injury varies substantially across NBA players and is associated with subsequent injury-related absence. (a) Three real return-to-play examples illustrate distinct workload-restoration patterns. Playing time is normalized to each player's preinjury baseline (100%); the injury interval is compressed horizontally, and the dashed line with shaded band shows the NBA median and interquartile range during the first 10 played games after return. Luka Dončić returned immediately to full workload and subsequently had another ankle injury-related absence, Jalen Suggs returned at a very low workload and remained well below baseline through the displayed period, and Jamal Murray remained restricted for several games before an abrupt step-up. Individual trajectories are illustrative rather than causal comparisons. (b) Mean schedule-aware workload through the first 10 scheduled games after return among episodes with and without a subsequent unplanned injury-related absence during the 30 days after the five-game landmark. Episodes are censored from the curve at the first later injury absence; the rows beneath the x-axis show the number of later-injury episodes remaining in the curve and the number of new injury absences at each scheduled game. Ribbons and capped error bars show bootstrap 95% confidence intervals. Pointwise asterisks are shown only for games 1–5, the prespecified exposure window, and denote player-clustered between-group differences after Benjamini–Hochberg correction across those five tests. Games 6–10 are descriptive because the risk set becomes survivor-selected after follow-up begins. The final corrected longitudinal model estimated an odds ratio of 1.12 (95% CI, 1.02–1.22) per 10-percentage-point increase in five-game Return Workload. (c) Outcome-specific adjusted associations from the final longitudinal analysis. More conservative early workload was associated with fewer subsequent injury-related absences overall and fewer anatomically different second injuries, but not with fewer exact recurrences of the original injury. (d) Mean playing time over the first 10 played games after return, stratified by games missed before return. Ribbons and capped error bars show bootstrap 95% confidence intervals. Asterisks denote a significant global difference among the four absence-duration groups at that game after player-clustered testing and Benjamini–Hochberg correction across the 10 displayed games (*q<.05, **q<.01, ***q<.001). Players returning from longer absences began farther below their normal minutes and generally restored workload more slowly. (e) Mean five-game schedule-aware Return Workload according to time lost and usual preinjury playing-time role. Short absences and high-minute players received the highest early workload, whereas long absences were managed more conservatively across roles."""
(OUT/"Figure_1_caption.txt").write_text(caption)

print(f"Figure font: {FONT_FAMILY}")
if FONT_FAMILY != "Arial":
    print("Tip: upload an Arial .ttf/.otf file with the data bundle, then rerun for true Arial.")
print("Created:")
print(composite)
print(OUT/"Figure_1_caption.txt")
