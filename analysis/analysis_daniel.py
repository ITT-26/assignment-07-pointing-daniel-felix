import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

matplotlib.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.prop_cycle": matplotlib.cycler(color=[
        "#0A84FF", "#30D158", "#FF453A", "#FFD60A"   # blue, green, red, yellow
    ]),
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
RESULT_DIR = os.path.join(PROJECT_ROOT, "assets")
os.makedirs(RESULT_DIR, exist_ok=True)

TECHNIQUE_ORDER  = ["mouse", "touchpad", "mouse+150ms", "pose"]
TECHNIQUE_LABELS = {
    "mouse":      "Mouse",
    "mouse+150ms":"Mouse +150 ms",
    "touchpad":   "Touchpad",
    "pose":       "Pose",
}

# ── colours consistent with order above ──────────────────────────────────────
TECH_COLOR = {t: c for t, c in zip(
    TECHNIQUE_ORDER, ["#0A84FF", "#30D158", "#FF453A", "#FFD60A"]
)}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_fitts(data_dir):
    paths = glob.glob(os.path.join(data_dir, "fitts_*.csv"))
    if not paths:
        return pd.DataFrame()
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:
            print(f"  skipping {os.path.basename(p)}: {e}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["success"] == 1].copy()          # keep hits only for MT analysis
    df = df[df["mt_ms"] > 50].copy()            # drop sub-50 ms artefacts
    df["ID"] = np.log2(df["target_d"] / df["target_w"] + 1)  # Fitts' ID
    # normalise technique name: files written with "mouse" but label may differ
    df["technique"] = df["technique"].str.strip()

    if "latency_ms" in df.columns:
        df.loc[(df["technique"] == "mouse") & (df["latency_ms"] == 150), "technique"] = "mouse+150ms"

    return df


def load_steering(data_dir):
    paths = glob.glob(os.path.join(data_dir, "steering_*.csv"))
    if not paths:
        return pd.DataFrame()
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:
            print(f"  skipping {os.path.basename(p)}: {e}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["duration_ms"] > 50].copy()
    df["technique"] = df["technique"].str.strip()

    if "latency_ms" in df.columns:
        df.loc[(df["technique"] == "mouse") & (df["latency_ms"] == 150), "technique"] = "mouse+150ms"
    
    return df


fitts   = load_fitts(DATA_DIR)
steering = load_steering(DATA_DIR)

print(f"Fitts rows loaded   : {len(fitts)}")
print(f"Steering rows loaded: {len(steering)}")

if fitts.empty and steering.empty:
    print("No data found in data/ — place your CSV files there and re-run.")
    raise SystemExit(1)

# helper: ordered list of techniques actually present
def present_techs(df):
    return [t for t in TECHNIQUE_ORDER if t in df["technique"].unique()]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fitts — mean MT per technique  (bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def plot_fitts_mt_bar(df):
    techs = present_techs(df)
    means = [df[df["technique"] == t]["mt_ms"].mean() for t in techs]
    sems  = [df[df["technique"] == t]["mt_ms"].sem()  for t in techs]
    labels = [TECHNIQUE_LABELS.get(t, t) for t in techs]
    colors = [TECH_COLOR[t] for t in techs]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, means, yerr=sems, capsize=5,
                  color=colors, width=0.55, error_kw={"elinewidth": 1.5})
    ax.set_ylabel("Mean Movement Time (ms)")
    ax.set_title("Fitts' Law — Mean MT per Input Technique")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f"{m:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "fitts_mt_bar.png"))
    plt.close(fig)
    print("  saved fitts_mt_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fitts — regression lines  MT vs ID  per technique
# ─────────────────────────────────────────────────────────────────────────────

def plot_fitts_regression(df):
    techs = present_techs(df)
    fig, ax = plt.subplots(figsize=(7, 5))

    for tech in techs:
        sub = df[df["technique"] == tech]
        # mean MT per (W, D) cell — one point per condition
        cell = sub.groupby(["target_w", "target_d"])["mt_ms"].mean().reset_index()
        cell["ID"] = np.log2(cell["target_d"] / cell["target_w"] + 1)
        x, y = cell["ID"].values, cell["mt_ms"].values
        slope, intercept, r, *_ = stats.linregress(x, y)
        xfit = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
        label = f"{TECHNIQUE_LABELS.get(tech, tech)}  (r²={r**2:.2f})"
        color = TECH_COLOR[tech]
        ax.scatter(x, y, color=color, zorder=3, s=40)
        ax.plot(xfit, intercept + slope * xfit, color=color, label=label)

    ax.set_xlabel("Index of Difficulty  ID = log₂(D/W + 1)  (bits)")
    ax.set_ylabel("Mean Movement Time (ms)")
    ax.set_title("Fitts' Law — MT vs. ID Regression by Technique")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "fitts_regression.png"))
    plt.close(fig)
    print("  saved fitts_regression.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fitts — MT vs W  and  MT vs D  (one panel each)
# ─────────────────────────────────────────────────────────────────────────────

def plot_fitts_param_sweep(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    techs = present_techs(df)

    for tech in techs:
        sub = df[df["technique"] == tech]
        color = TECH_COLOR[tech]
        label = TECHNIQUE_LABELS.get(tech, tech)

        # Width sweep (D fixed at 400)
        w_sub = sub[sub["target_d"] == 400].groupby("target_w")["mt_ms"].mean()
        if not w_sub.empty:
            axes[0].plot(w_sub.index, w_sub.values, marker="o", color=color, label=label)

        # Distance sweep (W fixed at 60)
        d_sub = sub[sub["target_w"] == 60].groupby("target_d")["mt_ms"].mean()
        if not d_sub.empty:
            axes[1].plot(d_sub.index, d_sub.values, marker="o", color=color, label=label)

    axes[0].set_xlabel("Target Width W (px)")
    axes[0].set_ylabel("Mean MT (ms)")
    axes[0].set_title("MT vs. Target Width  (D = 400 px)")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Target Distance D (px)")
    axes[1].set_ylabel("Mean MT (ms)")
    axes[1].set_title("MT vs. Target Distance  (W = 60 px)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "fitts_param_sweep.png"))
    plt.close(fig)
    print("  saved fitts_param_sweep.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Fitts — error rate per technique  (bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def plot_fitts_error(data_dir):
    """Reload raw (including misses) to compute error rate."""
    paths = glob.glob(os.path.join(data_dir, "fitts_*.csv"))
    if not paths:
        return
    frames = [pd.read_csv(p) for p in paths]
    raw = pd.concat(frames, ignore_index=True)
    raw["technique"] = raw["technique"].str.strip()

    if "latency_ms" in raw.columns:
        raw.loc[(raw["technique"] == "mouse") & (raw["latency_ms"] == 150), "technique"] = "mouse+150ms"

    raw = raw[raw["mt_ms"] > 50]

    techs = [t for t in TECHNIQUE_ORDER if t in raw["technique"].unique()]
    rates, labels, colors = [], [], []
    for t in techs:
        sub = raw[raw["technique"] == t]
        rates.append((1 - sub["success"].mean()) * 100)
        labels.append(TECHNIQUE_LABELS.get(t, t))
        colors.append(TECH_COLOR[t])

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, rates, color=colors, width=0.55)
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Fitts' Law — Click Error Rate per Technique")
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{r:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "fitts_error_rate.png"))
    plt.close(fig)
    print("  saved fitts_error_rate.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Steering — mean duration per technique  (bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def plot_steering_duration_bar(df):
    techs = present_techs(df)
    means  = [df[df["technique"] == t]["duration_ms"].mean() for t in techs]
    sems   = [df[df["technique"] == t]["duration_ms"].sem()  for t in techs]
    labels = [TECHNIQUE_LABELS.get(t, t) for t in techs]
    colors = [TECH_COLOR[t] for t in techs]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, means, yerr=sems, capsize=5,
                  color=colors, width=0.55, error_kw={"elinewidth": 1.5})
    ax.set_ylabel("Mean Traversal Time (ms)")
    ax.set_title("Steering Law — Mean Duration per Input Technique")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f"{m:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "steering_duration_bar.png"))
    plt.close(fig)
    print("  saved steering_duration_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Steering — violations per technique  (bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def plot_steering_violations_bar(df):
    techs  = present_techs(df)
    means  = [df[df["technique"] == t]["violations"].mean() for t in techs]
    sems   = [df[df["technique"] == t]["violations"].sem()  for t in techs]
    labels = [TECHNIQUE_LABELS.get(t, t) for t in techs]
    colors = [TECH_COLOR[t] for t in techs]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, means, yerr=sems, capsize=5,
                  color=colors, width=0.55, error_kw={"elinewidth": 1.5})
    ax.set_ylabel("Mean Wall Violations per Traversal")
    ax.set_title("Steering Law — Wall Violations per Technique")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "steering_violations_bar.png"))
    plt.close(fig)
    print("  saved steering_violations_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Steering — duration vs W  and  duration vs A  sweeps
# ─────────────────────────────────────────────────────────────────────────────

def plot_steering_param_sweep(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    techs = present_techs(df)

    for tech in techs:
        sub = df[df["technique"] == tech]
        color = TECH_COLOR[tech]
        label = TECHNIQUE_LABELS.get(tech, tech)

        # Width sweep (A fixed at 500)
        w_sub = sub[sub["tunnel_a"] == 500].groupby("tunnel_w")["duration_ms"].mean()
        if not w_sub.empty:
            axes[0].plot(w_sub.index, w_sub.values, marker="o", color=color, label=label)

        # Amplitude sweep (W fixed at 100)
        a_sub = sub[sub["tunnel_w"] == 100].groupby("tunnel_a")["duration_ms"].mean()
        if not a_sub.empty:
            axes[1].plot(a_sub.index, a_sub.values, marker="o", color=color, label=label)

    axes[0].set_xlabel("Tunnel Width W (px)")
    axes[0].set_ylabel("Mean Duration (ms)")
    axes[0].set_title("Steering — Duration vs. Width  (A = 500 px)")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Tunnel Amplitude A (px)")
    axes[1].set_ylabel("Mean Duration (ms)")
    axes[1].set_title("Steering — Duration vs. Amplitude  (W = 100 px)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "steering_param_sweep.png"))
    plt.close(fig)
    print("  saved steering_param_sweep.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Latency effect — mouse vs mouse+150ms on both tasks
# ─────────────────────────────────────────────────────────────────────────────

def plot_latency_effect(fitts_df, steering_df):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, df, metric, title, ylabel in [
        (axes[0], fitts_df,   "mt_ms",       "Fitts' Law",   "Mean MT (ms)"),
        (axes[1], steering_df,"duration_ms",  "Steering Law", "Mean Duration (ms)"),
    ]:
        if df.empty:
            continue
        compare = ["mouse", "mouse+150ms"]
        present = [t for t in compare if t in df["technique"].unique()]
        means  = [df[df["technique"] == t][metric].mean() for t in present]
        sems   = [df[df["technique"] == t][metric].sem()  for t in present]
        labels = [TECHNIQUE_LABELS.get(t, t) for t in present]
        colors = [TECH_COLOR[t] for t in present]
        bars = ax.bar(labels, means, yerr=sems, capsize=5,
                      color=colors, width=0.45, error_kw={"elinewidth": 1.5})
        ax.set_title(f"{title} — Latency Effect")
        ax.set_ylabel(ylabel)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    f"{m:.0f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "latency_effect.png"))
    plt.close(fig)
    print("  saved latency_effect.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Print summary statistics to console
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(fitts_df, steering_df):
    print("\n" + "═" * 60)
    print("FITTS' LAW — Mean MT (ms) and Error Rate by Technique")
    print("═" * 60)
    if not fitts_df.empty:
        paths = glob.glob(os.path.join(DATA_DIR, "fitts_*.csv"))
        raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        raw["technique"] = raw["technique"].str.strip()
        if "latency_ms" in raw.columns:
            raw.loc[(raw["technique"] == "mouse") & (raw["latency_ms"] == 150), "technique"] = "mouse+150ms"
        raw = raw[raw["mt_ms"] > 50]
        for t in present_techs(fitts_df):
            sub_hits = fitts_df[fitts_df["technique"] == t]["mt_ms"]
            sub_all  = raw[raw["technique"] == t]
            err = (1 - sub_all["success"].mean()) * 100
            slope, intercept, r, *_ = stats.linregress(
                fitts_df[fitts_df["technique"] == t]["ID"],
                fitts_df[fitts_df["technique"] == t]["mt_ms"]
            )
            print(f"  {TECHNIQUE_LABELS.get(t,t):18s}  "
                  f"MT={sub_hits.mean():.0f}±{sub_hits.std():.0f} ms  "
                  f"err={err:.1f}%  "
                  f"b={slope:.1f} ms/bit  r²={r**2:.3f}")

    print("\n" + "═" * 60)
    print("STEERING LAW — Mean Duration (ms) and Violations by Technique")
    print("═" * 60)
    if not steering_df.empty:
        for t in present_techs(steering_df):
            sub = steering_df[steering_df["technique"] == t]
            print(f"  {TECHNIQUE_LABELS.get(t,t):18s}  "
                  f"duration={sub['duration_ms'].mean():.0f}±{sub['duration_ms'].std():.0f} ms  "
                  f"violations={sub['violations'].mean():.2f}±{sub['violations'].std():.2f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Run everything
# ─────────────────────────────────────────────────────────────────────────────

print("\nGenerating plots → assets/")

if not fitts.empty:
    plot_fitts_mt_bar(fitts)
    plot_fitts_regression(fitts)
    plot_fitts_param_sweep(fitts)
    plot_fitts_error(DATA_DIR)

if not steering.empty:
    plot_steering_duration_bar(steering)
    plot_steering_violations_bar(steering)
    plot_steering_param_sweep(steering)

if not fitts.empty or not steering.empty:
    plot_latency_effect(fitts, steering)

print_summary(fitts, steering)
print(f"All plots saved to  {RESULT_DIR}/")