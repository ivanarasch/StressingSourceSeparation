"""Paper figures and summary tables for Experiment 2.

    python src/make_figures_exp2.py

Reads <RESULTS_DIR>/metrics.csv (and dryref_vocals.csv if present) and writes, under <RESULTS_DIR>:
    figures/fig1_vocal_delta_sdr_by_level.{png,pdf}          vocal delta-SDR vs. effect level, 95% bootstrap CI
    figures/fig2_stems_by_effect_level4.{png,pdf}            median delta-SDR for all four stems at level 4
    figures/fig3_processed_vs_dry_reference_level4.{png,pdf} vocal SDR vs. processed and vs. original dry vocal
    tables/*.csv                                              the numbers behind them
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import RESULTS_DIR

FIG, TAB = RESULTS_DIR / "figures", RESULTS_DIR / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
COL = {"reverb": "#D55E00", "delay": "#0072B2", "compression": "#009E73", "bitcrush": "#E69F00"}
MODELS = [("demucs", "Demucs (htdemucs)"), ("spleeter", "Spleeter (4-stem)")]
ORDER_E = ["reverb", "delay", "compression", "bitcrush"]
ORDER_S = ["vocals", "other", "bass", "drums"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def boot_ci(x, n=2000):
    x = np.asarray(x)
    idx = rng.integers(0, len(x), (n, len(x)))
    return np.percentile(np.median(x[idx], axis=1), [2.5, 97.5])


def save(name):
    plt.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close()


df = pd.read_csv(RESULTS_DIR / "metrics.csv")
eff = df[df.effect != "original"]
v = eff[eff.stem == "vocals"]

# tables
rows = []
for (m, e, l), g in v.groupby(["model", "effect", "level"]):
    x = g.delta_sdr.values
    lo, hi = boot_ci(x)
    rows.append(dict(model=m, effect=e, level=l, n_songs=len(x), median_dSDR=np.median(x), ci95_lo=lo, ci95_hi=hi,
                     mean_dSDR=x.mean(), pct_songs_worse_than_1dB=100 * np.mean(x < -1),
                     wilcoxon_p_less=wilcoxon(x, alternative="less").pvalue))
t1 = pd.DataFrame(rows)
t1.to_csv(TAB / "vocals_delta_sdr_summary.csv", index=False)
eff.groupby(["model", "stem", "effect"]).delta_sdr.median().unstack("effect").round(2).to_csv(TAB / "all_stems_median_dSDR_pooled_levels.csv")
t2b = eff[eff.level == 4].groupby(["model", "stem", "effect"]).delta_sdr.median().unstack("effect").round(2)
t2b.to_csv(TAB / "all_stems_median_dSDR_level4.csv")
df[df.effect == "original"].groupby(["model", "stem"])[["sdr", "si_sdr"]].agg(["mean", "median", "std"]).round(2).to_csv(TAB / "baseline_no_effect.csv")

# figure 1
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, (m, title) in zip(axes, MODELS):
    for e in ORDER_E:
        g = t1[(t1.model == m) & (t1.effect == e)].sort_values("level")
        ax.plot(g.level, g.median_dSDR, marker="o", color=COL[e], label=e.capitalize(), lw=1.8)
        ax.fill_between(g.level, g.ci95_lo, g.ci95_hi, color=COL[e], alpha=0.15, lw=0)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Effect level")
    ax.set_title(title, fontsize=10)
axes[0].set_ylabel("Vocal ΔSDR vs. no effect (dB)\nmedian over songs, 95% bootstrap CI")
axes[0].legend(frameon=False, loc="lower left", fontsize=9)
plt.tight_layout()
save("fig1_vocal_delta_sdr_by_level")

# figure 2
fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
for ax, (m, title) in zip(axes, MODELS):
    M = t2b.loc[m].reindex(ORDER_S)[ORDER_E]
    im = ax.imshow(M.values, cmap="RdBu", vmin=-3.5, vmax=3.5, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels([e.capitalize() for e in ORDER_E], rotation=20)
    ax.set_yticks(range(4))
    ax.set_yticklabels([s.capitalize() for s in ORDER_S])
    ax.set_title(title, fontsize=10)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.spines[:].set_visible(False)
fig.colorbar(im, ax=axes, shrink=0.8, label="Median ΔSDR (dB)")
save("fig2_stems_by_effect_level4")

# figure 3 (needs the dry-reference file)
dry_path = RESULTS_DIR / "dryref_vocals.csv"
if dry_path.exists():
    dry = pd.read_csv(dry_path)
    dry["gap"] = dry.sdr_dry - dry.sdr_wet
    dry.groupby(["model", "effect", "level"])[["sdr_wet", "sdr_dry", "gap"]].mean().round(2).to_csv(TAB / "vocals_sdr_vs_processed_and_dry_reference.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (m, title) in zip(axes, MODELS):
        g = dry[(dry.model == m) & (dry.level == 4)].groupby("effect")[["sdr_wet", "sdr_dry"]].mean().reindex(ORDER_E)
        x = np.arange(4)
        ax.bar(x - 0.19, g.sdr_wet, 0.38, color="#56B4E9", label="vs. processed vocal")
        ax.bar(x + 0.19, g.sdr_dry, 0.38, color="#CC79A7", label="vs. original (dry) vocal")
        base = dry[(dry.model == m) & (dry.effect == "original")].sdr_wet.mean()
        ax.axhline(base, color="0.4", ls="--", lw=0.9)
        ax.text(3.45, base + 0.15, "no-effect baseline", ha="right", fontsize=8, color="0.3")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([e.capitalize() for e in ORDER_E])
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Mean vocal SDR at level 4 (dB)")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    plt.tight_layout()
    save("fig3_processed_vs_dry_reference_level4")

print("figures ->", FIG, "\ntables  ->", TAB)
