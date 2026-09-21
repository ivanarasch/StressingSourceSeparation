"""Score Experiment 1: how well is the vocal recovered from each paired mixture?

    python src/evaluate_exp1.py [--workers 6]

For every song and pairing (bass / drums / other / accompaniment + vocals) and each model, two things are scored
against the true MUSDB18 stems (both signals downmixed to mono first):
  component "vocals"     the separated vocal vs. the true vocal stem            <- the scores in the write-up
  component "non_vocal"  the separated rest vs. the true non-vocal stem (bass, drums, other, or drums+bass+other)
Three metrics are computed,
with the definitions from the original first-pass script (evaluate_si_metrics.py):
  SI_SDR       scale-invariant SDR of the estimate against the true vocal
  SI_SAR_like  10 log10(||est||^2 / ||est - proj(est onto vocal)||^2)   (a fast artifact ratio, not BSS-Eval SAR)
  RMSE         root-mean-square error between the two signals (depends on level)
The summary reports the MEDIAN over the 50 songs, which reproduces the numbers in the original write-up exactly.

Reads : <EXP1_SPLEETER_DIR>/<song>/<pair>_vocals/{vocals,accompaniment}.wav
        <EXP1_DEMUCS_DIR>/<song>_<pair>/{vocals,no_vocals}.wav
Writes: <EXP1_RESULTS_DIR>/metrics.csv (per song), summary_median.csv, fig_exp1_median_si_sdr.{png,pdf} (vocal component)
"""
import argparse
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]

import numpy as np
import pandas as pd
import soundfile as sf

from config import EXP1_DEMUCS_DIR, EXP1_RESULTS_DIR, EXP1_SPLEETER_DIR, MUSDB_ROOT

PAIRS = ["bass", "drums", "other", "accompaniment"]


def load_mono(path):
    audio, _ = sf.read(path)
    return (audio.mean(axis=1) if audio.ndim > 1 else audio).astype(np.float64)


def si_sdr(ref, est, eps=1e-8):
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    return 10 * np.log10((np.sum(target ** 2) + eps) / (np.sum((est - target) ** 2) + eps))


def si_sar_like(ref, est, eps=1e-8):
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    artifacts = est - alpha * ref
    return 10 * np.log10((np.sum(est ** 2) + eps) / (np.sum(artifacts ** 2) + eps))


def rmse(a, b):
    return math.sqrt(float(np.mean((a - b) ** 2)))


def score_song(name):
    import musdb
    track = next(t for t in musdb.DB(root=str(MUSDB_ROOT), subsets="test").tracks if t.name == name)
    stem = {k: np.asarray(track.targets[k].audio, dtype=np.float64) for k in ["vocals", "drums", "bass", "other"]}
    truth = {"vocals": stem["vocals"].mean(axis=1),
             "bass": stem["bass"].mean(axis=1),
             "drums": stem["drums"].mean(axis=1),
             "other": stem["other"].mean(axis=1),
             "accompaniment": (stem["drums"] + stem["bass"] + stem["other"]).mean(axis=1)}
    rows = []
    for pair in PAIRS:
        for model, d, files in [("spleeter", EXP1_SPLEETER_DIR / name / f"{pair}_vocals", ("vocals.wav", "accompaniment.wav")),
                                ("demucs", EXP1_DEMUCS_DIR / f"{name}_{pair}", ("vocals.wav", "no_vocals.wav"))]:
            for component, fname, ref_full in [("vocals", files[0], truth["vocals"]), ("non_vocal", files[1], truth[pair])]:
                path = d / fname
                if not path.exists():
                    continue
                est = load_mono(path)
                n = min(len(ref_full), len(est))
                ref, est = ref_full[:n], est[:n]
                rows.append(dict(model=model, song=name, pair=pair, component=component, SI_SDR=si_sdr(ref, est),
                                 SI_SAR_like=si_sar_like(ref, est), RMSE=rmse(ref, est)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    import musdb
    names = [t.name for t in musdb.DB(root=str(MUSDB_ROOT), subsets="test").tracks]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(score_song, names) for r in part]
    df = pd.DataFrame(rows).sort_values(["model", "pair", "component", "song"])
    EXP1_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXP1_RESULTS_DIR / "metrics.csv", index=False)

    summ = df.groupby(["component", "pair", "model"])[["SI_SDR", "SI_SAR_like", "RMSE"]].median()
    summ["n_songs"] = df.groupby(["component", "pair", "model"]).size()
    summ = summ.reindex(PAIRS, level=1).round(5)
    summ.to_csv(EXP1_RESULTS_DIR / "summary_median.csv")
    print(summ.round(3).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(PAIRS))
    for k, (m, col, lab) in enumerate([("demucs", "#0072B2", "Demucs"), ("spleeter", "#E69F00", "Spleeter")]):
        g = df[(df.model == m) & (df.component == "vocals")].groupby("pair").SI_SDR
        med, q1, q3 = g.median().reindex(PAIRS), g.quantile(0.25).reindex(PAIRS), g.quantile(0.75).reindex(PAIRS)
        ax.bar(x + (k - 0.5) * 0.38, med, 0.38, color=col, label=lab, yerr=[med - q1, q3 - med],
               error_kw={"lw": 1, "capsize": 2, "ecolor": "0.35"})
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in PAIRS])
    ax.set_ylabel("Vocal SI-SDR (dB)\nmedian, bars = interquartile range")
    ax.set_xlabel("Non-vocal stem mixed with the vocal")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(EXP1_RESULTS_DIR / "fig_exp1_median_si_sdr.png", dpi=300)
    plt.savefig(EXP1_RESULTS_DIR / "fig_exp1_median_si_sdr.pdf")
    print("->", EXP1_RESULTS_DIR)


if __name__ == "__main__":
    main()
