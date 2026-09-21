"""Score the separated stems for Experiment 2.

    python src/evaluate_exp2.py [--workers 6] [--num N] [--dry-reference]

For every song, condition and stem it computes time-domain SDR and SI-SDR of the model's estimate against the
reference stem (the exact stem that went into the mix, with the vocal in its *processed* form), then
delta_sdr = SDR minus SDR of the same song/stem/model in the no-effect ("original") condition.

Writes <RESULTS_DIR>/metrics.csv. With --dry-reference it also scores the vocal estimate against the original,
unprocessed vocal (scaled the same way as the mix) and writes <RESULTS_DIR>/dryref_vocals.csv.

SDR = 10 log10( sum(ref^2) / sum((ref - est)^2) ), over the full track, both channels together.
This is plain SDR, not the BSS-Eval variant from `museval`.
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import soundfile as sf

from config import MODELS, REF_DIR, RESULTS_DIR, STEMS

SILENT = 1e-6


def read(path):
    return sf.read(path, always_2d=True)[0]


def sdr(ref, est, eps=1e-9):
    return 10 * np.log10((np.sum(ref ** 2) + eps) / (np.sum((ref - est) ** 2) + eps))


def si_sdr(ref, est, eps=1e-9):
    alpha = np.sum(ref * est) / (np.sum(ref ** 2) + eps)
    target = alpha * ref
    return 10 * np.log10((np.sum(target ** 2) + eps) / (np.sum((est - target) ** 2) + eps))


def parse_condition(cond):
    if cond == "original":
        return "original", 0
    effect, level = cond.rsplit("_l", 1)
    return effect, int(level)


def score_song(job):
    song, dry, scales = job
    rows, dry_rows = [], []
    raw_vocals = read(REF_DIR / song / "_raw" / "vocals.wav") if dry else None
    for model, root in MODELS.items():
        song_dir = root / song
        if not song_dir.is_dir():
            continue
        for cd in sorted(p for p in song_dir.iterdir() if p.is_dir()):
            cond = cd.name[len(song) + 1:]
            ref_dir = REF_DIR / song / cond
            if not ref_dir.is_dir() or not all((cd / f"{s}.wav").exists() for s in STEMS):
                continue
            effect, level = parse_condition(cond)
            for s in STEMS:
                ref, est = read(ref_dir / f"{s}.wav"), read(cd / f"{s}.wav")
                n = min(len(ref), len(est))
                ref, est = ref[:n], est[:n]
                if np.mean(ref ** 2) < SILENT:
                    continue
                rows.append(dict(model=model, song=song, condition=cond, effect=effect, level=level,
                                 stem=s, sdr=sdr(ref, est), si_sdr=si_sdr(ref, est)))
                if dry and s == "vocals":
                    dref = raw_vocals[:n] * 10 ** (scales.get((song, cond), 0.0) / 20)
                    dry_rows.append(dict(model=model, song=song, condition=cond, effect=effect, level=level,
                                         sdr_wet=sdr(ref, est), sdr_dry=sdr(dref, est)))
    return rows, dry_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--num", type=int, default=None, help="only the first N songs")
    ap.add_argument("--dry-reference", action="store_true")
    a = ap.parse_args()

    songs = sorted(p.name for p in REF_DIR.iterdir() if p.is_dir())
    if a.num:
        songs = songs[: a.num]
    scales = {}
    if a.dry_reference:
        log = pd.read_csv(RESULTS_DIR / "loudness_log.csv")
        scales = {(r.song, r.condition): r.mix_scale_db for r in log.itertuples()}
    jobs = [(s, a.dry_reference, scales) for s in songs]

    rows, dry_rows = [], []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for k, (r, d) in enumerate(pool.imap_unordered(score_song, jobs), 1):
            rows += r
            dry_rows += d
            if k % 10 == 0:
                print(f"scored {k}/{len(songs)} songs", flush=True)

    df = pd.DataFrame(rows)
    base = (df[df.effect == "original"].set_index(["model", "song", "stem"])[["sdr", "si_sdr"]]
            .rename(columns={"sdr": "base_sdr", "si_sdr": "base_si_sdr"}))
    df = df.join(base, on=["model", "song", "stem"])
    df["delta_sdr"] = df.sdr - df.base_sdr
    df["delta_si_sdr"] = df.si_sdr - df.base_si_sdr
    df = df.drop(columns=["base_sdr", "base_si_sdr"]).sort_values(["model", "song", "condition", "stem"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print(f"{len(df)} rows -> {RESULTS_DIR / 'metrics.csv'}")
    if dry_rows:
        pd.DataFrame(dry_rows).sort_values(["model", "song", "condition"]).to_csv(RESULTS_DIR / "dryref_vocals.csv", index=False)
        print(f"{len(dry_rows)} rows -> {RESULTS_DIR / 'dryref_vocals.csv'}")


if __name__ == "__main__":
    main()
