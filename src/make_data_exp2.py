"""Experiment 2 data: loudness-matched vocal-effect mixes and reference stems.

    python src/make_data_exp2.py [--num N | --only NAME ...] [--workers 8]

For every MUSDB18 test song and every condition (no effect + 4 effects x 4 levels) it writes
    <MIX_DIR>/<song>/<song>_<cond>.wav                        mix = drums + bass + other + processed vocal
    <REF_DIR>/<song>/<cond>/{drums,bass,other,vocals}.wav     the exact stems that are in the mix
    <RESULTS_DIR>/loudness_log.csv                            every gain that was applied

Loudness control: the processed vocal is scaled so its RMS equals the original vocal's RMS. If the mix would
exceed PEAK_LIMIT, the whole mix (and the matching reference stems) is scaled by one common factor, which keeps
the vocal-to-accompaniment balance. Raw drums/bass/other are stored once per song (<song>/_raw) and hard-linked
into each condition folder unless a condition needed scaling. Finished conditions are skipped, so it is safe to re-run.
"""
import argparse
import csv
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import soundfile as sf

import effects
from config import MIX_DIR, MUSDB_ROOT, REF_DIR, RESULTS_DIR

LEVELS_TO_KEEP = {1, 2, 3, 4}
PEAK_LIMIT = 0.999
FIELDS = ["song", "condition", "vocal_rms_db_raw", "vocal_rms_db_after_effect",
          "rms_match_gain_db", "mix_scale_db", "mix_peak"]
_DB = None


def match_length(a, b):
    if len(a) > len(b):
        return a[:len(b)]
    if len(a) < len(b):
        return np.vstack([a, np.zeros((len(b) - len(a), a.shape[1]), dtype=a.dtype)])
    return a


def apply_effect(vocals, rate, func, params, takes_numpy):
    if takes_numpy:
        fx = func(vocals, rate=rate, **params)
    else:
        fx = effects.audiosegment_to_numpy(func(effects.audiosegment_from_numpy(vocals, rate), **params))
    if fx.ndim == 1:
        fx = np.stack([fx, fx], axis=1)
    return fx


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-12)


def db(x):
    return 20 * np.log10(x)


def link_or_copy(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def process_song(name):
    """Generate every condition for one song. Returns (song, log rows)."""
    global _DB
    import musdb
    if _DB is None:
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass
        _DB = musdb.DB(root=str(MUSDB_ROOT), subsets="test")
    track = next(t for t in _DB.tracks if t.name == name)

    song = track.name.replace(" ", "_")
    rate = track.rate
    stems = {s: np.asarray(track.targets[s].audio, dtype=np.float32) for s in ["vocals", "drums", "bass", "other"]}
    vocals = stems["vocals"]
    accomp = stems["drums"] + stems["bass"] + stems["other"]
    raw_rms = rms(vocals)

    raw_dir = REF_DIR / song / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for s in stems:
        if not (raw_dir / f"{s}.wav").exists():
            sf.write(raw_dir / f"{s}.wav", stems[s], rate)

    conditions = [("original", None, None, None)]
    for ename, func, levels, takes_numpy in effects.EFFECTS:
        for i, params in enumerate(levels, 1):
            if i in LEVELS_TO_KEEP:
                conditions.append((f"{ename}_l{i}", func, params, takes_numpy))

    rows = []
    for cond, func, params, takes_numpy in conditions:
        mix_path = MIX_DIR / song / f"{song}_{cond}.wav"
        cond_dir = REF_DIR / song / cond
        if mix_path.exists() and (cond_dir / "vocals.wav").exists():
            continue
        cond_dir.mkdir(parents=True, exist_ok=True)
        mix_path.parent.mkdir(parents=True, exist_ok=True)

        if cond == "original":
            fx = vocals
        else:
            fx = match_length(apply_effect(vocals, rate, func, params, takes_numpy), vocals).astype(np.float32)

        fx_rms = rms(fx)
        gain = raw_rms / fx_rms
        mix = accomp + fx * gain
        peak = float(np.max(np.abs(mix)))
        scale = PEAK_LIMIT / peak if peak > PEAK_LIMIT else 1.0

        sf.write(cond_dir / "vocals.wav", (fx * gain * scale).astype(np.float32), rate)
        for s in ["drums", "bass", "other"]:
            dst = cond_dir / f"{s}.wav"
            if scale == 1.0:
                link_or_copy(raw_dir / f"{s}.wav", dst)
            else:
                if dst.exists():
                    dst.unlink()
                sf.write(dst, (stems[s] * scale).astype(np.float32), rate)
        sf.write(mix_path, (mix * scale).astype(np.float32), rate)   # written last: marks the condition as done

        rows.append(dict(song=song, condition=cond, vocal_rms_db_raw=round(db(raw_rms), 2),
                         vocal_rms_db_after_effect=round(db(fx_rms), 2),
                         rms_match_gain_db=round(db(gain), 2), mix_scale_db=round(db(scale), 3),
                         mix_peak=round(peak * scale, 4)))
    return song, rows


def main(num="all", only=None, workers=1):
    import musdb
    MIX_DIR.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log_path = RESULTS_DIR / "loudness_log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="")
    log = csv.DictWriter(log_f, fieldnames=FIELDS)
    if new_log:
        log.writeheader()

    names = [t.name for t in musdb.DB(root=str(MUSDB_ROOT), subsets="test").tracks]
    if num != "all":
        names = names[: int(num)]
    if only:
        names = [n for n in names if any(n.startswith(o) for o in only)]

    def record(result, k):
        song, rows = result
        for r in rows:
            log.writerow(r)
        log_f.flush()
        print(f"[{k}/{len(names)}] {song}: {len(rows)} conditions written", flush=True)

    try:
        if workers <= 1:
            for k, n in enumerate(names, 1):
                record(process_song(n), k)
        else:
            with mp.get_context("spawn").Pool(workers) as pool:
                for k, result in enumerate(pool.imap_unordered(process_song, names), 1):
                    record(result, k)
    finally:
        log_f.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num", default="all", help="only the first N songs")
    ap.add_argument("--only", nargs="*", default=None, help="only songs whose name starts with one of these")
    ap.add_argument("--workers", type=int, default=1, help="parallel worker processes (about 8-10 is fast)")
    a = ap.parse_args()
    main(a.num, a.only, a.workers)
