"""Spleeter 2-stem on the Experiment 1 pairs, with a FRESH Separator for every file.

    ~/spleeter_env/bin/python src/run_spleeter_exp1.py <paired_stems_dir> <output_dir> [--workers 3] [--limit N]

Output: <output_dir>/<song>/<pair name>/{vocals,accompaniment}.wav   (e.g. <song>/bass_vocals/vocals.wav)

Why a fresh Separator per file: when one Separator is reused across files, Spleeter 2.3.2 (TensorFlow 2.13, Python 3.8)
can hand back the result of the file it processed just before, so every output is shifted by one file and scores come
out wrong (for example, the bass_vocals folder holds the separation of drums_vocals). Each file therefore gets its own
Separator inside a worker process that is restarted after every file. Runs on CPU. Finished files are skipped.
"""
import argparse
import os
import time
import warnings
from multiprocessing import get_context
from pathlib import Path

NAMES = ["vocals", "accompaniment"]


def outputs_ok(out_dir, n_frames):
    import soundfile as sf
    for name in NAMES:
        p = out_dir / f"{name}.wav"
        if not p.exists() or sf.info(p).frames != n_frames:
            return False
    return True


def work(job):
    wav_path, out_root = job
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import logging
    logging.disable(logging.CRITICAL)
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import numpy as np
    import soundfile as sf

    wav_path = Path(wav_path)
    out_dir = Path(out_root) / wav_path.parent.name / wav_path.stem
    n_frames = sf.info(wav_path).frames
    if outputs_ok(out_dir, n_frames):
        return wav_path, "skipped"

    from spleeter.separator import Separator
    mix, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    msg = ""
    for _ in range(3):
        try:
            sep = Separator("spleeter:2stems", multiprocess=False)
            pred = sep.separate(mix)
            del sep
            if any(len(pred[n]) != len(mix) for n in NAMES):
                msg = "length mismatch"
                continue
            if np.corrcoef(pred["vocals"].mean(1)[::50], pred["accompaniment"].mean(1)[::50])[0, 1] > 0.98:
                msg = "stems are nearly identical: the 2-stem model probably did not load (check pretrained_models/2stems)"
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            for n in NAMES:
                sf.write(out_dir / f"{n}.wav", np.clip(pred[n], -1, 1), sr)
            if not outputs_ok(out_dir, n_frames):
                msg = "written files failed the length check"
                continue
            return wav_path, "ok"
        except Exception as e:
            msg = f"error: {str(e)[:200]}"
            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass
    return wav_path, "FAILED " + msg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_home", help="folder with <song>/<pair>_vocals.wav")
    ap.add_argument("output_home", help="where <song>/<pair>_vocals/{vocals,accompaniment}.wav is written")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="only the first N files (for a quick test)")
    a = ap.parse_args()

    os.makedirs(a.output_home, exist_ok=True)
    files = sorted(Path(a.data_home).glob("*/*.wav"))
    if a.limit:
        files = files[: a.limit]
    print(f"Found {len(files)} files, {a.workers} workers.", flush=True)

    failures, t0 = [], time.time()
    with get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        jobs = [(str(f), a.output_home) for f in files]
        for k, (path, status) in enumerate(pool.imap_unordered(work, jobs, chunksize=1), 1):
            if status.startswith("FAILED"):
                failures.append(f"{path}\t{status}")
                print(f"   {status}: {path}", flush=True)
            if k % 10 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}] {path.parent.name} / {path.stem}  {(time.time() - t0) / 60:.1f} min", flush=True)
    Path(a.output_home, "failures.txt").write_text("\n".join(failures))
    print(f"done: {len(files) - len(failures)} ok, {len(failures)} failed, {(time.time() - t0) / 60:.1f} min", flush=True)


# Spleeter workers re-import this file on macOS, so everything that runs must sit behind this guard.
if __name__ == "__main__":
    main()
