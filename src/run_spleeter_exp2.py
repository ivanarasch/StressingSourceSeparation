"""Run Spleeter (4 stems) on the Experiment 2 mixes, with a FRESH Separator for every file.

    python src/run_spleeter_exp2.py [--workers 3] [--limit N] [--only TEXT] [--gpu]

Why a fresh Separator per file: reusing one Separator across files makes Spleeter's output depend on the file it
processed just before (the same mix scored 3.4 dB or 9.8 dB vocal SDR depending on what ran first). Each file
therefore gets its own Separator inside a worker process that is restarted after every file, which also stops
TensorFlow memory from building up. Runs on CPU by default: with the Apple Metal GPU each worker grew to about
50 GB and crashed the machine.

Run it in the Spleeter environment (Python 3.8, spleeter 2.3.2, tensorflow 2.13). Finished files are skipped.
Writes <SPLEETER_DIR>/check.csv (correlation of the summed stems with the mix, per file) and failures.txt.
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MIX_DIR, SPLEETER_DIR, STEMS


def outputs_ok(out_dir, n_frames):
    import soundfile as sf
    for s in STEMS:
        p = out_dir / f"{s}.wav"
        if not p.exists() or sf.info(p).frames != n_frames:
            return False
    return True


def work(job):
    wav_path, out_root, use_gpu = job
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import logging
    logging.disable(logging.CRITICAL)
    if not use_gpu:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
    import numpy as np
    import soundfile as sf

    wav_path = Path(wav_path)
    out_dir = Path(out_root) / wav_path.parent.name / wav_path.stem
    n_frames = sf.info(wav_path).frames
    if outputs_ok(out_dir, n_frames):
        return wav_path.stem, "skipped", None

    from spleeter.separator import Separator
    mix, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    msg = ""
    for _ in range(3):
        try:
            sep = Separator("spleeter:4stems", multiprocess=False)
            pred = sep.separate(mix)
            del sep
            if any(len(pred[s]) != len(mix) for s in STEMS):
                msg = "length mismatch"
                continue
            total = sum(pred[s] for s in STEMS)
            corr = float(np.corrcoef(total.mean(1)[::50], mix.mean(1)[::50])[0, 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            for s in STEMS:
                sf.write(out_dir / f"{s}.wav", np.clip(pred[s], -1, 1), sr)
            if not outputs_ok(out_dir, n_frames):
                msg = "written files failed the length check"
                continue
            return wav_path.stem, "ok", round(corr, 4)
        except Exception as e:
            msg = f"error: {str(e)[:200]}"
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except Exception:
                pass
    return wav_path.stem, "FAILED " + msg, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--gpu", action="store_true", help="use the GPU (not recommended, see the docstring)")
    a = ap.parse_args()

    SPLEETER_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(MIX_DIR.glob("*/*.wav"))
    if a.only:
        files = [f for f in files if a.only in str(f)]
    if a.limit:
        files = files[: a.limit]
    jobs = [(str(f), str(SPLEETER_DIR), a.gpu) for f in files]
    print(f"{len(jobs)} files, {a.workers} workers", flush=True)

    check_path = SPLEETER_DIR / "check.csv"
    new = not check_path.exists()
    chk = open(check_path, "a", newline="")
    w = csv.writer(chk)
    if new:
        w.writerow(["file", "status", "stems_sum_corr"])
    failures, t0, done = [], time.time(), 0
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        for k, (name, status, corr) in enumerate(pool.imap_unordered(work, jobs, chunksize=1), 1):
            w.writerow([name, status, corr])
            chk.flush()
            if status.startswith("FAILED"):
                failures.append(f"{name}\t{status}")
            elif status == "ok":
                done += 1
            if k % 10 == 0 or status.startswith("FAILED"):
                print(f"[{k}/{len(jobs)}] {name} {status} {corr}  {(time.time() - t0) / 60:.1f} min", flush=True)
    chk.close()
    (SPLEETER_DIR / "failures.txt").write_text("\n".join(failures))
    print(f"done: {done} separated, {len(failures)} failed, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
