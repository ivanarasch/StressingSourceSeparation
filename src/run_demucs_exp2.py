"""Run Demucs (htdemucs, 4 stems) on the Experiment 2 mixes.

    python src/run_demucs_exp2.py [--device mps|cuda|cpu] [--limit N] [--only TEXT]

Input : <MIX_DIR>/<song>/<song>_<cond>.wav
Output: <DEMUCS_DIR>/<song>/<song>_<cond>/{vocals,drums,bass,other}.wav

Every output is checked (length equals the input's; the four stems sum to something correlated with the mix).
Failures are retried once, then listed in failures.txt. Finished files are skipped, so it is safe to re-run.
Settings match the `demucs` command line: htdemucs, shifts=1 (random time shift, so runs differ slightly),
overlap=0.25, 16-bit WAV, clip mode "rescale". Run it in an environment with `demucs` installed.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import soundfile as sf
import torch
from demucs.api import Separator, save_audio

from config import DEMUCS_DIR, MIX_DIR, STEMS


def outputs_ok(out_dir, n_frames):
    for s in STEMS:
        p = out_dir / f"{s}.wav"
        if not p.exists() or sf.info(p).frames != n_frames:
            return False
    return True


def separate_one(sep, wav_path, out_dir):
    n_frames = sf.info(wav_path).frames
    origin, res = sep.separate_audio_file(wav_path)
    if origin.shape[-1] != n_frames:
        return False, f"length {origin.shape[-1]} != {n_frames}"
    total = sum(res[s] for s in STEMS)
    corr = float(np.corrcoef(total.mean(0).cpu().numpy()[::50], origin.mean(0).cpu().numpy()[::50])[0, 1])
    if not corr > 0.9:
        return False, f"stems sum correlation with mix is {corr:.3f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in STEMS:
        save_audio(res[s], str(out_dir / f"{s}.wav"), samplerate=sep.samplerate,
                   clip="rescale", bits_per_sample=16, as_float=False)
    if not outputs_ok(out_dir, n_frames):
        return False, "written files failed the length check"
    return True, f"corr {corr:.3f}"


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="only files whose path contains this text")
    a = ap.parse_args()

    DEMUCS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(MIX_DIR.glob("*/*.wav"))
    if a.only:
        files = [f for f in files if a.only in str(f)]
    if a.limit:
        files = files[: a.limit]
    print(f"{len(files)} files, device={a.device}", flush=True)

    sep = Separator(model="htdemucs", device=a.device, shifts=1, split=True, overlap=0.25, progress=False)
    failures, t0, done, skipped = [], time.time(), 0, 0
    for k, f in enumerate(files, 1):
        out_dir = DEMUCS_DIR / f.parent.name / f.stem
        if outputs_ok(out_dir, sf.info(f).frames):
            skipped += 1
            continue
        ok, msg = False, ""
        for _ in range(2):
            try:
                ok, msg = separate_one(sep, f, out_dir)
            except Exception as e:
                ok, msg = False, f"error: {e}"
            if ok:
                break
        if ok:
            done += 1
        else:
            failures.append(f"{f}\t{msg}")
            print(f"   FAILED {f.name}: {msg}", flush=True)
        print(f"[{k}/{len(files)}] {f.stem}  ({msg})  {(time.time() - t0) / 60:.1f} min", flush=True)

    (DEMUCS_DIR / "failures.txt").write_text("\n".join(failures))
    print(f"done: {done} separated, {skipped} already done, {len(failures)} failed", flush=True)


if __name__ == "__main__":
    main()
