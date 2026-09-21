#!/usr/bin/env python
"""
Fast evaluation script using SI-SDR + SI-SAR-like artifact metric (mono).
- Uses ProcessPoolExecutor for real parallel CPU usage.
- Auto-detects original folders if they contain 'original' (case-insensitive)
- Writes results to an Excel file.
"""

import os
import numpy as np
import soundfile as sf
import pandas as pd
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import math
import multiprocessing

# --- Configuration (edit as needed) ---
ROOT_DIR = "/Sample/Data/Path"
OUTPUT_FILE = "evaluation_results_si_metrics.xlsx"
ORIGINAL_SUFFIX = "_original"  # still used if you have exact matches
# --------------------------

# Determine number of workers automatically (leave one core for system)
CPU_COUNT = multiprocessing.cpu_count() or 1
MAX_WORKERS = max(1, CPU_COUNT - 1)


# ---------------- Utility functions ---------------- #
def load_audio_mono(path):
    """
    Load audio and downmix to mono.
    Returns 1D numpy array and sample rate.
    """
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    else:
        audio = audio.astype(np.float64)
    return audio.astype(np.float64), sr


def safe_padding_crop(a, b):
    """
    Crop arrays a and b to the same minimum length.
    Returns cropped (a, b)
    """
    min_len = min(a.shape[0], b.shape[0])
    return a[:min_len], b[:min_len]


def si_sdr(ref, est, eps=1e-8):
    """
    Compute SI-SDR between 1D reference and estimate arrays.
    SI-SDR = 10 * log10( ||s_target||^2 / ||e_res||^2 )
    where s_target is projection of estimate onto reference.
    """
    # ensure 1D
    r = ref.flatten().astype(np.float64)
    e = est.flatten().astype(np.float64)

    # zero-vector guard
    if np.allclose(r, 0):
        return float('nan')

    # projection
    alpha = np.dot(e, r) / (np.dot(r, r) + eps)
    e_target = alpha * r
    e_res = e - e_target

    num = np.sum(e_target ** 2)
    den = np.sum(e_res ** 2) + eps
    return 10.0 * np.log10((num + eps) / den)


def si_sar_like(ref, est, eps=1e-8):
    """
    Fast SI-SAR-like artifact ratio approximation.

    We compute:
      artifacts = est - proj(est onto ref)
      SI-SAR-like = 10 * log10( ||est||^2 / ||artifacts||^2 )

    This is not the exact BSS Eval SAR, but correlates well and is ~100x faster.
    """
    r = ref.flatten().astype(np.float64)
    e = est.flatten().astype(np.float64)

    if np.allclose(e, 0):
        return float('nan')

    # projection of estimate onto reference
    if np.allclose(r, 0):
        # no reference energy -> artifacts = estimate
        artifacts_energy = np.sum(e ** 2)
        est_energy = np.sum(e ** 2)
        return 10.0 * np.log10((est_energy + eps) / (artifacts_energy + eps))

    alpha = np.dot(e, r) / (np.dot(r, r) + eps)
    e_target = alpha * r
    artifacts = e - e_target

    est_energy = np.sum(e ** 2)
    artifacts_energy = np.sum(artifacts ** 2) + eps

    return 10.0 * np.log10((est_energy + eps) / artifacts_energy)


def rmse(ref, est):
    a, b = safe_padding_crop(ref, est)
    return math.sqrt(float(np.mean((a - b) ** 2)))


def get_wav_files(folder_path):
    return sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.wav') and not f.startswith('._') and not f.startswith('.')])


def find_original_folder(versions, original_suffix=ORIGINAL_SUFFIX):
    """
    Given a list of subfolders, find the 'original' folder.
    Priority:
      1) folder that endswith original_suffix
      2) folder whose name (lower) == 'original'
      3) folder containing 'original' (case-insensitive)
    Returns folder name or None.
    """
    for v in versions:
        if v.endswith(original_suffix):
            return v
    for v in versions:
        if v.lower() == 'original':
            return v
    for v in versions:
        if 'original' in v.lower():
            return v
    return None


# ---------------- Per-song worker ---------------- #
def process_single_song_folder(song_folder_path):
    """
    Process one song folder (runs inside a worker process).
    Returns list of result dicts (one per source file) or empty list on error.
    """
    results = []
    try:
        if not os.path.isdir(song_folder_path):
            return results

        versions = [d for d in os.listdir(song_folder_path) if os.path.isdir(os.path.join(song_folder_path, d))]
        if not versions:
            return results

        original_folder = find_original_folder(versions)
        if original_folder is None:
            # no original found -> skip
            return results

        original_path = os.path.join(song_folder_path, original_folder)
        original_files = get_wav_files(original_path)
        if not original_files:
            return results

        # Parse song name from original folder (fallback to folder name)
        song_name = re.sub(r'_(reverb|delay|compression|bitcrush)_l\d+$', '', original_folder)
        if song_name.endswith(ORIGINAL_SUFFIX):
            song_name = song_name[:-len(ORIGINAL_SUFFIX)]
        song_name = song_name.strip('_ ').strip()
        if not song_name:
            # use parent folder name
            song_name = os.path.basename(song_folder_path)

        # Preload original stems ONCE (mono)
        original_audio_cache = {}
        sr_ref = None
        for f in original_files:
            p = os.path.join(original_path, f)
            audio, sr = load_audio_mono(p)
            if sr_ref is None:
                sr_ref = sr
            original_audio_cache[f] = audio

        # For every version (effect) compute metrics vs original
        for v in versions:
            if v == original_folder:
                continue

            # Identify effect and level if possible
            effect, level = None, None
            match = re.search(r'_(reverb|delay|compression|bitcrush)_l(\d+)$', v)
            if match:
                effect = match.group(1)
                try:
                    level = int(match.group(2))
                except Exception:
                    level = None
            else:
                # fallback: use folder name as effect
                effect = v

            effect_path = os.path.join(song_folder_path, v)
            effect_files = get_wav_files(effect_path)
            if not effect_files:
                continue

            # Require same file lists (names) between original and effect to compare stems
            if original_files != effect_files:
                # try best-effort: intersect common files
                common = sorted(list(set(original_files).intersection(set(effect_files))))
                if not common:
                    # no matching files -> skip version
                    continue
                compare_files = common
            else:
                compare_files = original_files

            # For each stem compute metrics
            for stem in compare_files:
                ref = original_audio_cache.get(stem)
                est_path = os.path.join(effect_path, stem)
                if ref is None:
                    # maybe ref wasn't loaded due to mismatch, attempt load
                    ref, _ = load_audio_mono(os.path.join(original_path, stem))
                try:
                    est, sr_est = load_audio_mono(est_path)
                except Exception as e:
                    # if estimate missing or unreadable, skip
                    continue

                # ensure same sampling rate? (we assume same SR; ignore resampling here)
                a_ref, a_est = safe_padding_crop(ref, est)

                # compute metrics
                si_sdr_val = si_sdr(a_ref, a_est)
                si_sar_val = si_sar_like(a_ref, a_est)
                rmse_val = rmse(a_ref, a_est)

                results.append({
                    "song": song_name,
                    "version_folder": v,
                    "effect": effect,
                    "level": level,
                    "source": os.path.splitext(stem)[0],
                    "SI_SDR": si_sdr_val,
                    "SI_SAR_like": si_sar_val,
                    "RMSE": rmse_val,
                    "sr_ref": sr_ref,
                })

    except Exception as e:
        # Worker-level exception: return empty list but don't crash main pool
        return [{"error": f"Error processing {song_folder_path}: {e}"}]

    return results


# ---------------- Main ---------------- #
def main():
    # Discover song folders
    all_song_paths = [os.path.join(ROOT_DIR, d) for d in os.listdir(ROOT_DIR)
                      if os.path.isdir(os.path.join(ROOT_DIR, d))]
    print(f"Found {len(all_song_paths)} song folders in {ROOT_DIR}")
    if not all_song_paths:
        print("No song folders found. Exiting.")
        return

    all_results = []

    # Use ProcessPoolExecutor for real parallel CPU usage
    workers = min(MAX_WORKERS, len(all_song_paths))
    print(f"Using {workers} worker processes (CPU cores: {CPU_COUNT})")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_song_folder, path): path for path in all_song_paths}

        # as each song finishes, extend results and update progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating songs"):
            path = futures[future]
            try:
                res = future.result()
            except Exception as e:
                print(f"Worker error for {path}: {e}")
                continue

            # In case a worker returned an error dict, include that as a row for debugging
            if res:
                # If result is a list of dicts, extend
                if isinstance(res, list):
                    # filter out error dicts
                    for item in res:
                        if isinstance(item, dict) and "error" in item:
                            print(item["error"])
                        else:
                            all_results.append(item)
                else:
                    # unexpected return type
                    print(f"Unexpected worker return for {path}: {type(res)}")

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_excel(OUTPUT_FILE, index=False)
        print(f"\nEvaluation complete! Results saved to {OUTPUT_FILE}")
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
