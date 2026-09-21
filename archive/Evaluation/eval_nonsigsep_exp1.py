#!/usr/bin/env python
"""
Combined Evaluation Script
- Metrics: SI-SDR, SI-SAR-like, RMSE (from run_eval_ivana.py)
- Structure: Demucs Flat Reference vs Spleeter Nested Models (from eval_demucs_ref_optimized_exp1.py)
"""

import os
import numpy as np
import soundfile as sf
import pandas as pd
import argparse
import warnings
import concurrent.futures
from tqdm import tqdm
import re
import math
import multiprocessing

# --- CONFIGURATION ---
# Determine number of workers automatically
#CPU_COUNT = multiprocessing.cpu_count() or 1
MAX_WORKERS = 2

# Suppress warnings
warnings.filterwarnings("ignore")

# ---------------- Utility functions (from run_eval_ivana) ---------------- #

def load_audio_mono(path):
    """
    Load audio and downmix to mono.
    Returns 1D numpy array and sample rate.
    """
    try:
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        else:
            audio = audio.astype(np.float64)
        return audio.astype(np.float64), sr
    except Exception as e:
        return None, None

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
    """
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

# ---------------- Structure Logic (from eval_demucs_ref_optimized) ---------------- #

def group_demucs_folders(ref_root):
    """
    Scans a Demucs 'Flat' directory and groups folders by Song Name.
    Returns: dict { "SongName": { "bass": "Path/To/Song_bass", "drums": ... } }
    """
    songs = {}
    
    # Regex to capture "Song Name" and "stem" from "Song Name_stem"
    # Matches: "Artist - Title_bass" -> Group 1: "Artist - Title", Group 2: "bass"
    pattern = re.compile(r"(.*)_(bass|drums|other|vocals)$")

    try:
        for folder in os.listdir(ref_root):
            full_path = os.path.join(ref_root, folder)
            if not os.path.isdir(full_path):
                continue

            match = pattern.match(folder)
            if match:
                song_name = match.group(1)
                stem_name = match.group(2)
                
                if song_name not in songs:
                    songs[song_name] = {}
                
                songs[song_name][stem_name] = full_path
    except OSError as e:
        print(f"Error scanning directory {ref_root}: {e}")
        return {}

    return songs

def find_spleeter_files(model_root, song_name, stem_name):
    """
    Finds Spleeter 'Nested' files: Model/Song/stem_vocals/[vocals.wav, accompaniment.wav]
    Returns tuple: (vocals_path, accompaniment_path) or None
    """
    # Spleeter folder naming convention: "stem_vocals"
    nested_dir = os.path.join(model_root, song_name, f"{stem_name}_vocals")
    
    if os.path.exists(nested_dir):
        v_path = os.path.join(nested_dir, "vocals.wav")
        acc_path = os.path.join(nested_dir, "accompaniment.wav")
        
        # Fallback for filenames if Spleeter used different naming
        if not os.path.exists(acc_path):
            acc_path = os.path.join(nested_dir, "no_vocals.wav")
            
        if os.path.exists(v_path) and os.path.exists(acc_path):
            return (v_path, acc_path)
    
    return None

def process_single_song_comparison(args):
    """
    Worker Function:
    1. Receives a Demucs Song Map (containing paths to bass, drums, etc.)
    2. Loads Demucs Audio (Ref) -> Mono.
    3. Compares against all Models (Spleeter) -> Mono.
    4. Computes SI-SDR, SI-SAR, RMSE.
    """
    song_name, stems_map, model_infos = args
    results = []

    try:
        # Iterate through the stems found for this song (bass, drums, etc.)
        for stem_name, ref_dir_path in stems_map.items():
            
            # --- 1. Identify Reference (Demucs) Files ---
            # Demucs Flat folders usually contain 'vocals.wav' and 'no_vocals.wav'
            # (Note: In a 'bass' folder, 'vocals.wav' is the separated bass, 'no_vocals' is the rest)
            ref_stem_path = os.path.join(ref_dir_path, "vocals.wav")
            ref_acc_path = os.path.join(ref_dir_path, "no_vocals.wav")
            
            # Fallback if named accompaniment
            if not os.path.exists(ref_acc_path):
                ref_acc_path = os.path.join(ref_dir_path, "accompaniment.wav")

            # Check existence
            if not (os.path.exists(ref_stem_path) and os.path.exists(ref_acc_path)):
                continue

            # --- 2. Load Reference Audio (Mono) ---
            ref_stem_audio, sr_ref = load_audio_mono(ref_stem_path)
            ref_acc_audio, _ = load_audio_mono(ref_acc_path)

            if ref_stem_audio is None or ref_acc_audio is None:
                continue

            # --- 3. Compare against Models ---
            for model_name, model_root in model_infos:
                
                # A. Find Model Files (Spleeter Nested Structure)
                est_paths = find_spleeter_files(model_root, song_name, stem_name)
                
                if not est_paths:
                    continue
                
                est_stem_path, est_acc_path = est_paths

                # B. Load Model Audio (Mono)
                est_stem_audio, sr_est = load_audio_mono(est_stem_path)
                est_acc_audio, _ = load_audio_mono(est_acc_path)

                if est_stem_audio is None or est_acc_audio is None:
                    continue

                # C. Evaluate Both "Stem" and "Accompaniment" components
                # Pair 1: The specific stem (e.g., bass)
                # Pair 2: The accompaniment (e.g., everything else)
                
                pairs = [
                    (stem_name, ref_stem_audio, est_stem_audio),
                    ("accompaniment", ref_acc_audio, est_acc_audio)
                ]

                for source_label, ref_sig, est_sig in pairs:
                    # Crop
                    r_crop, e_crop = safe_padding_crop(ref_sig, est_sig)
                    
                    # Compute Metrics
                    si_sdr_val = si_sdr(r_crop, e_crop)
                    si_sar_val = si_sar_like(r_crop, e_crop)
                    rmse_val = rmse(r_crop, e_crop)

                    results.append({
                        "Model": model_name,
                        "Song": song_name,
                        "Combination": f"{stem_name}_vocals", # Maintaining naming convention
                        "Source": source_label,
                        "SI_SDR": si_sdr_val,
                        "SI_SAR_like": si_sar_val,
                        "RMSE": rmse_val,
                        "sr_ref": sr_ref
                    })

    except Exception as e:
        # Return error as a result row for debugging
        results.append({"error": f"Error in {song_name}: {str(e)}"})

    return results

# ---------------- Main ---------------- #

def main():
    parser = argparse.ArgumentParser(description="Demucs vs Spleeter Comparison (SI-SDR/SI-SAR/RMSE).")
    parser.add_argument("--ref", required=True, help="Path to Reference Root (Demucs Flat Folder)")
    parser.add_argument("--models", nargs='+', required=True, help="Path to Model Root directories (Spleeter Nested)")
    parser.add_argument("--output", default="evaluation_results_combined.xlsx", help="Output Excel filename")

    args = parser.parse_args()

    all_results = []
    print(f"Reference Path (Demucs): {args.ref}")
    print(f"Workers: {MAX_WORKERS}")

    # 1. Group Demucs folders by Song
    print("Scanning Reference folder...")
    songs_map = group_demucs_folders(args.ref)
    
    if not songs_map:
        print("Error: No 'SongName_stem' folders found in Reference. Is it a flat Demucs folder?")
        return

    print(f"Found {len(songs_map)} unique songs in Reference.")

    # 2. Prepare Model Info
    model_infos = []
    for m_path in args.models:
        m_name = os.path.basename(os.path.normpath(m_path))
        model_infos.append((m_name, m_path))

    # 3. Create Tasks
    # Task: (SongName, DictOfStems, ListOfModels)
    tasks = [(song, stems, model_infos) for song, stems in songs_map.items()]

    # 4. Execute
    print("Starting processing...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Use tqdm for progress bar
        futures = {executor.submit(process_single_song_comparison, task): task for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                res = future.result()
                if res:
                    for item in res:
                        if "error" in item:
                            print(f"[WARN] {item['error']}")
                        else:
                            all_results.append(item)
            except Exception as e:
                print(f"Fatal worker error: {e}")

    # 5. Save
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Organize columns
        cols = ["Model", "Song", "Combination", "Source", "SI_SDR", "SI_SAR_like", "RMSE", "sr_ref"]
        # Ensure all exist
        final_cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
        df = df[final_cols]

        df.to_excel(args.output, index=False)
        print(f"\nSuccess! Results saved to {args.output}")
        
        # Print summary if data exists
        if "SI_SDR" in df.columns:
            try:
                print("\nMedian SI-SDR Summary:")
                print(df.groupby(["Model", "Source"])["SI_SDR"].median())
            except:
                pass
    else:
        print("\nNo results generated. Check paths.")

if __name__ == "__main__":
    main()
