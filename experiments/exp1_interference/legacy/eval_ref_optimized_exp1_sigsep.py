import os
import numpy as np
import soundfile as sf
import museval
import librosa
import pandas as pd
import argparse
import warnings
import concurrent.futures
from tqdm import tqdm
import re

# --- CONFIGURATION ---
# 4 workers is safe for M3 Pro (18GB RAM)
MAX_WORKERS = 4

# Suppress warnings
warnings.filterwarnings("ignore")

def load_audio(path):
    """
    Robust audio loading using Librosa. 
    Forces stereo (2 channels) and original sample rate.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        # Check readability
        with sf.SoundFile(path) as f:
            pass
    except Exception as e:
        raise RuntimeError(f"File unreadable: {path}\nError: {e}")

    # Load with librosa
    audio, sr = librosa.load(path, sr=None, mono=False)
    
    # Handle shape
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    
    audio = audio.T
    return audio

def group_demucs_folders(ref_root):
    """
    Scans a Demucs 'Flat' directory and groups folders by Song Name.
    Returns: dict { "SongName": { "bass": "Path/To/Song_bass", "drums": ... } }
    """
    songs = {}
    
    # Regex to capture "Song Name" and "stem" from "Song Name_stem"
    # Matches: "Artist - Title_bass" -> Group 1: "Artist - Title", Group 2: "bass"
    pattern = re.compile(r"(.*)_(bass|drums|other|vocals)$")

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

    return songs

def find_spleeter_files(model_root, song_name, stem_name):
    """
    Finds Spleeter 'Nested' files: Model/Song/stem_vocals/[vocals.wav, accompaniment.wav]
    """
    # Spleeter folder naming convention: "stem_vocals"
    nested_dir = os.path.join(model_root, song_name, f"{stem_name}_vocals")
    
    if os.path.exists(nested_dir):
        v_path = os.path.join(nested_dir, "vocals.wav")
        acc_path = os.path.join(nested_dir, "accompaniment.wav")
        
        # Fallback for filenames
        if not os.path.exists(acc_path):
            acc_path = os.path.join(nested_dir, "no_vocals.wav")
            
        if os.path.exists(v_path) and os.path.exists(acc_path):
            return [v_path, acc_path]
    
    return None

def process_single_song_demucs_ref(args):
    """
    Worker Function:
    1. Receives a Demucs Song Map (containing paths to bass, drums, etc.)
    2. Loads Demucs Audio ONCE.
    3. Compares against all Models (Spleeter).
    """
    song_name, stems_map, model_infos = args
    results = []

    try:
        # Iterate through the stems found for this song (bass, drums, etc.)
        for stem_name, ref_dir_path in stems_map.items():
            
            # --- 1. Identify Reference (Demucs) Files ---
            # Demucs Flat folders usually contain 'vocals.wav' and 'no_vocals.wav'
            # (Note: In a 'bass' folder, 'vocals.wav' is the separated bass, 'no_vocals' is the rest)
            
            ref_files = [
                os.path.join(ref_dir_path, "vocals.wav"),
                os.path.join(ref_dir_path, "no_vocals.wav")
            ]
            
            # Fallback if named accompaniment
            if not os.path.exists(ref_files[1]):
                ref_files[1] = os.path.join(ref_dir_path, "accompaniment.wav")
                
            if not all(os.path.exists(f) for f in ref_files):
                continue

            # --- 2. Load Reference Audio (CACHE IT) ---
            # Optimization: Load Demucs once, compare to all models
            try:
                loaded_refs = [load_audio(p) for p in ref_files]
            except Exception:
                continue

            # --- 3. Compare against Spleeter (and other models) ---
            for model_name, model_root in model_infos:
                
                # A. Find Model Files (Spleeter Nested Structure)
                est_files = find_spleeter_files(model_root, song_name, stem_name)
                
                if not est_files:
                    # print(f"Missing {stem_name} for {song_name} in {model_name}")
                    continue

                # B. Load Model Audio
                try:
                    loaded_ests = [load_audio(p) for p in est_files]
                except Exception:
                    continue

                # C. Align Lengths
                min_len = min([x.shape[0] for x in loaded_refs + loaded_ests])
                
                curr_refs = [x[:min_len, :] for x in loaded_refs]
                curr_ests = [x[:min_len, :] for x in loaded_ests]

                references = np.stack(curr_refs)
                estimates = np.stack(curr_ests)

                # D. Evaluate (Fast Mode win=1.0)
                sdr, isr, sir, sar = museval.evaluate(references, estimates, win=1.0)

                # E. Store Results
                # Label 0 is usually the Stem (Bass), Label 1 is Accompaniment
                source_labels = [stem_name, "accompaniment"]
                
                for i, label in enumerate(source_labels):
                    results.append({
                        "Model": model_name,
                        "Song": song_name,
                        "Combination": f"{stem_name}_vocals", # Maintaining naming convention
                        "Source": label,
                        "SDR": np.nanmedian(sdr[i]),
                        "SIR": np.nanmedian(sir[i]),
                        "SAR": np.nanmedian(sar[i]),
                        "ISR": np.nanmedian(isr[i])
                    })

    except Exception:
        pass

    return results

def main():
    parser = argparse.ArgumentParser(description="Demucs-Ref Optimized Comparison.")
    parser.add_argument("--ref", required=True, help="Path to Demucs (Flat) Root Directory")
    parser.add_argument("--models", nargs='+', required=True, help="Path to Spleeter (Nested) Root Directories")
    parser.add_argument("--output", default="demucs_ref_results.xlsx", help="Output Excel filename")

    args = parser.parse_args()

    all_results = []
    print(f"Reference Path (Demucs): {args.ref}")
    print(f"Workers: {MAX_WORKERS}")

    # 1. Group Demucs folders by Song
    print("Scanning Demucs Reference folder...")
    songs_map = group_demucs_folders(args.ref)
    
    if not songs_map:
        print("Error: No 'SongName_stem' folders found in Reference. Is it a flat Demucs folder?")
        return

    print(f"Found {len(songs_map)} unique songs.")

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
        results_generator = list(tqdm(executor.map(process_single_song_demucs_ref, tasks), total=len(tasks)))
        
        for res in results_generator:
            if res:
                all_results.extend(res)

    # 5. Save
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_excel(args.output, index=False)
        print(f"\nSuccess! Saved to {args.output}")
        try:
            print("\nMedian SDR Summary:")
            print(df.groupby(["Model", "Source"])["SDR"].median())
        except:
            pass
    else:
        print("\nNo results generated. Check paths.")

if __name__ == "__main__":
    main()
