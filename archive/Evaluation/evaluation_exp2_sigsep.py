import os
import numpy as np
import sys
import soundfile as sf
import museval
import pandas as pd
import re
import concurrent.futures
from tqdm import tqdm  # Progress bar

# --- Configuration ---
# Adjust MAX_WORKERS based on your CPU cores. 
# If you have an 8-core machine, try 6 or 8.
# Warning: High workers = High RAM usage. If it crashes, lower this number.
MAX_WORKERS = 4 

def load_audio_fast(path):
    """
    Replaces librosa.load with soundfile for speed.
    Returns: audio (n_samples, n_channels), sample_rate
    """
    try:
        audio, sr = sf.read(path)
    except Exception as e:
        raise RuntimeError(f"File unreadable: {path}\nError: {e}")
        
    # Ensure shape is (n_samples, n_channels)
    # Soundfile usually returns (samples, channels) for stereo,
    # or (samples,) for mono.
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    
    # Note: Museval expects (n_sources, n_samples, n_channels)
    # We don't transpose here because sf.read already gives (Samples, Channels)
    # unlike librosa which gives (Channels, Samples).
    return audio, sr

def evaluate_custom_separation(reference_files, estimate_files):
    refs = []
    # Load References
    for p in reference_files:
        audio, _ = load_audio_fast(p)
        refs.append(audio)
        
    ests = []
    # Load Estimates
    for p in estimate_files:
        audio, _ = load_audio_fast(p)
        ests.append(audio) 

    # Crop to minimum length
    min_len = min([x.shape[0] for x in refs + ests])
    refs = [x[:min_len, :] for x in refs]
    ests = [x[:min_len, :] for x in ests]

    references = np.stack(refs)
    estimates = np.stack(ests)

    # Compute metrics (Global)
    scores = museval.evaluate(references, estimates, win=1.0)
    return scores

def get_source_files(version_dir):
    return sorted([f for f in os.listdir(version_dir)
                   if f.endswith('.wav') and not f.startswith('._') and not f.startswith('.')])

def parse_effect_info(folder_name, original_suffix):
    if folder_name.endswith(original_suffix):
        return None, None
    match = re.search(r'_(reverb|delay|compression|bitcrush)_l(\d+)$', folder_name)
    if match:
        return match.group(1), int(match.group(2))
    else:
        return None, None

def parse_song_name(folder_name, original_suffix):
    cleaned_name = re.sub(r'_(reverb|delay|compression|bitcrush)_l\d+$', '', folder_name)
    if cleaned_name.endswith(original_suffix):
        cleaned_name = cleaned_name[:-len(original_suffix)]
    return cleaned_name.strip('_ ').strip()

def process_single_song_folder(song_folder_path, original_suffix):
    """
    Worker function to process a single song folder.
    Returns a list of result dictionaries.
    """
    results = []
    try:
        # Check if path exists
        if not os.path.exists(song_folder_path):
            return results

        versions = [d for d in os.listdir(song_folder_path) if os.path.isdir(os.path.join(song_folder_path, d))]
        
        # Identify original version folder
        original_folders = [v for v in versions if v.endswith(original_suffix)]
        if not original_folders:
            return results # Skip if no original
            
        original_folder = original_folders[0]
        original_path = os.path.join(song_folder_path, original_folder)
        original_files = get_source_files(original_path)
        
        if not original_files:
            return results

        song_name = parse_song_name(original_folder, original_suffix)

        for v in versions:
            # Skip the original folder itself
            if v == original_folder:
                continue
                
            effect, level = parse_effect_info(v, original_suffix)
            if effect is None:
                continue

            effect_path = os.path.join(song_folder_path, v)
            effect_files = get_source_files(effect_path)

            if original_files != effect_files:
                # Mismatch in file count/names
                continue

            ref_paths = [os.path.join(original_path, f) for f in original_files]
            est_paths = [os.path.join(effect_path, f) for f in effect_files]

            # Evaluate
            sdr, isr, sir, sar = evaluate_custom_separation(ref_paths, est_paths)
            
            sources = [os.path.splitext(f)[0] for f in original_files]
            for i, source in enumerate(sources):
                results.append({
                    "song": song_name,
                    "effect": effect,
                    "level": level,
                    "source": source,
                    "SDR": np.nanmedian(sdr[i]),
                    "ISR": np.nanmedian(isr[i]),
                    "SIR": np.nanmedian(sir[i]),
                    "SAR": np.nanmedian(sar[i]),
                })
                
    except Exception as e:
        print(f"\nError processing {song_folder_path}: {e}")
        
    return results

def main():
    # --- Arguments ---
    root_dir = "/Sample/Data/Path"
    original_suffix = "_original"
    output_file = "results_fast.xlsx"

    print(f"Scanning {root_dir}...")
    all_song_paths = []
    
    # Gather all song folder paths first
    if os.path.exists(root_dir):
        for song_folder in os.listdir(root_dir):
            full_path = os.path.join(root_dir, song_folder)
            if os.path.isdir(full_path):
                all_song_paths.append(full_path)
    
    print(f"Found {len(all_song_paths)} song folders. Starting processing with {MAX_WORKERS} workers...")
    
    all_results = []
    
    # Use ProcessPoolExecutor to parallelize
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_song_folder, path, original_suffix) for path in all_song_paths]
        
        # Use tqdm to show a progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_song_paths)):
            result = future.result()
            if result:
                all_results.extend(result)

    # Save
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_excel(output_file, index=False)
        print(f"\nSuccess! Saved {len(df)} rows to {output_file}")
    else:
        print("\nNo results found.")

if __name__ == "__main__":
    main()
