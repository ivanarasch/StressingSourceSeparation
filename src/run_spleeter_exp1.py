"""Spleeter 2-stem on the Experiment 1 pairs (this is the loop originally run in Google Colab).

    ~/spleeter_env/bin/python src/run_spleeter_exp1.py <paired_stems_dir> <output_dir>

Output: <output_dir>/<song>/<pair name>/{vocals,accompaniment}.wav
Note: this reuses one Separator across files, which we checked is fine for this 2-stem run (its outputs matched a
clean fresh-per-file re-run on a 20-file spot check). Do NOT copy this pattern to the 4-stem Experiment 2 run;
there, reusing the Separator corrupted the outputs (see src/run_spleeter_exp2.py).
"""
import glob
import os
import sys

from spleeter.separator import Separator

data_home, output_home = sys.argv[1], sys.argv[2]
os.makedirs(output_home, exist_ok=True)
separator = Separator("spleeter:2stems")
audio_files = glob.glob(os.path.join(data_home, "**/*.wav"), recursive=True)
print(f"Found {len(audio_files)} files.")

for audio_path in audio_files:
    try:
        print("Separating:", audio_path)
        song_out_dir = os.path.join(output_home, os.path.basename(os.path.dirname(audio_path)))
        os.makedirs(song_out_dir, exist_ok=True)
        separator.separate_to_file(audio_path, song_out_dir)
    except Exception as e:
        print("Failed on", audio_path, "->", e)
