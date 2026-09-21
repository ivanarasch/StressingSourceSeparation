"""Experiment 1 data: pair the vocal stem with each single non-vocal stem.

    python src/make_pairs_exp1.py

For each MUSDB18 test song writes <EXP1_PAIRS_DIR>/<song>/{drums,bass,other,accompaniment}_vocals.wav,
where accompaniment = drums + bass + other (everything except vocals; it is NOT a single instrument).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import musdb
import soundfile as sf

from config import EXP1_PAIRS_DIR, MUSDB_ROOT

out_root = EXP1_PAIRS_DIR
out_root.mkdir(parents=True, exist_ok=True)

for track in musdb.DB(root=str(MUSDB_ROOT), subsets="test").tracks:
    print(f"Processing {track.name}...")
    rate = track.rate
    vocals = track.targets["vocals"].audio
    drums = track.targets["drums"].audio
    bass = track.targets["bass"].audio
    other = track.targets["other"].audio
    accompaniment = drums + bass + other

    out = out_root / track.name
    out.mkdir(exist_ok=True)
    sf.write(out / "drums_vocals.wav", drums + vocals, rate)
    sf.write(out / "bass_vocals.wav", bass + vocals, rate)
    sf.write(out / "other_vocals.wav", other + vocals, rate)
    sf.write(out / "accompaniment_vocals.wav", accompaniment + vocals, rate)
