"""Central place for every path. Override any of them with environment variables.

    SSS_DATA_ROOT          where generated audio lives (default: <repo>/data)
    MUSDB18_ROOT           MUSDB18 dataset root (default: <data root>/musdb18)
    SSS_IMPULSE_RESPONSE   room impulse response used by the reverb effect
    SSS_MIX_DIR, SSS_REF_DIR, SSS_DEMUCS_DIR, SSS_SPLEETER_DIR   override individual Experiment 2 folders
    SSS_EXP1_PAIRS_DIR, SSS_EXP1_SPLEETER_DIR, SSS_EXP1_DEMUCS_DIR, SSS_EXP1_RESULTS_DIR   Experiment 1 folders
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SSS_DATA_ROOT", REPO_ROOT / "data"))
MUSDB_ROOT = Path(os.environ.get("MUSDB18_ROOT", DATA_ROOT / "musdb18"))
IMPULSE_RESPONSE = Path(os.environ.get("SSS_IMPULSE_RESPONSE", REPO_ROOT / "assets" / "impulseresponse.wav"))

# Experiment 2 layout
MIX_DIR = Path(os.environ.get("SSS_MIX_DIR", DATA_ROOT / "exp2" / "mixes"))                # <song>/<song>_<cond>.wav
REF_DIR = Path(os.environ.get("SSS_REF_DIR", DATA_ROOT / "exp2" / "references"))           # <song>/<cond>/{drums,bass,other,vocals}.wav (+ <song>/_raw)
DEMUCS_DIR = Path(os.environ.get("SSS_DEMUCS_DIR", DATA_ROOT / "exp2" / "separated_demucs"))       # <song>/<song>_<cond>/{stem}.wav
SPLEETER_DIR = Path(os.environ.get("SSS_SPLEETER_DIR", DATA_ROOT / "exp2" / "separated_spleeter"))  # same layout
RESULTS_DIR = Path(os.environ.get("SSS_RESULTS_DIR", REPO_ROOT / "results" / "exp2"))

# Experiment 1 layout
EXP1_PAIRS_DIR = Path(os.environ.get("SSS_EXP1_PAIRS_DIR", DATA_ROOT / "exp1" / "paired_stems"))   # <song>/<pair>_vocals.wav
EXP1_SPLEETER_DIR = Path(os.environ.get("SSS_EXP1_SPLEETER_DIR", DATA_ROOT / "exp1" / "separated_spleeter"))   # <song>/<pair>_vocals/vocals.wav
EXP1_DEMUCS_DIR = Path(os.environ.get("SSS_EXP1_DEMUCS_DIR", DATA_ROOT / "exp1" / "separated_outputs_experiment1"))  # <song>_<pair>/vocals.wav
EXP1_RESULTS_DIR = Path(os.environ.get("SSS_EXP1_RESULTS_DIR", REPO_ROOT / "results" / "exp1"))

STEMS = ["vocals", "drums", "bass", "other"]
MODELS = {"demucs": DEMUCS_DIR, "spleeter": SPLEETER_DIR}
