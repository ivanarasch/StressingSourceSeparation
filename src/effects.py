"""Vocal effects used in Experiment 2, plus their four-level parameter tables.

The effects are deliberately simple, self-contained implementations (not commercial plugins).
Reverb works on a numpy array (samples, channels); the others work on pydub AudioSegments.
"""
import os

os.environ.setdefault("TORCHAUDIO_USE_TORCHCODEC", "0")

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from pydub import AudioSegment

from config import IMPULSE_RESPONSE

# ---------------------------------------------------------------- parameter tables (level 1 = mild ... 4 = extreme)
REVERB_LEVELS = [
    {"mix": 0.15, "ir_gain": 0.75},
    {"mix": 0.35, "ir_gain": 1.0},
    {"mix": 0.60, "ir_gain": 1.25},
    {"mix": 0.85, "ir_gain": 1.50},
]
DELAY_LEVELS = [
    {"delay_ms": 120, "repeats": 1},
    {"delay_ms": 200, "repeats": 2},
    {"delay_ms": 300, "repeats": 3},
    {"delay_ms": 400, "repeats": 4},
]
BITCRUSH_LEVELS = [
    {"bit_depth": 12, "downsample_factor": 4},
    {"bit_depth": 10, "downsample_factor": 4},
    {"bit_depth": 8, "downsample_factor": 4},
    {"bit_depth": 6, "downsample_factor": 4},
]
COMPRESSION_LEVELS = [
    dict(threshold=-25, ratio=1.4, attack_ms=5, release_ms=50, makeup_gain_db=0),
    dict(threshold=-30, ratio=3, attack_ms=10, release_ms=100, makeup_gain_db=2),
    dict(threshold=-35, ratio=8, attack_ms=20, release_ms=200, makeup_gain_db=8),
    dict(threshold=-40, ratio=20, attack_ms=40, release_ms=400, makeup_gain_db=12),
]


# ---------------------------------------------------------------- AudioSegment <-> numpy
def audiosegment_from_numpy(samples: np.ndarray, rate: int) -> AudioSegment:
    samples_int16 = (samples * 32767).astype(np.int16)
    return AudioSegment(
        samples_int16.tobytes(),
        frame_rate=rate,
        sample_width=2,
        channels=samples.shape[1] if samples.ndim > 1 else 1,
    )


def audiosegment_to_numpy(audio: AudioSegment) -> np.ndarray:
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32767
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels))
    return samples


# ---------------------------------------------------------------- reverb
_RIR = None


def _get_rir():
    """Load the impulse response once: keep the 1.01 s to 1.30 s portion, normalised to unit energy."""
    global _RIR
    if _RIR is None:
        if not IMPULSE_RESPONSE.exists():
            raise FileNotFoundError(
                f"Impulse response not found at {IMPULSE_RESPONSE}. See assets/README.md, "
                "or set SSS_IMPULSE_RESPONSE to its location."
            )
        raw, sr = torchaudio.load(str(IMPULSE_RESPONSE))
        rir = raw[:, int(sr * 1.01): int(sr * 1.3)]
        rir = rir / torch.norm(rir)
        _RIR = (rir, sr)
    return _RIR


def match_length_torch(a, b):
    """Trim or zero-pad tensor a (C, T) to the length of tensor b (C, T)."""
    Ta, Tb = a.shape[1], b.shape[1]
    if Ta > Tb:
        return a[:, :Tb]
    if Ta < Tb:
        return torch.cat([a, torch.zeros((a.shape[0], Tb - Ta), dtype=a.dtype)], dim=1)
    return a


def add_reverb_torch(vocals_np: np.ndarray, rate: int, mix=0.3, ir_gain=1.0):
    """Wet/dry reverb by FFT convolution with the (scaled) impulse response. vocals_np: (T, C)."""
    rir, rir_sr = _get_rir()
    wav = torch.tensor(vocals_np.T, dtype=torch.float32)
    if rate != rir_sr:
        wav = torchaudio.functional.resample(wav, rate, rir_sr)
    wet = F.fftconvolve(wav, rir * ir_gain)
    wet = wet[:, : wav.shape[1]]
    if rate != rir_sr:
        wav = torchaudio.functional.resample(wav, rir_sr, rate)
        wet = torchaudio.functional.resample(wet, rir_sr, rate)
    wet = match_length_torch(wet, wav)
    return ((1 - mix) * wav + mix * wet).T.numpy()


# ---------------------------------------------------------------- delay
def add_delay(audio: AudioSegment, delay_ms=200, repeats=2):
    """Feed-forward echoes: `repeats` copies delayed by k * delay_ms, each 8 dB down, added to the dry signal."""
    output = audio
    for i in range(1, repeats + 1):
        delayed = AudioSegment.silent(duration=delay_ms * i) + (audio - 8)
        output = output.overlay(delayed)
    return output


# ---------------------------------------------------------------- bitcrush
def add_bitcrush(audio: AudioSegment, bit_depth=8, downsample_factor=2):
    """Bit-depth reduction plus sample-and-hold downsampling (no anti-alias filter)."""
    samples = audiosegment_to_numpy(audio)
    max_val = 2 ** (bit_depth - 1)
    crushed = np.round(samples * max_val) / max_val
    for i in range(0, len(crushed), downsample_factor):
        crushed[i:i + downsample_factor] = crushed[i]
    return audiosegment_from_numpy(np.clip(crushed, -1.0, 1.0), audio.frame_rate)


# ---------------------------------------------------------------- compression
def add_compression(audio: AudioSegment, threshold=-25, ratio=4.0, attack_ms=10, release_ms=100, makeup_gain_db=0.0):
    """Simple hard-knee feed-forward compressor with an attack/release peak-envelope follower.

    It runs over the interleaved sample stream, so both channels share one envelope.
    """
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2 ** 15)
    attack_coeff = np.exp(-1.0 / (attack_ms * audio.frame_rate / 1000))
    release_coeff = np.exp(-1.0 / (release_ms * audio.frame_rate / 1000))

    env = 0.0
    out = np.zeros_like(samples)
    for i, x in enumerate(samples):
        rectified = abs(x)
        if rectified > env:
            env = attack_coeff * env + (1 - attack_coeff) * rectified
        else:
            env = release_coeff * env + (1 - release_coeff) * rectified

        env_db = 20 * np.log10(env + 1e-8)
        gain_db = (threshold + (env_db - threshold) / ratio) - env_db if env_db > threshold else 0.0
        out[i] = x * 10 ** ((gain_db + makeup_gain_db) / 20)

    out = (out * (2 ** 15)).astype(np.int16)
    return audio._spawn(out.tobytes())


# ---------------------------------------------------------------- registry used by the data-generation script
# (name, function, level table, takes_numpy_array)
EFFECTS = [
    ("reverb", add_reverb_torch, REVERB_LEVELS, True),
    ("delay", add_delay, DELAY_LEVELS, False),
    ("bitcrush", add_bitcrush, BITCRUSH_LEVELS, False),
    ("compression", add_compression, COMPRESSION_LEVELS, False),
]
