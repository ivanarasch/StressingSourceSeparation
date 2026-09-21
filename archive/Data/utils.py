from pydub import AudioSegment
import numpy as np

# ============================================================
# AudioSegment <-> NumPy conversions
# ============================================================

def audiosegment_from_numpy(samples: np.ndarray, rate: int) -> AudioSegment:
    samples_int16 = (samples * 32767).astype(np.int16)
    audio = AudioSegment(
        samples_int16.tobytes(),
        frame_rate=rate,
        sample_width=2,
        channels=samples.shape[1] if samples.ndim > 1 else 1
    )
    return audio

def audiosegment_to_numpy(audio: AudioSegment) -> np.ndarray:
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32767
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels))
    return samples

# ============================================================
# Reverb
# ============================================================

def add_reverb(audio: AudioSegment, delay_ms=50, repeats=3):
    output = audio
    for i in range(1, repeats + 1):
        delayed = AudioSegment.silent(duration=delay_ms * i) + (audio - 6)
        output = output.overlay(delayed)
    return output

# ============================================================
# Delay
# ============================================================

def add_delay(audio: AudioSegment, delay_ms=200, repeats=2):
    output = audio
    for i in range(1, repeats + 1):
        delayed = AudioSegment.silent(duration=delay_ms * i) + (audio - 8)
        output = output.overlay(delayed)
    return output

# ============================================================
# Chorus (simple pydub-based modulation)
# ============================================================

def add_chorus(audio: AudioSegment, depth=0.3, rate=0.5, wet=0.25):
    modulated = audio + AudioSegment.silent(duration=0)
    delay_range = int(depth * 10)

    for i in range(delay_range):
        delayed = AudioSegment.silent(duration=i) + (audio - 10)
        modulated = modulated.overlay(delayed)

    dry = audio - int(wet * 10)
    wet_mix = modulated - int((1 - wet) * 10)
    return dry.overlay(wet_mix)

# ============================================================
# Bitcrush
# ============================================================

def add_bitcrush(audio: AudioSegment, bit_depth=8, downsample_factor=2):
    samples = audiosegment_to_numpy(audio)

    max_val = 2 ** (bit_depth - 1)
    crushed = np.round(samples * max_val) / max_val

    for i in range(0, len(crushed), downsample_factor):
        crushed[i:i+downsample_factor] = crushed[i]

    crushed = np.clip(crushed, -1.0, 1.0)
    return audiosegment_from_numpy(crushed, audio.frame_rate)

# ============================================================
# Compression (simple soft knee)
# ============================================================

def add_compression(audio: AudioSegment, threshold=-25, ratio=3.0):
    samples = audiosegment_to_numpy(audio)

    thr = 10 ** (threshold / 20)

    over = np.abs(samples) > thr
    samples[over] = np.sign(samples[over]) * (
        thr + (np.abs(samples[over]) - thr) / ratio
    )

    samples = np.clip(samples, -1.0, 1.0)
    return audiosegment_from_numpy(samples, audio.frame_rate)



