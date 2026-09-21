# Experiment 2: vocal effects

**Question.** How do reverb, delay, compression and bitcrushing applied to the vocal stem change source separation quality, for Spleeter and Demucs?

**Design.**
- 50 MUSDB18 test songs; 17 conditions each: no effect, plus 4 effects x 4 levels; 850 mixes per model.
- Only the vocal stem is processed. The mix is drums + bass + other + processed vocal.
- The processed vocal is scaled to the original vocal's RMS. If a mix would clip, the whole mix (and the matching reference stems) is scaled by one common factor. Every gain is in `results/exp2/loudness_log.csv`.
- Metric: full-track time-domain SDR against the exact stems in the mix (vocal in its processed form). delta-SDR is the change relative to the no-effect condition for the same song, stem and model. A second analysis scores the vocal against the original dry vocal.

**Effect parameters** are in `src/effects.py` (`REVERB_LEVELS`, `DELAY_LEVELS`, `COMPRESSION_LEVELS`, `BITCRUSH_LEVELS`) and tabulated in `docs/EXPERIMENT2_HANDOFF.md` (Section 4.1).

**Run it:** see the top-level README (or `notebooks/exp2_vocal_effects.ipynb`).

**Results:** `results/exp2/` (metrics, dry-reference scores, loudness log, figures, tables) and the write-up in `docs/EXPERIMENT2_HANDOFF.md`.

**Models.** Spleeter 2.3.2 `spleeter:4stems` (CPU, a fresh Separator per file); Demucs `htdemucs` (version 4.1.0a2 in our runs, default settings including a random time shift, so reruns differ slightly).
