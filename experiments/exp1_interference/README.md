# Experiment 1: instrumental interference

**Question.** How do different non-vocal stems interfere with extracting the vocal?

**Design.** For each of the 50 MUSDB18 test songs, the vocal stem is added to a single non-vocal stem to make four mixtures: `drums_vocals`, `bass_vocals`, `other_vocals` and `accompaniment_vocals`, where **accompaniment = drums + bass + other** (everything except vocals; it is not a single instrument, and `other` is a catch-all that includes guitars, keys, synths and more). Both models (2-stem mode) then separate the vocal, and the estimate is scored against the true vocal stem.

**Run it:** `notebooks/exp1_interference.ipynb`, or the scripts below in order (all in `src/`; this folder is the description of the experiment).

| Step | File | What it does |
|---|---|---|
| 1 | `src/make_pairs_exp1.py` | builds the paired mixtures with `musdb` |
| 2 | `src/run_spleeter_exp1.py` | Spleeter 2-stem over the mixtures (originally run in Colab) |
| 2 | `src/run_demucs_exp1.sh` | Demucs 2-stem (`--two-stems=vocals`) over the mixtures |
| 3 | `src/evaluate_exp1.py` | scores the vocal estimates; writes `results/exp1/` |
| | `legacy/evaluate_si_metrics.py` | the original first-pass script (compares versions with a separated *original*); kept for reference, see its header |

**Two things are scored for every pairing.**
- **The vocal component** (what the write-up reports): the separated vocal against the true vocal stem. This measures how well the vocal is recovered when it is mixed with that one stem.
- **The non-vocal component**: the separated remainder (Spleeter `accompaniment.wav`, Demucs `no_vocals.wav`) against the true non-vocal stem (bass, drums, other, or drums+bass+other). It is in `results/exp1/` but not in the write-up.

**Metrics.** On mono downmixes: SI-SDR; an SI-SAR-like artifact ratio (`10 log10(||est||^2 / ||est - proj(est onto ref)||^2)`, a fast approximation, not BSS-Eval SAR); and RMSE (depends on level). The **median over the 50 songs** is reported.

**Where the write-up numbers come from.** They are the vocal-component medians. In the project repository they are the `Source = vocals` rows of `archive/Evaluation/Evaluation Results/Median Scores/exp1_summary_All_Results.csv` (from `evaluation_exp1_results_to_gt.xlsx`, "to ground truth"). `src/evaluate_exp1.py` reproduces that whole file from the saved separated outputs: SI-SDR and SI-SAR-like agree to within 0.001 dB on every row, RMSE within 0.00002. In the write-up, "Bass", "Drums" and so on name the pairing; the number is the recovered vocal's score.

**Vocal component (median over 50 songs; SI-SDR / SI-SAR-like in dB, RMSE unitless)**

| Pairing | Model | SI-SDR | SI-SAR-like | RMSE |
|---|---|---|---|---|
| Bass | Demucs | 20.94 | 20.98 | 0.00459 |
| Bass | Spleeter | 15.04 | 15.18 | 0.00868 |
| Drums | Demucs | 15.57 | 15.69 | 0.00762 |
| Drums | Spleeter | 11.78 | 12.05 | 0.01154 |
| Other | Demucs | 9.84 | 10.27 | 0.01384 |
| Other | Spleeter | 7.86 | 8.52 | 0.01804 |
| Accompaniment | Demucs | 8.75 | 9.29 | 0.01576 |
| Accompaniment | Spleeter | 6.50 | 7.38 | 0.02124 |

**Non-vocal component (same conventions)**

| Pairing (true stem it is scored against) | Model | SI-SDR | SI-SAR-like | RMSE |
|---|---|---|---|---|
| Bass | Demucs | 20.19 | 20.23 | 0.00451 |
| Bass | Spleeter | 14.76 | 14.90 | 0.00824 |
| Drums | Demucs | 16.98 | 17.06 | 0.00797 |
| Drums | Spleeter | 12.65 | 12.88 | 0.01279 |
| Other | Demucs | 11.01 | 11.34 | 0.01415 |
| Other | Spleeter | 9.13 | 9.63 | 0.01779 |
| Accompaniment | Demucs | 15.48 | 15.60 | 0.01671 |
| Accompaniment | Spleeter | 13.27 | 13.47 | 0.02153 |

Per-song scores are in `results/exp1/metrics.csv` (column `component`), the summary in `results/exp1/summary_median.csv`, and the figure (vocal component) in `results/exp1/`.

**The `legacy/` folder** holds two scripts from the project's `archive/Evaluation/` folder. They score **Spleeter against Demucs's output** (model agreement) and wrote the `demucs_vs_spleeter_*.xlsx` files. They were not used for the write-up table. See `legacy/README.md` for their limitations.

**Checks done.** A spot check of 20 Spleeter outputs (5 songs x 4 pairings) matched a clean fresh-per-file re-run to about 77 dB, so reusing one Separator was not a problem in this 2-stem run (it was in the 4-stem Experiment 2 run; see `src/run_spleeter_exp2.py`).

**Caveats.** "SI-SAR-like" and RMSE are non-standard, so define them clearly. The explanation that overlapping harmonic content causes the interference is a hypothesis and has not been tested. Say "median" wherever these numbers are reported.
