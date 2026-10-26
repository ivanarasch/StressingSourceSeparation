# archive

Material from the first version of this project (course work, first pass), kept for the record.
**Superseded.** The current code, data pipeline and results are in `src/`, `experiments/` and `results/`.

- `README_first_version.md`: the original write-up of the project.
- `Data/`, `Models/`, `Evaluation/`: the original data synthesis, model-running and evaluation code, and the first spreadsheets.

Notes:
- The Experiment 2 results here were produced before loudness was controlled (effects changed the vocal's loudness by
  several dB), so they should not be cited. See `results/exp2/` for the corrected results.
- Experiment 1's ground-truth results (`Evaluation/Evaluation Results/`) are the source of the numbers in the original
  write-up. They are reproduced by `src/evaluate_exp1.py`; see `experiments/exp1_interference/README.md`.

## Notes on the first-version evaluation scripts

- `Evaluation/eval_nonsigsep_exp1.py` (SI-SDR, SI-SAR-like, RMSE) and `Evaluation/eval_ref_optimized_exp1_sigsep.py`
  (BSS-Eval via `museval`) compare Spleeter's separated stems with **Demucs's separated stems, used as the reference**.
  They measure how much the two models agree, not how accurate either is, so their numbers differ from the
  ground-truth scores in the write-up. They produced the `demucs_vs_spleeter_*` spreadsheets in `Evaluation/Evaluation Results/`.
  The write-up's Experiment 1 table does not come from them. Known limitations:
  - The folder-matching pattern only recognizes `bass|drums|other|vocals`, so the `<song>_accompaniment` pairing is silently skipped.
  - The first component of each pair is the separated **vocal**, but it is labelled with the stem name (for example "bass");
    the second is the separated non-vocal part, labelled "accompaniment". The built-in summary groups by that label and mixes different quantities.
  - The `museval` script passes `win=1.0`. Check whether `museval.evaluate` expects seconds or samples before trusting its output.
  - They read Spleeter's nested folder layout only, and use hard-coded worker counts.
- `Evaluation/evaluate_si_metrics.py` is the earliest first-pass script. It compares each processed version's separated stems
  with the separated stems of the *original* (unprocessed) mixture, which is how the first version of Experiment 2 was scored.
  It was not used for the final results of either experiment.
