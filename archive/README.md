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
- `Evaluation/eval_nonsigsep_exp1.py` and `eval_ref_optimized_exp1_sigsep.py` compare Spleeter with Demucs's output
  (model agreement) and were not used for the write-up table; see `experiments/exp1_interference/legacy/README.md`.
