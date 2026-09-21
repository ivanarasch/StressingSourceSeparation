# legacy: Spleeter-vs-Demucs agreement scripts

`eval_nonsigsep_exp1.py` (SI-SDR, SI-SAR-like, RMSE) and `eval_ref_optimized_exp1_sigsep.py` (BSS-Eval via `museval`) compare Spleeter's separated stems with **Demucs's separated stems, used as the reference**. They measure how much the two models agree, not how accurate either is, so their numbers differ from the ground-truth scores in the write-up. They produced the `demucs_vs_spleeter_*` spreadsheets in the project's `archive/Evaluation/Evaluation Results` folder. The write-up's Experiment 1 table does not come from them (see `../README.md`).

Kept for the record and credit to their author. Known limitations, if you reuse them:
- The folder-matching pattern only recognizes `bass|drums|other|vocals`, so the `<song>_accompaniment` pairing is silently skipped.
- The first component of each pair is the separated **vocal**, but it is labelled with the stem name (for example "bass"); the second is the separated non-vocal part, labelled "accompaniment". The built-in summary groups by that label and mixes different quantities.
- The `museval` script passes `win=1.0`. Check whether `museval.evaluate` expects seconds or samples before trusting its output (a window of one sample would be meaningless).
- They read Spleeter's nested layout only, and use hard-coded worker counts.

`evaluate_si_metrics.py` is the earliest first-pass script. It compares each processed version's separated stems with the separated stems of the *original* (unprocessed) mixture, which is how the first version of Experiment 2 was scored. It was not used for the final results of either experiment.
