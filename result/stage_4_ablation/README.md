# Stage 4 ablations: GRU hidden size and depth (report 3.7.3 and 3.7.4)

Produced by `script/stage_4_script/run_stage4_ablations.py` with the team's unchanged Stage-4 code: `script_movie_compare.py --models GRU` (movie) and `script_joke_generation.py --cell GRU` (jokes). Each configuration changes one setting of the GRU baseline; the baseline was re-trained in the same environment. Checkpoints and per-review predictions are not included.

### 3.7.3 GRU Movie Classification: Hidden Size and Depth

| GRU setting | Test accuracy | Macro F1 | Best epoch | Parameters |
|---|---|---|---|---|
| Baseline (128 hidden, 1 layer) | 86.41% | 0.8640 | 4 | 3,299,201 |
| Hidden size 64 | 85.47% | 0.8546 | 6 | 3,237,313 |
| Hidden size 256 | 86.45% | 0.8645 | 10 | 3,496,705 |
| 2 GRU layers | 86.19% | 0.8619 | 8 | 3,398,273 |

Starting from the GRU classifier in 3.2 and changing one setting at a time (all other settings, the data split and seed 2 unchanged; the baseline was re-trained in the same environment), a hidden size of 64 gave 85.47% test accuracy (-0.9 pts vs. the baseline's 86.41%), a hidden size of 256 gave 86.45% (+0.0 pts), and a second GRU layer gave 86.19% (-0.2 pts). The best setting was: Hidden size 256. Each setting was trained once, so differences of a few tenths of a point are within run-to-run variation.

### 3.7.4 GRU Joke Generation: Hidden Size and Depth

| GRU setting | Test perplexity | Next-word accuracy | Best epoch | Parameters |
|---|---|---|---|---|
| Baseline (256 hidden, 1 layer) | 57.43 | 34.65% | 7 | 1,844,533 |
| Hidden size 128 | 58.38 | 34.93% | 14 | 1,132,469 |
| Hidden size 512 | 59.24 | 34.53% | 5 | 3,563,573 |
| 2 GRU layers | 59.96 | 34.57% | 11 | 2,239,285 |

Starting from the GRU joke generator (256 hidden units, one layer) and changing one setting at a time, a hidden size of 128 gave a test perplexity of 58.38 (+0.95 vs. the baseline's 57.43), a hidden size of 512 gave 59.24 (+1.81), and a second GRU layer gave 59.96 (+2.53); lower is better. The best setting was: Baseline (256 hidden, 1 layer). With only 1,269 training jokes, each run was trained once, so small differences should be read with caution.
