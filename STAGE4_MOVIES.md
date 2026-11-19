# Stage 4 movie classification

Andrew's contribution adds the required LSTM and GRU movie classifiers alongside Madison's RNN. The original team cleaning, vocabulary builder, dataset loader and base classes are reused unchanged. Madison's files remain under `ECS170_Fall_2026_Source_Code_Template/`.

## Data and settings

The original NLTK preprocessing cleans reviews and preserves negations. The team's stratified split uses seed 42: 22,500 training reviews, 2,500 validation reviews and 25,000 official test reviews. The vocabulary is built from training reviews only, with a maximum of 25,000 entries and minimum frequency 3. Reviews retain their first 300 cleaned tokens.

Both models use trainable 128-dimensional embeddings, one unidirectional recurrent layer with 128 hidden units, a linear binary classifier, no dropout, batch size 64, 10 epochs, Adam at learning rate 0.001, binary cross-entropy with logits and gradient clipping at norm 1.0. Training seed is 2. The checkpoint with highest validation accuracy is selected before evaluating the test set. These settings match the corresponding hyperparameters in Madison's RNN script.

## Run

`requirements-stage4.txt` lists the Python packages needed to run this code. Install them into your Python environment, then run from the repository root:

```bash
python -m pip install -r requirements-stage4.txt
python -m nltk.downloader -d ../nltk_data punkt punkt_tab stopwords
python script/stage_4_script/script_movie_prepare.py
python script/stage_4_script/script_movie_compare.py --models LSTM GRU
```

On Andrew's Mac, the environment is `/Users/andrew/ecs170/.venv/bin/python`.

The default raw data folder is `../stage_4_data/text_classification/`. The preparation script calls the team's original preprocessing functions and writes cleaned CSVs and `movie_encoded.pt` to `data/stage_4_data/text_classification/`. Use `--data-dir`, `--output-dir` and `--nltk-data` to change preparation paths. To reuse an existing encoding, pass `--data /path/to/movie_encoded.pt` to the comparison script.

The runner refuses to overwrite an existing checkpoint. Use a new `--output` directory for a repeat experiment. For a quick integration check, add `--smoke --epochs 1`; smoke results use small subsets and are not final experiment results.

## Results

Results are saved in `result/stage_4_movie/full/`.

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 | Selected epoch |
|---|---:|---:|---:|---:|---:|
| LSTM | 85.38% | 0.8538 | 0.8538 | 0.8538 | 10 |
| GRU | 85.88% | 0.8590 | 0.8588 | 0.8587 | 4 |

Each model's `lstm/` or `gru/` directory contains `history.csv`, `metrics.json` and `predictions.csv`. Training also saves a local `best.pt` checkpoint with weights, vocabulary and settings. The full results folder contains comparison tables, learning curves and confusion matrices. Both models show overfitting in their validation-loss curves.

Raw data, encoded tensors, smoke results and checkpoints are excluded from Git. `test_index` in the saved predictions refers to the zero-based row of `test_clean.csv`.

## Checks

`tests/test_movie_recurrent.py` contains three automated checks covering both models: agreement with packed-sequence outputs, independence from extra padding and batch context, and correct gradient propagation without updating the padding embedding. These are code checks, not extra training experiments.

```bash
python -m unittest discover -s tests -v
python local_code/stage_4_code/movie_test_dataset_loader.py
python script/stage_4_script/script_movie_verify_results.py
```

The first command runs our model checks, the second runs the team's loader checks, and the third recalculates final metrics from saved predictions and verifies checkpoint selection/reloading. Checkpoints must be present locally for the third command.

To predict a new review:

```bash
python script/stage_4_script/script_movie_predict.py \
  --checkpoint result/stage_4_movie/full/gru/best.pt \
  --text "The acting was excellent and I enjoyed the story."
```

Combine these movie results with Madison's RNN and the jokes team's results for the final assignment. Confirm matching preprocessing/data before comparing against the RNN; saved metrics include the encoded dataset's SHA-256 hash.
