# Stage 4 movie classification

Andrew's contribution implements and evaluates LSTM and GRU classifiers, extending Madison's basic movie RNN. The jokes models belong to the other task pair. All three movie architectures are required by Stage 4 step 4-5; the LSTM/GRU comparison is not an optional extra ablation.

## What is reused

The original `movie_clean_dataset.py`, `movie_build_vocab.py`, `Dataset_Loader_Movie.py`, and base classes are unchanged. Madison's RNN and training script remain unchanged under `ECS170_Fall_2026_Source_Code_Template/`. The new preprocessing entry point only sets input/output paths and invokes the teammates' existing functions; there is no replacement tokenizer.

The preprocessing expands negations, removes HTML/URLs, tokenizes with NLTK, and removes stopwords while retaining sentiment words such as `not` and `very`. The provided 25,000 training reviews are split into 22,500 train and 2,500 validation reviews using the team's stratified seed 42 split. The 25,000 official test reviews remain separate. The vocabulary uses only the training portion, caps the total size at 25,000, and requires frequency at least 3. Each review keeps its first 300 cleaned tokens. `<PAD>` is 0; `<UNK>` is 1.

## Architecture and controlled settings

| Setting | LSTM | GRU |
|---|---|---|
| Trainable embedding size | 128 | 128 |
| Recurrent hidden size | 128 | 128 |
| Recurrent layers | 1, unidirectional | 1, unidirectional |
| Classifier | Linear 128 → 1 logit | Linear 128 → 1 logit |
| Optimizer | Adam, learning rate 0.001 | Adam, learning rate 0.001 |
| Training | 10 epochs, batch size 64 | 10 epochs, batch size 64 |
| Loss | Binary cross-entropy with logits | Binary cross-entropy with logits |
| Gradient clipping | Maximum norm 1.0 | Maximum norm 1.0 |
| Dropout | 0 | 0 |
| Training seed | 2 | 2 |
| Checkpoint selection | Highest validation accuracy; first epoch wins ties | Same |

These match the corresponding settings in Madison's RNN script. Embeddings and recurrent weights use PyTorch default initialization; pretrained GloVe is not used. LSTM and GRU necessarily have different parameter counts because of their gates. The classification head uses the output at each review's true final token, not at the end of its padding. Unit tests verify numerical agreement with packed-sequence hidden states.

## Run the experiments

From this repository root, using the shared environment at `/Users/andrew/ecs170/.venv/bin/python`:

```bash
PYTHON=/Users/andrew/ecs170/.venv/bin/python
$PYTHON -m pip install -r requirements-stage4.txt
$PYTHON -m nltk.downloader -d ../nltk_data punkt punkt_tab stopwords
$PYTHON script/stage_4_script/script_movie_prepare.py
$PYTHON -m unittest discover -s tests -v
$PYTHON local_code/stage_4_code/movie_test_dataset_loader.py
$PYTHON script/stage_4_script/script_movie_compare.py --models LSTM GRU
```

The default raw data path is `../stage_4_data/text_classification/`; encoded files and cleaned CSVs are created in `data/stage_4_data/text_classification/`. Use `--data-dir`, `--output-dir`, or `--nltk-data` to change preprocessing paths. To use an existing team encoding, pass `--data /absolute/path/movie_encoded.pt` to the comparison script instead of rebuilding it.

A quick integration check can be run before the full experiment:

```bash
$PYTHON script/stage_4_script/script_movie_compare.py --models LSTM GRU --smoke --epochs 1
```

Smoke runs use only 256 training, 128 validation and 128 test reviews. They are written under `smoke/`, marked in every metrics JSON, and must not be used as final report results. Full experiments always use the entire encoded dataset and are written under `full/`.

The training script refuses to overwrite an existing checkpoint. To repeat an experiment, use a new output directory, for example `--output result/stage_4_movie_repeat`. The saved `requirements-stage4-lock.txt` records the environment used for this work; exact package versions and hardware can affect numerical reproducibility.

## Outputs

The full results are in `result/stage_4_movie/full/`:

| Output | Contents |
|---|---|
| `lstm/` and `gru/` | Model-specific results |
| `best.pt` in each model directory | Selected model weights, vocabulary, model settings and selected epoch |
| `history.csv` in each model directory | Per-epoch training loss/accuracy, validation loss/accuracy and duration |
| `metrics.json` in each model directory | Test accuracy, macro precision/recall/F1, per-class metrics, confusion matrix, dataset hash and configuration |
| `predictions.csv` in each model directory | Test row index, true/predicted labels and positive probability |
| `comparison.csv`, `comparison.json`, `comparison.md` | Combined LSTM/GRU results |
| `learning_curves.png` | Training and validation cross-entropy versus epoch |
| `confusion_matrices.png` | Counts with true labels on rows and predictions on columns |

`test_index` in `predictions.csv` is the zero-based row of `test_clean.csv`, which contains each review's original ID, rating and label. Raw data, encoded tensors, smoke runs and model checkpoints are excluded from Git to keep the repository small. Small full-run result tables and plots can be committed. The checkpoint files remain available locally.

To classify a new review:

```bash
$PYTHON script/stage_4_script/script_movie_predict.py \
  --checkpoint result/stage_4_movie/full/gru/best.pt \
  --text "The acting was excellent and I enjoyed the story."
```

## Build the report contribution

After the full runs finish:

```bash
$PYTHON script/stage_4_script/script_movie_report.py
```

This creates `report/movie_lstm_gru/movie_lstm_gru_report.md` and a self-contained HTML version with actual results and figures, following the instructor's section headings. It refuses smoke results or models trained on different encoded datasets.

Verify completed experiments with:

```bash
$PYTHON script/stage_4_script/script_movie_verify_results.py
```

This independently recalculates all four test metrics from saved predictions and checks the split sizes, validation-selected epochs and reloaded checkpoints.

## Team integration

The report contribution is organized by the instructor's Sections 1, 2 and 3.1–3.7. It covers only the movie LSTM/GRU work. Combine it with Madison's RNN results and Sai/Dani's generation results in the shared report, observing the Stage 4 assignment's stricter five-page limit. Confirm that Madison uses the same encoded dataset and preprocessing before combining numbers; `dataset_sha256` in the saved metrics provides an exact identity check when using the shared runner. No RNN results are invented or attributed to Madison.

If a common runner is needed for all three movie models, it also supports `--models RNN LSTM GRU`. This is an optional way to standardize the comparison; it is not necessary to duplicate Madison's experiment if she already has compatible results.
