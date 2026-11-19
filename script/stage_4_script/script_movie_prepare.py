"""Run the teammates' original movie preprocessing with configurable paths."""
import argparse
import csv
import importlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT.parent / 'stage_4_data/text_classification')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/stage_4_data/text_classification')
    parser.add_argument('--nltk-data', type=Path, default=ROOT.parent / 'nltk_data')
    args = parser.parse_args()
    source = args.data_dir.resolve()
    output = args.output_dir.resolve()
    encoded = output / 'movie_encoded.pt'
    if encoded.exists():
        parser.error(f'{encoded} already exists; use a new --output-dir to retain it.')
    import nltk
    nltk.data.path.insert(0, str(args.nltk_data.resolve()))
    cleaner = importlib.import_module('local_code.stage_4_code.movie_clean_dataset')
    try:
        stop_words = cleaner.build_stop_words()
        cleaner.clean_text('A very good film. I did not dislike it.', stop_words)
    except LookupError:
        parser.error('NLTK resources are missing. Download punkt, punkt_tab and stopwords first.')
    output.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'test'):
        print(f'Cleaning {split} using the original team NLTK pipeline...', flush=True)
        df = cleaner.load_split(source, split, stop_words)
        cleaner.report(df, split)
        df[['id', 'rating', 'label', 'text']].to_csv(
            output / f'{split}_clean.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
    builder = importlib.import_module('local_code.stage_4_code.movie_build_vocab')
    builder.DATA_DIR = output
    builder.OUT_PATH = encoded
    builder.main()
    import torch
    payload = torch.load(encoded, weights_only=True)
    stats = dict(tokenizer='original_team_nltk_word_tokenize', vocabulary_size=len(payload['vocab']['itos']),
                 config=payload['config'], splits={})
    for name in ('train', 'val', 'test'):
        data = payload[name]
        if bool((data['lengths'] < 1).any()):
            raise ValueError(f'{name} contains empty encoded reviews; do not start training.')
        stats['splits'][name] = dict(n=len(data['y']), positive=int(data['y'].sum()),
            negative=int((data['y'] == 0).sum()), mean_length=float(data['lengths'].float().mean()),
            unknown_fraction=float((data['x'] == 1).sum()) / int(data['lengths'].sum()))
    encoded.with_suffix('.json').write_text(json.dumps(stats, indent=2) + '\n')


if __name__ == '__main__':
    main()
