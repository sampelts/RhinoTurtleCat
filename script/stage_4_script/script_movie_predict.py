"""Predict the sentiment of a new review using a saved LSTM or GRU checkpoint."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import nltk
from local_code.stage_4_code.Method_Recurrent_Movie import MovieRecurrent
from local_code.stage_4_code.movie_clean_dataset import build_stop_words, clean_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--nltk-data', type=Path, default=ROOT.parent / 'nltk_data')
    args = parser.parse_args()
    nltk.data.path.insert(0, str(args.nltk_data.resolve()))
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = MovieRecurrent(**saved['model_config'])
    model.load_state_dict(saved['model_state'])
    model.eval()
    tokens = clean_text(args.text, build_stop_words())[:saved['data_config']['max_len']]
    vocab = saved['vocab']['stoi']
    ids = [vocab.get(token, 1) for token in tokens] or [1]
    with torch.no_grad():
        logit = model(torch.tensor([ids]), torch.tensor([len(ids)]))
        probability = float(torch.sigmoid(logit).item())
    print(f"Sentiment: {'positive' if probability >= .5 else 'negative'}")
    print(f'Positive probability: {probability:.4f}')
    print(f'Tokens used: {len(ids)}; unknown tokens: {ids.count(1)}')


if __name__ == '__main__':
    main()
