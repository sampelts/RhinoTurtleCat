"""Check saved full-run metrics, predictions, checkpoint selection and split identity."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from local_code.stage_4_code.Method_Recurrent_Movie import MovieRecurrent
from local_code.stage_4_code.Dataset_Loader_Movie import Dataset_Loader, MoviesDataset
from torch.utils.data import DataLoader


def verify(results, encoded):
    combined = json.loads((results / 'comparison.json').read_text())
    assert len(combined) == 2 and {m['cell'] for m in combined} == {'LSTM', 'GRU'}
    assert len({m['dataset_sha256'] for m in combined}) == 1, 'Dataset hashes differ'
    loader = Dataset_Loader('IMDb', 'verification')
    loader.dataset_source_folder_path = str(encoded.parent)
    loader.dataset_source_file_name = encoded.name
    datasets = loader.load()
    batch = next(iter(DataLoader(datasets['test'], batch_size=64, shuffle=False,
                                collate_fn=MoviesDataset.collate_fn)))
    torch.set_num_threads(2)
    for metric in combined:
        folder = results / metric['cell'].lower()
        rows = list(csv.DictReader((folder / 'predictions.csv').open()))
        assert not metric['smoke_run'] and len(rows) == 25000
        assert metric['used_split_sizes'] == {'train': 22500, 'val': 2500, 'test': 25000}
        assert [int(row['test_index']) for row in rows] == list(range(25000))
        truth = np.array([int(row['true_label']) for row in rows])
        predicted = np.array([int(row['predicted_label']) for row in rows])
        probability = np.array([float(row['positive_probability']) for row in rows])
        assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
        np.testing.assert_array_equal(predicted, probability >= .5)
        np.testing.assert_array_equal(truth, datasets['test'].y.numpy())
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, labels=[0, 1], average='macro', zero_division=0)
        for key, value in [('accuracy', accuracy_score(truth, predicted)),
                           ('precision_macro', precision), ('recall_macro', recall), ('f1_macro', f1)]:
            assert abs(metric[key] - value) < 1e-12, (metric['cell'], key)
        assert metric['confusion_matrix'] == confusion_matrix(truth, predicted, labels=[0, 1]).tolist()
        history = list(csv.DictReader((folder / 'history.csv').open()))
        assert [int(row['epoch']) for row in history] == list(range(1, 11))
        best = max(history, key=lambda row: float(row['val_accuracy']))
        assert int(best['epoch']) == metric['best_epoch']
        checkpoint = torch.load(folder / 'best.pt', weights_only=True, map_location='cpu')
        assert checkpoint['epoch'] == metric['best_epoch']
        model = MovieRecurrent(**checkpoint['model_config']).eval()
        model.load_state_dict(checkpoint['model_state'])
        with torch.no_grad():
            actual_probability = torch.sigmoid(model(batch[0], batch[1]).double()).numpy()
        np.testing.assert_allclose(actual_probability, probability[:64], atol=1e-6, rtol=1e-5)
        print(f"PASS {metric['cell']}: 25,000 predictions reproduce metrics; 10 epochs; "
              f"validation-selected checkpoint reloads correctly.")
    print('PASS both runs use the same full dataset and preserved held-out test labels.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'result/stage_4_movie/full')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/stage_4_data/text_classification/movie_encoded.pt')
    args = parser.parse_args()
    verify(args.results, args.data)
