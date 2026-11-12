'''
Test for the stage 4 dataset loader (Dataset_Loader_Movie.py).
Run after movie_build_vocab.py. Takes no parameters.
Prints [PASS] / [FAIL] lines and statistics.
'''

import os
import sys

# Project root = two levels above this script (<root>/local_code/stage_4_code/).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from local_code.stage_4_code.Dataset_Loader_Movie import Dataset_Loader, MoviesDataset

DATA_DIR = os.path.join(ROOT, 'data', 'stage_4_data', 'text_classification')
FILE_NAME = 'movie_encoded.pt'
BATCH_SIZE = 64

EXPECTED_SIZES = {'train': 22500, 'val': 2500, 'test': 25000}

results = []


def check(name, condition, detail=''):
    results.append(bool(condition))
    tag = '[PASS]' if condition else '[FAIL]'
    print(f"{tag} {name}" + (f"  ({detail})" if detail else ''))


def main():
    loader = Dataset_Loader('movie reviews', 'IMDb encoded reviews')
    loader.dataset_source_folder_path = DATA_DIR
    loader.dataset_source_file_name = FILE_NAME
    datasets = loader.load()

    print('\n== loader attributes ==')
    print(f"vocab_size={loader.vocab_size}  max_len={loader.max_len}  "
          f"pad_idx={loader.pad_idx}  unk_idx={loader.unk_idx}")
    check('vocab_size is 25000', loader.vocab_size == 25000, str(loader.vocab_size))
    check('max_len is 300', loader.max_len == 300, str(loader.max_len))
    check('pad_idx 0 and unk_idx 1', loader.pad_idx == 0 and loader.unk_idx == 1)

    print('\n== load() return value ==')
    check('load() returns a dict', isinstance(datasets, dict))
    check("keys are train, val, test", set(datasets.keys()) == {'train', 'val', 'test'},
          str(sorted(datasets.keys())))
    for split, ds in datasets.items():
        check(f"{split}: MoviesDataset and torch Dataset",
              isinstance(ds, MoviesDataset) and isinstance(ds, Dataset))
        check(f"{split}: size {EXPECTED_SIZES[split]}", len(ds) == EXPECTED_SIZES[split],
              str(len(ds)))

    print('\n== __getitem__ ==')
    for split, ds in datasets.items():
        x, length, y = ds[0]
        check(f"{split}: x int32 shape [{loader.max_len}]",
              x.dtype == torch.int32 and tuple(x.shape) == (loader.max_len,),
              f"{x.dtype}, {tuple(x.shape)}")
        check(f"{split}: length int64 scalar, 1..max_len",
              length.dtype == torch.int64 and length.dim() == 0 and 1 <= int(length) <= loader.max_len,
              f"{length.dtype}, {int(length)}")
        check(f"{split}: y int64 scalar, 0 or 1",
              y.dtype == torch.int64 and y.dim() == 0 and int(y) in (0, 1),
              f"{y.dtype}, {int(y)}")

    print('\n== right padding (first 1000 reviews of each split) ==')
    for split, ds in datasets.items():
        ok = True
        for i in range(min(1000, len(ds))):
            x, length, _ = ds[i]
            n = int(length)
            if (x[:n] == loader.pad_idx).any() or (x[n:] != loader.pad_idx).any():
                ok = False
                break
        check(f"{split}: real tokens first, <PAD> only at the end", ok)

    print('\n== label balance ==')
    for split, ds in datasets.items():
        pos = int(ds.y.sum())
        neg = len(ds) - pos
        print(f"{split}: pos={pos} neg={neg}")
        check(f"{split}: balanced", pos == neg)

    print('\n== sample review ==')
    x, length, y = datasets['train'][0]
    words = [loader.itos[int(t)] for t in x[:int(length)]]
    print(f"label={int(y)} length={int(length)}")
    print(' '.join(words[:40]) + (' ...' if len(words) > 40 else ''))

    print('\n== collate_fn (DataLoader, batch size 64) ==')
    train_loader = DataLoader(datasets['train'], batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=MoviesDataset.collate_fn)
    x, lengths, y = next(iter(train_loader))
    print(f"x {tuple(x.shape)} {x.dtype} | lengths {tuple(lengths.shape)} {lengths.dtype} "
          f"{lengths.device} | y {tuple(y.shape)} {y.dtype}")
    check('x is int64 [B, T]', x.dtype == torch.int64 and x.dim() == 2 and x.shape[0] == BATCH_SIZE)
    check('T equals the longest review in the batch', x.shape[1] == int(lengths.max()),
          f"T={x.shape[1]}, max length={int(lengths.max())}")
    check('T <= max_len', x.shape[1] <= loader.max_len)
    check('lengths int64 [B] on CPU',
          lengths.dtype == torch.int64 and tuple(lengths.shape) == (BATCH_SIZE,)
          and lengths.device.type == 'cpu')
    check('y int64 [B] with values in {0, 1}',
          y.dtype == torch.int64 and tuple(y.shape) == (BATCH_SIZE,) and bool(((y == 0) | (y == 1)).all()))
    pad_ok = all(bool((x[i, int(lengths[i]):] == loader.pad_idx).all()) and
                 bool((x[i, :int(lengths[i])] != loader.pad_idx).all())
                 for i in range(x.shape[0]))
    check('batch is right-padded with <PAD>', pad_ok)

    # One full pass over train: every review appears exactly once, trimming never cuts real tokens.
    total = 0
    pos_total = 0
    trim_ok = True
    for xb, lb, yb in train_loader:
        total += xb.shape[0]
        pos_total += int(yb.sum())
        if xb.shape[1] != int(lb.max()):
            trim_ok = False
    check('one pass over train covers all reviews', total == EXPECTED_SIZES['train'], str(total))
    check('one pass over train keeps the label counts',
          pos_total == int(datasets['train'].y.sum()), str(pos_total))
    check('every batch trimmed to its longest review', trim_ok)

    # Same check for val and test without shuffling.
    for split in ('val', 'test'):
        dl = DataLoader(datasets[split], batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=MoviesDataset.collate_fn)
        count = sum(xb.shape[0] for xb, _, _ in dl)
        check(f"{split}: one pass covers all reviews", count == EXPECTED_SIZES[split], str(count))

    print('\n== nn.Embedding + pack_padded_sequence smoke check ==')
    embedding = torch.nn.Embedding(loader.vocab_size, 32, padding_idx=loader.pad_idx)
    emb = embedding(x)
    check('embedding output [B, T, 32]', tuple(emb.shape) == (BATCH_SIZE, x.shape[1], 32),
          str(tuple(emb.shape)))
    packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
    check('packed data has one row per real token',
          packed.data.shape[0] == int(lengths.sum()),
          f"{packed.data.shape[0]} rows, {int(lengths.sum())} real tokens")
    unpacked, out_lengths = pad_packed_sequence(packed, batch_first=True)
    check('pad_packed_sequence restores lengths and shape',
          bool((out_lengths == lengths).all()) and unpacked.shape[0] == BATCH_SIZE
          and unpacked.shape[1] == int(lengths.max()))

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    if passed != len(results):
        sys.exit(1)


if __name__ == '__main__':
    main()
