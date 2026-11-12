#!/usr/bin/env python3
"""Test Dataset_Loader_Joke.py (the joke loader) with [PASS] / [FAIL] lines.

Usage:
    python joke_test_dataset_loader.py

Run joke_build_vocab.py first so that joke_encoded.pt exists. This script sits
in <project root>/local_code/stage_4_code/ next to Dataset_Loader_Joke.py; it
adds the project root to sys.path so that `local_code...` imports work outside
PyCharm too. No model is built; the nn.Embedding + pack_padded_sequence +
CrossEntropyLoss part only checks that a batch can be fed to a recurrent model.

Parts:
    1. loader and dataset basics
    2. batch format on val / test (no <UNK> replacement) compared with the raw data
    3. training split with the default settings (inputs-only <UNK> replacement)
    4. the switch: replace_targets=True, and unk_prob=0 (off)
    5. quick PyTorch compatibility check (embedding, packing, loss)
Exits with code 1 if any check fails.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn.utils.rnn import pack_padded_sequence  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from Dataset_Loader_Joke import (  # noqa: E402
    Dataset_Loader, JokesDataset, RARE_MAX_COUNT, UNK_PROB)

BATCH_SIZE = 64
RATE_TOLERANCE = 0.05
DATA_DIR = str((Path(__file__).resolve().parent
                / "../../data/stage_4_data/text_generation").resolve())


def make_loader(name, description="encoded joke dataset", **settings):
    """Create a Dataset_Loader and set its source path the way the course scripts do."""
    ld = Dataset_Loader(name, description, **settings)
    ld.dataset_source_folder_path = DATA_DIR
    ld.dataset_source_file_name = "joke_encoded.pt"
    return ld

n_pass = n_fail = 0


def check(label, ok, detail=""):
    global n_pass, n_fail
    ok = bool(ok)
    if ok:
        n_pass += 1
    else:
        n_fail += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    return ok


def structure_checks(inputs, targets, lengths, ids, check_shift=True):
    """Format checks for one batch. Returns {description: bool}."""
    pad, bos, eos = ids
    b, t = inputs.shape
    ar = torch.arange(t).unsqueeze(0)
    pad_mask = ar >= lengths.unsqueeze(1)
    res = {}
    res["inputs, targets, lengths are int64 and lengths is on CPU"] = (
        inputs.dtype == torch.int64 and targets.dtype == torch.int64
        and lengths.dtype == torch.int64 and lengths.device.type == "cpu")
    res["inputs and targets are [B, T] with T = longest joke in the batch"] = (
        inputs.shape == targets.shape and lengths.shape == (b,) and t == int(lengths.max()))
    res["every input row starts with <BOS>"] = bool((inputs[:, 0] == bos).all())
    res["no <EOS> in inputs; one <EOS> per target row, at position lengths-1"] = bool(
        (inputs == eos).sum() == 0
        and (targets[torch.arange(b), lengths - 1] == eos).all()
        and ((targets == eos).sum(1) == 1).all())
    res["positions >= lengths are <PAD>, real positions are not <PAD> (inputs and targets)"] = bool(
        (inputs[pad_mask] == pad).all() and (targets[pad_mask] == pad).all()
        and (inputs[~pad_mask] != pad).all() and (targets[~pad_mask] != pad).all())
    if check_shift:
        valid = ar[:, :t - 1] < (lengths - 1).unsqueeze(1)
        res["targets are the inputs shifted by one (targets[:, j] == inputs[:, j+1])"] = bool(
            (inputs[:, 1:][valid] == targets[:, :-1][valid]).all())
    return res


def clean_reference(data, split, start, n, pad):
    """What a batch of rows start..start+n should look like without any replacement."""
    x = data[split]["x"][start:start + n].long()
    lengths = data[split]["lengths"][start:start + n] - 1
    t = int(lengths.max())
    inputs = x[:, :t].clone()
    inputs[torch.arange(t).unsqueeze(0) >= lengths.unsqueeze(1)] = pad
    targets = x[:, 1:t + 1].clone()
    return inputs, targets, lengths


def run_pass(dataset_, shuffle=False):
    dl = DataLoader(dataset_, batch_size=BATCH_SIZE, shuffle=shuffle,
                    collate_fn=JokesDataset.collate_fn)
    return list(dl)


def all_structure(name, batches, ids, check_shift=True):
    ok = {}
    for inputs, targets, lengths in batches:
        for k, v in structure_checks(inputs, targets, lengths, ids, check_shift).items():
            ok[k] = ok.get(k, True) and v
    for k, v in ok.items():
        check(f"{name}: {k}", v)


def main():
    torch.manual_seed(0)

    # ------------------------------------------------------------------
    print("-- 1. Loader and dataset basics")
    loader = make_loader("jokes")
    sets = loader.load()
    data = loader.data
    ids = (loader.pad_idx, loader.bos_idx, loader.eos_idx)
    pad, bos, eos = ids
    unk = loader.unk_idx

    check("load() returns a dict with keys train, val, test",
          isinstance(sets, dict) and set(sets) == {"train", "val", "test"})
    check("each dataset has as many items as its split in the file",
          all(len(sets[k]) == len(data[k]["lengths"]) for k in sets),
          ", ".join(f"{k} {len(sets[k])}" for k in ("train", "val", "test")))
    check("loader attributes (vocab_size, max_len, seq_len, special indices, stoi, itos) are set",
          loader.vocab_size == len(loader.itos) == len(loader.stoi)
          and loader.max_len == data["config"]["max_len"]
          and loader.seq_len == data["config"]["seq_len"]
          and (pad, unk, bos, eos) == (0, 1, 2, 3),
          f"vocab_size {loader.vocab_size}, max_len {loader.max_len}, seq_len {loader.seq_len}")
    item = sets["val"][0]
    check("an item is (input_seq, target_seq, length) with int64 sequences of width seq_len",
          len(item) == 3 and item[0].dtype == torch.int64 and item[1].dtype == torch.int64
          and tuple(item[0].shape) == (loader.seq_len,) and item[2].dtype == torch.int64
          and item[2].dim() == 0)

    # ------------------------------------------------------------------
    print("\n-- 2. Batch format on val / test (never modified)")
    for split in ("val", "test"):
        batches = run_pass(sets[split])
        check(f"{split}: batches cover every joke exactly once",
              sum(len(b[2]) for b in batches) == len(sets[split]),
              f"{len(batches)} batches")
        all_structure(split, batches, ids)
        same = True
        start = 0
        for inputs, targets, lengths in batches:
            ci, ct, cl = clean_reference(data, split, start, len(lengths), pad)
            same &= bool(torch.equal(inputs, ci) and torch.equal(targets, ct)
                         and torch.equal(lengths, cl))
            start += len(lengths)
        check(f"{split}: batches equal the raw encoded data (inputs = x[:, :-1], targets = x[:, 1:])", same)
    print("   decoded first val joke:")
    vb = run_pass(sets["val"])[0]
    print(f"     input : {loader.decode(vb[0][0])}")
    print(f"     target: {loader.decode(vb[1][0])}")

    # ------------------------------------------------------------------
    print(f"\n-- 3. Training split, default settings (unk_prob {UNK_PROB}, "
          f"rare_max_count {RARE_MAX_COUNT}, inputs only)")
    word_counts = data["word_counts"]
    rare = torch.zeros(loader.vocab_size, dtype=torch.bool)
    for idx in range(4, loader.vocab_size):
        rare[idx] = word_counts[loader.itos[idx]] <= RARE_MAX_COUNT
    print(f"   {int(rare.sum())} of {loader.vocab_size} vocabulary entries are eligible "
          f"(seen at most {RARE_MAX_COUNT} time(s) in train)")

    def analyse_train(batches, targets_replaced):
        """Compare an unshuffled training pass with the clean data."""
        n_elig = n_changed = 0
        only_unk = only_rare = targets_clean = bos_ok = True
        sample = None
        start = 0
        for inputs, targets, lengths in batches:
            ci, ct, _ = clean_reference(data, "train", start, len(lengths), pad)
            start += len(lengths)
            changed_i = inputs != ci
            if targets_replaced:
                changed = targets != ct
                ref, noisy = ct, targets
            else:
                changed = changed_i
                ref, noisy = ci, inputs
                targets_clean &= bool(torch.equal(targets, ct))
            only_unk &= bool((noisy[changed] == unk).all())
            only_rare &= bool(rare[ref[changed]].all())
            bos_ok &= bool((inputs[:, 0] == bos).all())
            n_elig += int(rare[ref].sum())
            n_changed += int(changed.sum())
            if sample is None and changed_i.any():
                r = int(changed_i.any(1).nonzero()[0])
                sample = (ci[r], inputs[r], ct[r], targets[r])
        rate = n_changed / max(n_elig, 1)
        return only_unk, only_rare, targets_clean, bos_ok, rate, n_elig, sample

    batches1 = run_pass(sets["train"])
    all_structure("train", batches1, ids, check_shift=False)
    only_unk, only_rare, targets_clean, bos_ok, rate, n_elig, sample = analyse_train(batches1, False)
    check("train: only eligible rare words are replaced, and only by <UNK>", only_unk and only_rare)
    check("train: <BOS>, <EOS>, <PAD>, <UNK> and frequent words and punctuation are untouched", only_rare and bos_ok)
    check("train: targets are identical to the clean data (inputs-only mode)", targets_clean)
    check(f"train: about {UNK_PROB:.0%} of eligible tokens are replaced",
          abs(rate - UNK_PROB) < RATE_TOLERANCE, f"{rate:.1%} of {n_elig} eligible tokens")
    batches2 = run_pass(sets["train"])
    differs = any(not torch.equal(a[0], b[0]) for a, b in zip(batches1, batches2))
    check("train: the replacement is random, a second pass over the data differs", differs)
    if sample is not None:
        print("   example row (clean input -> noisy input; target stays clean):")
        print(f"     clean input : {loader.decode(sample[0])}")
        print(f"     noisy input : {loader.decode(sample[1])}")
        print(f"     target      : {loader.decode(sample[3])}")

    # ------------------------------------------------------------------
    print("\n-- 4. The switches")
    loader_t = make_loader("jokes", "targets replaced too", replace_targets=True)
    sets_t = loader_t.load()
    batches_t = run_pass(sets_t["train"])
    all_structure("replace_targets=True, train", batches_t, ids, check_shift=True)
    only_unk, only_rare, _, bos_ok, rate, n_elig, sample = analyse_train(batches_t, True)
    check("replace_targets=True: only rare words are replaced in the targets, only by <UNK>",
          only_unk and only_rare)
    check(f"replace_targets=True: about {UNK_PROB:.0%} of eligible target tokens are replaced",
          abs(rate - UNK_PROB) < RATE_TOLERANCE, f"{rate:.1%} of {n_elig} eligible tokens")

    loader_off = make_loader("jokes", "no replacement", unk_prob=0)
    sets_off = loader_off.load()
    batches_off = run_pass(sets_off["train"])
    off_ok = True
    start = 0
    for inputs, targets, lengths in batches_off:
        ci, ct, _ = clean_reference(data, "train", start, len(lengths), pad)
        off_ok &= bool(torch.equal(inputs, ci) and torch.equal(targets, ct))
        start += len(lengths)
    check("unk_prob=0: training batches equal the raw encoded data (replacement is off)", off_ok)

    # ------------------------------------------------------------------
    print("\n-- 5. PyTorch compatibility (no model)")
    inputs, targets, lengths = batches1[0]
    emb = nn.Embedding(loader.vocab_size, 16, padding_idx=pad)
    out = emb(inputs)
    check("nn.Embedding accepts the inputs", tuple(out.shape) == (*inputs.shape, 16))
    packed = pack_padded_sequence(out, lengths, batch_first=True, enforce_sorted=False)
    check("pack_padded_sequence accepts lengths (enforce_sorted=False)",
          packed.data.shape[0] == int(lengths.sum()), f"{packed.data.shape[0]} packed steps")
    logits = torch.randn(*inputs.shape, loader.vocab_size)
    ce = nn.CrossEntropyLoss(ignore_index=pad)
    loss = ce(logits.reshape(-1, loader.vocab_size), targets.reshape(-1))
    check("CrossEntropyLoss(ignore_index=<PAD>) works on the targets",
          torch.isfinite(loss) and int((targets != pad).sum()) == int(lengths.sum()),
          f"loss on random logits {float(loss):.2f}")

    print(f"\nSummary: {n_pass} passed, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
