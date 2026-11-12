'''
Concrete IO class for the stage 4 text generation dataset (jokes)
'''

# Follows the pattern of Dataset_Loader_Movie.py (stage 4 text classification).
# License: TBD

from local_code.base_class.dataset import dataset
from torch.utils.data import Dataset
import torch
import os

# ---------------------------------------------------------------------------
# Defaults for the random <UNK> replacement (see JokesDataset)
# ---------------------------------------------------------------------------
UNK_PROB = 0.5
RARE_MAX_COUNT = 1
UNK_REPLACE_TARGETS = False

PAD_IDX = 0  # checked against the saved file's config in Dataset_Loader.load()

DEFAULT_FOLDER_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'stage_4_data', 'text_generation'))
DEFAULT_FILE_NAME = 'joke_encoded.pt'


class JokesDataset(Dataset):
    '''
    One split (train, val or test) of the encoded jokes.

    __getitem__(i) returns one joke as a tuple (input_seq, target_seq, length):
        input_seq:  int64 tensor [seq_len], <BOS> w1 ... wn <EOS> then <PAD>
        target_seq: int64 tensor [seq_len], the same row (see below for the training split)
        length:     int64 scalar tensor, number of real tokens including <BOS> and <EOS>

    Use JokesDataset.collate_fn as the collate_fn of a torch DataLoader:

        loader = DataLoader(datasets['train'], batch_size=64, shuffle=True,
                            collate_fn=JokesDataset.collate_fn)
        for inputs, targets, lengths in loader:
            ...

    Every batch is a tuple (inputs, targets, lengths):
        inputs:  int64 tensor [B, T], <BOS> w1 ... wn, right-padded with <PAD> (index 0)
        targets: int64 tensor [B, T], w1 ... wn <EOS>, right-padded with <PAD>;
                 targets[:, t] is the token that follows inputs[:, t]
        lengths: int64 tensor [B] on CPU, number of real positions per row of inputs /
                 targets (joke length including <BOS>/<EOS>, minus 1); ready for
                 torch.nn.utils.rnn.pack_padded_sequence
        T is the number of real positions of the longest joke in the batch (at most
        seq_len - 1). Positions >= lengths are <PAD> in both tensors, so use
        CrossEntropyLoss(ignore_index=0) (or mask with lengths).

    Random <UNK> replacement (training split only):
        With min_freq = 1 the model never sees <UNK> in training, yet about 12% of the
        val / test tokens are <UNK>. So each time a training joke is fetched, every word
        that occurs at most rare_max_count times in train is replaced by <UNK> with
        probability unk_prob (different every epoch). <PAD>, <BOS>, <EOS>, <UNK>,
        frequent words and punctuation are never replaced; val / test are never changed.
        replace_targets=False: only the inputs get <UNK>, the targets keep the real word.
        replace_targets=True:  inputs and targets are replaced the same way, as in val / test
        (then block <UNK> when generating). The three settings are chosen in Dataset_Loader.
    '''

    def __init__(self, x, lengths, unk_prob_table=None, replace_targets=False, unk_idx=1):
        '''
        x               int32 tensor [N, seq_len]
        lengths         int64 tensor [N]
        unk_prob_table  float tensor [vocab_size] with the replacement probability of each
                        token index, or None for no replacement (val / test)
        '''
        if len(x) != len(lengths):
            raise ValueError("x and lengths have different numbers of jokes")
        self.x = x
        self.lengths = lengths
        self.unk_prob_table = unk_prob_table
        self.replace_targets = replace_targets
        self.unk_idx = unk_idx

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        seq = self.x[idx].to(torch.int64)
        if self.unk_prob_table is None:
            return seq, seq, self.lengths[idx]
        replace = torch.rand(seq.shape) < self.unk_prob_table[seq]
        noisy = torch.where(replace, torch.full_like(seq, self.unk_idx), seq)
        target_seq = noisy if self.replace_targets else seq
        return noisy, target_seq, self.lengths[idx]

    @staticmethod
    def collate_fn(batch):
        '''
        Turn a list of (input_seq, target_seq, length) items into one batch
        (inputs, targets, lengths): shift by one position and trim to the longest
        joke in the batch (rows are already right-padded, so no extra padding).
        '''
        input_seqs, target_seqs, lengths = zip(*batch)
        input_seqs = torch.stack(input_seqs)
        target_seqs = torch.stack(target_seqs)
        lengths = torch.stack(lengths).to(device='cpu', dtype=torch.int64) - 1
        longest = max(int(lengths.max()), 1)
        inputs = input_seqs[:, :longest].clone()
        targets = target_seqs[:, 1:longest + 1].clone()
        # the last real input of a short row would otherwise be <EOS> sitting in padding
        pad_mask = torch.arange(longest).unsqueeze(0) >= lengths.unsqueeze(1)
        inputs[pad_mask] = PAD_IDX
        return inputs, targets, lengths


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    # Filled in by load() from the saved file's config / vocabulary.
    vocab_size = None
    max_len = None   # max tokens per joke, not counting <BOS> and <EOS>
    seq_len = None   # stored row width: max_len + 2
    pad_idx = None
    unk_idx = None
    bos_idx = None
    eos_idx = None
    stoi = None   # token -> index
    itos = None   # index -> token

    REQUIRED_KEYS = ('vocab', 'word_counts', 'train', 'val', 'test', 'config')
    SPLIT_KEYS = ('x', 'lengths')

    def __init__(self, dName=None, dDescription=None, unk_prob=UNK_PROB,
                 rare_max_count=RARE_MAX_COUNT, replace_targets=UNK_REPLACE_TARGETS):
        '''
        unk_prob         probability of replacing an eligible rare word by <UNK> in the
                         training split (0 turns the replacement off)
        rare_max_count   words seen at most this many times in train are eligible
        replace_targets  False: only inputs get <UNK>;  True: targets too (see JokesDataset)
        These can also be set as attributes before calling load().
        '''
        super().__init__(dName, dDescription)
        self.unk_prob = unk_prob
        self.rare_max_count = rare_max_count
        self.replace_targets = replace_targets

    def _build_unk_prob_table(self):
        '''Per-token replacement probability: unk_prob for rare words, 0 for everything else.'''
        if self.unk_prob <= 0:
            return None
        table = torch.zeros(self.vocab_size)
        word_counts = self.data['word_counts']
        n_special = 4  # <PAD>, <UNK>, <BOS>, <EOS> are never replaced
        for idx in range(n_special, self.vocab_size):
            if word_counts[self.itos[idx]] <= self.rare_max_count:
                table[idx] = self.unk_prob
        return table

    def load(self):
        '''
        Load the encoded dataset file (produced by joke_build_vocab.py) and return
        a dictionary of JokesDataset objects:

            {'train': JokesDataset, 'val': JokesDataset, 'test': JokesDataset}

        See JokesDataset for the item format, the batch format and the <UNK> replacement.
        If dataset_source_folder_path / dataset_source_file_name are not set, the
        default location (data/stage_4_data/text_generation/joke_encoded.pt) is used.
        '''
        print('loading data...')
        folder = self.dataset_source_folder_path or DEFAULT_FOLDER_PATH
        file_name = self.dataset_source_file_name or DEFAULT_FILE_NAME
        path = os.path.join(folder, file_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Encoded dataset not found: {path} (run joke_build_vocab.py first)")
        data = torch.load(path, weights_only=True)

        missing = [k for k in self.REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(f"Encoded dataset file is missing keys: {missing}")
        for split in ('train', 'val', 'test'):
            missing = [k for k in self.SPLIT_KEYS if k not in data[split]]
            if missing:
                raise ValueError(f"Split '{split}' is missing keys: {missing}")

        self.data = data
        config = data['config']
        if config['pad_idx'] != PAD_IDX:
            raise ValueError(f"Expected pad_idx {PAD_IDX}, file has {config['pad_idx']}")
        self.max_len = config['max_len']
        self.seq_len = config['seq_len']
        self.pad_idx = config['pad_idx']
        self.unk_idx = config['unk_idx']
        self.bos_idx = config['bos_idx']
        self.eos_idx = config['eos_idx']
        self.stoi = data['vocab']['stoi']
        self.itos = data['vocab']['itos']
        self.vocab_size = len(self.itos)

        table = self._build_unk_prob_table()
        return {
            'train': JokesDataset(data['train']['x'], data['train']['lengths'],
                                  unk_prob_table=table, replace_targets=self.replace_targets,
                                  unk_idx=self.unk_idx),
            'val': JokesDataset(data['val']['x'], data['val']['lengths'],
                                unk_idx=self.unk_idx),
            'test': JokesDataset(data['test']['x'], data['test']['lengths'],
                                 unk_idx=self.unk_idx),
        }

    def decode(self, ids, skip_special=False):
        '''Turn a sequence of token indices (list or 1-D tensor) back into text.'''
        if torch.is_tensor(ids):
            ids = ids.tolist()
        tokens = [self.itos[i] for i in ids if i != self.pad_idx]
        if skip_special:
            tokens = [t for t in tokens if not (t.startswith('<') and t.endswith('>'))]
        return ' '.join(tokens)
