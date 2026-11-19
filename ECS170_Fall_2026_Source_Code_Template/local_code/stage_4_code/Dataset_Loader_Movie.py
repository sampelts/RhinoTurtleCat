'''
Concrete IO class for the stage 4 text classification dataset (IMDb movie reviews)
'''

# Follows the pattern of the stage 3 Dataset_Loader_Image class.
# License: TBD

from local_code.base_class.dataset import dataset
from torch.utils.data import Dataset
import torch
import os


class MoviesDataset(Dataset):
    '''
    One split (train, val or test) of the encoded IMDb reviews.

    __getitem__(i) returns one review as a tuple (x, length, y):
        x:      int32 tensor [max_len], token indices, right-padded with <PAD>
        length: int64 scalar tensor, number of real tokens in x
        y:      int64 scalar tensor, 0 = neg, 1 = pos

    Use MoviesDataset.collate_fn as the collate_fn of a torch DataLoader:

        loader = DataLoader(datasets['train'], batch_size=64, shuffle=True,
                            collate_fn=MoviesDataset.collate_fn)
        for x, lengths, y in loader:
            ...

    Every batch is a tuple (x, lengths, y):
        x:       int64 tensor [B, T], T = length of the longest review in the batch
                 (at most max_len), right-padded with <PAD> (index 0)
        lengths: int64 tensor [B] on CPU, number of real tokens per review
                 (ready for torch.nn.utils.rnn.pack_padded_sequence)
        y:       int64 tensor [B], 0 = neg, 1 = pos (convert with .float() if
                 the loss function needs float labels)
    '''

    def __init__(self, x, lengths, y):
        if not (len(x) == len(lengths) == len(y)):
            raise ValueError("x, lengths and y have different numbers of reviews")
        self.x = x
        self.lengths = lengths
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.lengths[idx], self.y[idx]

    @staticmethod
    def collate_fn(batch):
        '''
        Turn a list of (x, length, y) items into one batch (x, lengths, y).
        Reviews are already right-padded to max_len, so the batch is trimmed to
        the longest review it contains instead of being padded again.
        '''
        xs, lengths, ys = zip(*batch)
        lengths = torch.stack(lengths).to(device='cpu', dtype=torch.int64)
        ys = torch.stack(ys).to(torch.int64)
        longest = max(int(lengths.max()), 1)
        x = torch.stack(xs)[:, :longest].to(torch.int64)
        return x, lengths, ys


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    # Filled in by load() from the saved file's config / vocabulary.
    vocab_size = None
    max_len = None
    pad_idx = None
    unk_idx = None
    stoi = None   # word -> index
    itos = None   # index -> word

    REQUIRED_KEYS = ('vocab', 'word_counts', 'train', 'val', 'test', 'config')
    SPLIT_KEYS = ('x', 'lengths', 'y')

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        '''
        Load the encoded dataset file (produced by movie_build_vocab.py) and return
        a dictionary of MoviesDataset objects:

            {'train': MoviesDataset, 'val': MoviesDataset, 'test': MoviesDataset}

        See MoviesDataset for the item format and the batch format.
        '''
        print('loading data...')
        path = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Encoded dataset not found: {path} (run movie_build_vocab.py first)")
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
        self.max_len = config['max_len']
        self.pad_idx = config['pad_idx']
        self.unk_idx = config['unk_idx']
        self.stoi = data['vocab']['stoi']
        self.itos = data['vocab']['itos']
        self.vocab_size = len(self.itos)

        return {
            split: MoviesDataset(data[split]['x'], data[split]['lengths'], data[split]['y'])
            for split in ('train', 'val', 'test')
        }
