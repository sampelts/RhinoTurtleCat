'''
Word-level language model for joke generation: embedding -> RNN / LSTM / GRU -> linear over the vocabulary.
At every position the model predicts the next word, so one network covers all three cells of step 4-5.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import copy
import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from local_code.stage_4_code.Dataset_Loader_Joke import JokesDataset


class Method_Recurrent_Joke(method, nn.Module):
    data = None

    # default configuration; any key can be overridden through the constructor
    config = {
        'cell': 'RNN',            # 'RNN', 'LSTM' or 'GRU'
        'embedding_dim': 128,
        'hidden_size': 256,
        'num_layers': 1,
        'dropout': 0.3,           # on the embeddings and the recurrent outputs (and between layers)
        'learning_rate': 2e-3,
        'batch_size': 32,
        'max_epoch': 40,
        'clip_norm': 1.0,         # gradient clipping, as in the movie models
    }

    def __init__(self, mName, mDescription, vocab_size, pad_idx=0, unk_idx=1, **overrides):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        unknown = set(overrides) - set(self.config)
        if unknown:
            raise ValueError('unknown config keys: ' + str(sorted(unknown)))
        self.config = {**self.config, **overrides}
        c = self.config
        if c['cell'] not in ('RNN', 'LSTM', 'GRU'):
            raise ValueError('cell must be RNN, LSTM or GRU')
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.history = []  # one row per epoch, filled by fit()
        self.best_epoch = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = nn.Embedding(vocab_size, c['embedding_dim'], padding_idx=pad_idx)
        self.recurrent = getattr(nn, c['cell'])(c['embedding_dim'], c['hidden_size'], num_layers=c['num_layers'],
                                               batch_first=True, dropout=c['dropout'] if c['num_layers'] > 1 else 0.0)
        self.dropout = nn.Dropout(c['dropout'])
        self.fc = nn.Linear(c['hidden_size'], vocab_size)
        self.to(self.device)

    def forward(self, x, hidden=None):
        '''x: [B, T] token indices -> logits [B, T, vocab_size] and the final hidden state'''
        # the network is unidirectional, so the outputs at real positions never see the
        # right padding; padded positions are skipped by the loss instead of packing
        outputs, hidden = self.recurrent(self.dropout(self.embedding(x)), hidden)
        return self.fc(self.dropout(outputs)), hidden

    def make_loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.config['batch_size'], shuffle=shuffle,
                          collate_fn=JokesDataset.collate_fn)

    def evaluate_loss(self, ds):
        '''
        Mean next-word cross-entropy, perplexity and top-1 next-word accuracy on a split.
        <PAD> and <UNK> targets are skipped: the model is never trained to predict <UNK>,
        and about 12% of the val / test words are <UNK> (see the data hand-off).
        '''
        nn.Module.train(self, False)
        total_loss, correct, count = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets, _ in self.make_loader(ds, shuffle=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits, _ = self.forward(inputs)
                keep = (targets != self.pad_idx) & (targets != self.unk_idx)
                total_loss += nn.functional.cross_entropy(logits[keep], targets[keep], reduction='sum').item()
                correct += (logits[keep].argmax(1) == targets[keep]).sum().item()
                count += keep.sum().item()
        loss = total_loss / count
        return {'loss': loss, 'perplexity': math.exp(loss), 'accuracy': correct / count}

    # named fit (not train) so nn.Module.train(mode) keeps working
    def fit(self, train_ds, val_ds):
        c = self.config
        optimizer = torch.optim.Adam(self.parameters(), lr=c['learning_rate'])
        loss_function = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        train_loader = self.make_loader(train_ds, shuffle=True)
        best_val_loss, best_state = float('inf'), None

        for epoch in range(c['max_epoch']):
            nn.Module.train(self, True)
            total_loss, count = 0.0, 0
            for inputs, targets, _ in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits, _ = self.forward(inputs)
                train_loss = loss_function(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
                optimizer.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), c['clip_norm'])
                optimizer.step()
                n = (targets != self.pad_idx).sum().item()
                total_loss += train_loss.item() * n
                count += n

            val = self.evaluate_loss(val_ds)
            record = {'Epoch': epoch + 1, 'Train_Loss': total_loss / count, 'Val_Loss': val['loss'],
                      'Val_Perplexity': val['perplexity'], 'Val_Accuracy': val['accuracy']}
            # keep the weights from the epoch with the lowest validation loss
            if val['loss'] < best_val_loss:
                best_val_loss, best_state = val['loss'], copy.deepcopy(self.state_dict())
                self.best_epoch = epoch + 1
            self.history.append(record)
            print(', '.join(k + ': ' + (f'{v:.4f}' if isinstance(v, float) else str(v)) for k, v in record.items()))

        self.load_state_dict(best_state)
        print('restored best epoch:', self.best_epoch, 'val loss:', f'{best_val_loss:.4f}')

    def generate(self, prompt_ids, bos_idx, eos_idx, banned_ids, max_len=50, temperature=1.0, greedy=False):
        '''
        Continue a prompt one word at a time until <EOS> or max_len words in total.
        greedy=True always takes the most likely word; otherwise the next word is sampled
        from the softmax at the given temperature (lower = safer, higher = more random).
        banned_ids (<PAD>, <BOS>, <UNK>) can never be produced.
        '''
        nn.Module.train(self, False)
        words = list(prompt_ids)
        with torch.no_grad():
            # read <BOS> + the prompt, then feed back one generated word at a time
            logits, hidden = self.forward(torch.tensor([[bos_idx] + words], device=self.device))
            while len(words) < max_len:
                next_logits = logits[0, -1].clone()
                next_logits[banned_ids] = float('-inf')
                if greedy:
                    next_id = int(next_logits.argmax())
                else:
                    probs = torch.softmax(next_logits / temperature, dim=0)
                    next_id = int(torch.multinomial(probs, 1))
                if next_id == eos_idx:
                    break
                words.append(next_id)
                logits, hidden = self.forward(torch.tensor([[next_id]], device=self.device), hidden)
        return words

    def run(self):
        print('method running...')
        print('--start training...')
        self.fit(self.data['train'], self.data['val'])
        print('--start testing...')
        return {'test': self.evaluate_loss(self.data['test']), 'best_epoch': self.best_epoch}
