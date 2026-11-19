'''
Stage 4 joke generation, step 4-4: compare the generated jokes with the training jokes.

For every joke in result/stage_4_joke/<run>/generated.txt it measures
    exact copy       the whole joke is identical to a training joke
    longest shared   the longest run of consecutive words it shares with any training joke
    copied 4-grams   % of its 4-word phrases that also appear in the training jokes
                     (100% = stitched entirely from training pieces, 0% = all new combinations)
Only the training split is used, since that is all the model saw.

usage (from the repository root):
    python script/stage_4_script/script_joke_compare.py                 # every run that has generated.txt
    python script/stage_4_script/script_joke_compare.py rnn lstm gru     # only these runs

Saves comparison.csv (one row per joke) in each run folder and
result/stage_4_joke/comparison_summary.csv (one row per run and decoding mode).
'''

from pathlib import Path
import argparse
import csv
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
RESULT_DIR = PROJECT_ROOT / 'result' / 'stage_4_joke'

from local_code.stage_4_code.Dataset_Loader_Joke import Dataset_Loader

NGRAM = 4
LINE_RE = re.compile(r'^(greedy|sample \d+) ?(?:\(T=[\d.]+\))?:\s+(.*)$')


def read_generated(path):
    '''Returns [(prompt, mode, tokens)] where mode is "greedy" or "sampled"'''
    jokes, prompt = [], None
    for line in path.read_text().splitlines():
        if line.startswith('=== '):
            prompt = line[4:].strip()
            continue
        match = LINE_RE.match(line)
        if match:
            mode = 'greedy' if match.group(1) == 'greedy' else 'sampled'
            jokes.append((prompt, mode, match.group(2).split()))
    return jokes


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


class TrainingIndex:
    '''All phrases of every length in the training jokes, mapped to the first joke that contains them.'''

    def __init__(self, train_jokes):
        self.jokes = train_jokes
        self.exact = {tuple(j) for j in train_jokes}
        self.phrases = {}
        for idx, joke in enumerate(train_jokes):
            for n in range(1, len(joke) + 1):
                for gram in ngrams(joke, n):
                    self.phrases.setdefault(gram, idx)

    def longest_shared(self, tokens):
        for n in range(len(tokens), 0, -1):
            for gram in ngrams(tokens, n):
                if gram in self.phrases:
                    return n, ' '.join(gram), ' '.join(self.jokes[self.phrases[gram]])
        return 0, '', ''

    def copied_ngram_fraction(self, tokens, n=NGRAM):
        grams = ngrams(tokens, n)
        return sum(g in self.phrases for g in grams) / len(grams) if grams else 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='*', help='run folder names (default: every run with generated.txt)')
    args = parser.parse_args()
    run_names = args.runs or sorted(p.parent.name for p in RESULT_DIR.glob('*/generated.txt'))
    if not run_names:
        sys.exit(f'no generated.txt found under {RESULT_DIR}; run script_joke_generation.py first')

    # training jokes as word lists, decoded the same way generated.txt was written
    data_obj = Dataset_Loader('jokes', 'encoded joke dataset')
    train = data_obj.load()['train']
    train_jokes = [data_obj.decode(row[:length], skip_special=True).split()
                   for row, length in zip(train.x, train.lengths.tolist())]
    index = TrainingIndex(train_jokes)
    print(f'{len(train_jokes)} training jokes indexed')

    summary = []
    for run in run_names:
        path = RESULT_DIR / run / 'generated.txt'
        if not path.is_file():
            print(f'skip {run}: no generated.txt')
            continue
        rows = []
        for prompt, mode, tokens in read_generated(path):
            shared_len, shared, closest = index.longest_shared(tokens)
            rows.append({'prompt': prompt, 'mode': mode, 'joke': ' '.join(tokens), 'words': len(tokens),
                         'exact_copy': tuple(tokens) in index.exact,
                         'longest_shared_words': shared_len, 'shared_phrase': shared,
                         'closest_training_joke': closest,
                         f'copied_{NGRAM}grams_pct': round(100 * index.copied_ngram_fraction(tokens), 1)})
        with open(RESULT_DIR / run / 'comparison.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        for mode in ('greedy', 'sampled'):
            part = [r for r in rows if r['mode'] == mode]
            if not part:
                continue
            summary.append({
                'run': run, 'mode': mode, 'jokes': len(part),
                'exact_copies': sum(r['exact_copy'] for r in part),
                'avg_longest_shared_words': round(sum(r['longest_shared_words'] for r in part) / len(part), 1),
                f'avg_copied_{NGRAM}grams_pct': round(sum(r[f'copied_{NGRAM}grams_pct'] for r in part) / len(part), 1),
                'avg_joke_words': round(sum(r['words'] for r in part) / len(part), 1)})

    with open(RESULT_DIR / 'comparison_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    header = ['run', 'mode', 'jokes', 'exact', 'avg longest shared', f'% {NGRAM}-grams copied', 'avg words']
    print('\n' + ' | '.join(f'{h:>20}' if i > 1 else f'{h:<20}' for i, h in enumerate(header)))
    for s in summary:
        values = list(s.values())
        print(' | '.join(f'{str(v):>20}' if i > 1 else f'{str(v):<20}' for i, v in enumerate(values)))
    print('\nper-joke details: result/stage_4_joke/<run>/comparison.csv')
    print('summary saved to', RESULT_DIR / 'comparison_summary.csv')
