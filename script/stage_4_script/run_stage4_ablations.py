'''
Stage 4 ablations (section 3.7): GRU hidden size and depth for movie classification and joke generation.

Uses the team's own code on the Stage-4 branch, unchanged:
  - data:  script_movie_prepare.py (Samantha's movie cleaning + vocabulary),
           joke_clean_dataset.py + joke_build_vocab.py (Samantha's joke pipeline)
  - movie: script_movie_compare.py (Andrew's GRU classifier, baseline 128 hidden units, 1 layer)
  - jokes: script_joke_generation.py (Sai's GRU generator, baseline 256 hidden units, 1 layer)
Each configuration changes exactly one setting of the GRU baseline, and the baseline is
re-trained in the same environment so every comparison uses the same hardware.

Every finished run is copied to OUT_DIR (Google Drive in Colab). Re-running a step skips
finished runs, so a disconnected Colab session can simply run the same cell again.

usage:
  python run_stage4_ablations.py prepare --repo REPO --raw RAW_DIR --out OUT_DIR
  python run_stage4_ablations.py movie   --repo REPO --out OUT_DIR
  python run_stage4_ablations.py joke    --repo REPO --out OUT_DIR
  python run_stage4_ablations.py report  --repo REPO --out OUT_DIR
'''

from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys

# one change at a time relative to each task's GRU baseline
MOVIE_CONFIGS = {  # baseline: 128 hidden units, 1 layer (Andrew's settings in 3.2)
    'baseline': ([], 'Baseline (128 hidden, 1 layer)'),
    'hidden_64': (['--hidden-size', '64'], 'Hidden size 64'),
    'hidden_256': (['--hidden-size', '256'], 'Hidden size 256'),
    'layers_2': (['--num-layers', '2'], '2 GRU layers'),
}
JOKE_CONFIGS = {  # baseline: 256 hidden units, 1 layer (Sai's settings)
    'baseline': ([], 'Baseline (256 hidden, 1 layer)'),
    'hidden_128': (['--hidden-size', '128'], 'Hidden size 128'),
    'hidden_512': (['--hidden-size', '512'], 'Hidden size 512'),
    'layers_2': (['--num-layers', '2'], '2 GRU layers'),
}
ANDREW_MOVIE_SHA256 = '7020050acdc0349159440ca84e3cae1b71eaa92c5dfdfd45ad0cddf79cb9a3c2'


def run(cmd, cwd):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def find_raw_root(raw):
    '''The folder that holds text_classification/ and text_generation/ (any nesting in the zip).'''
    for path in [raw, *sorted(raw.rglob('*'))]:
        if path.is_dir() and '__MACOSX' not in path.parts and \
                (path / 'text_classification').is_dir() and (path / 'text_generation').is_dir():
            return path
    sys.exit(f'STOP: could not find text_classification/ and text_generation/ under {raw}')


# ---------------------------------------------------------------- data
def prepare(args):
    repo, out = args.repo, args.out
    cache = out / 'data_cache'
    cache.mkdir(parents=True, exist_ok=True)
    movie_dir = repo / 'data/stage_4_data/text_classification'
    joke_dir = repo / 'data/stage_4_data/text_generation'
    movie_pt, joke_pt = movie_dir / 'movie_encoded.pt', joke_dir / 'joke_encoded.pt'
    raw = None

    # movie reviews: Samantha's cleaning + vocabulary through Andrew's preparation script
    if (cache / 'movie_encoded.pt').exists():
        movie_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache / 'movie_encoded.pt', movie_pt)
        print('movie data: reused', cache / 'movie_encoded.pt')
    elif not movie_pt.exists():
        raw = find_raw_root(args.raw)
        # default NLTK location (newer NLTK versions refuse arbitrary download folders)
        nltk_dir = Path.home() / 'nltk_data' if args.nltk_data is None else args.nltk_data
        run([sys.executable, '-c', 'import nltk\nfor p in ("punkt","punkt_tab","stopwords"): '
             'assert nltk.download(p, quiet=True), p'], repo)
        run([sys.executable, 'script/stage_4_script/script_movie_prepare.py', '--data-dir',
             raw / 'text_classification', '--output-dir', movie_dir, '--nltk-data', nltk_dir], repo)
        shutil.copy2(movie_pt, cache / 'movie_encoded.pt')

    # jokes: Samantha's cleaning + vocabulary scripts (paths are fixed relative to the repo)
    if (cache / 'joke_encoded.pt').exists():
        joke_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache / 'joke_encoded.pt', joke_pt)
        print('joke data: reused', cache / 'joke_encoded.pt')
    elif not joke_pt.exists():
        raw = raw or find_raw_root(args.raw)
        joke_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw / 'text_generation' / 'data', joke_dir / 'data')
        run([sys.executable, 'local_code/stage_4_code/joke_clean_dataset.py'], repo)
        run([sys.executable, 'local_code/stage_4_code/joke_build_vocab.py'], repo)
        shutil.copy2(joke_pt, cache / 'joke_encoded.pt')

    # same data as the team? (the movie hash is recorded in Andrew's metrics.json)
    import torch
    digest = sha256(movie_pt)
    print('\nmovie data sha256:', digest, '| matches Andrew\'s run:', digest == ANDREW_MOVIE_SHA256)
    jokes = torch.load(joke_pt, weights_only=True)
    sizes = {k: len(jokes[k]['x']) if isinstance(jokes.get(k), dict) and 'x' in jokes[k] else None
             for k in ('train', 'val', 'test')}
    print('joke splits:', sizes, '(team: 1269 / 158 / 158)')
    print('DATA READY')


# ---------------------------------------------------------------- experiments
def movie(args):
    for name, (flags, _) in MOVIE_CONFIGS.items():
        target = args.out / 'movie' / name
        if (target / 'full/gru/metrics.json').exists():
            print(f'skip movie/{name} (already done)')
            continue
        shutil.rmtree(target, ignore_errors=True)  # unfinished earlier attempt
        extra = ['--epochs', str(args.epochs)] if args.epochs else []
        print(f'\n************ movie GRU: {name} ************', flush=True)
        run([sys.executable, 'script/stage_4_script/script_movie_compare.py', '--models', 'GRU',
             '--output', target, *flags, *extra], args.repo)
        (target / 'full/gru/best.pt').unlink(missing_ok=True)  # 13 MB checkpoint, not needed
        m = json.loads((target / 'full/gru/metrics.json').read_text())
        print(f"movie/{name}: test accuracy {m['accuracy']:.4f}, macro F1 {m['f1_macro']:.4f}", flush=True)


def joke(args):
    for name, (flags, _) in JOKE_CONFIGS.items():
        target = args.out / 'joke' / name
        if (target / 'metrics.json').exists():
            print(f'skip joke/{name} (already done)')
            continue
        run_name = 'ablation_gru_' + name
        extra = (['--epochs', str(args.epochs)] if args.epochs else []) + (['--allow-cpu'] if args.allow_cpu else [])
        print(f'\n************ joke GRU: {name} ************', flush=True)
        run([sys.executable, 'script/stage_4_script/script_joke_generation.py', '--cell', 'GRU',
             '--run-name', run_name, *flags, *extra], args.repo)
        source = args.repo / 'result/stage_4_joke' / run_name
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('best.pt'))
        m = json.loads((target / 'metrics.json').read_text())
        print(f"joke/{name}: test perplexity {m['test']['perplexity']:.2f}, "
              f"next-word accuracy {m['test']['accuracy']:.4f}", flush=True)


# ---------------------------------------------------------------- report
def md_table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    return '\n'.join(lines + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def report(args):
    out = args.out
    movie_rows, joke_rows, missing = {}, {}, []
    for name, (_, label) in MOVIE_CONFIGS.items():
        p = out / 'movie' / name / 'full/gru/metrics.json'
        if p.exists():
            m = json.loads(p.read_text())
            movie_rows[name] = dict(label=label, acc=m['accuracy'], f1=m['f1_macro'],
                                    epoch=m['best_epoch'], params=m['parameter_count'])
        else:
            missing.append('movie/' + name)
    for name, (_, label) in JOKE_CONFIGS.items():
        p = out / 'joke' / name / 'metrics.json'
        if p.exists():
            m = json.loads(p.read_text())
            joke_rows[name] = dict(label=label, ppl=m['test']['perplexity'], acc=m['test']['accuracy'],
                                   epoch=m['best_epoch'], params=m['parameters'])
        else:
            missing.append('joke/' + name)
    if missing:
        sys.exit('STOP: these runs are not finished yet: ' + ', '.join(missing) + '. Re-run their cells.')

    def delta(value, base, scale=100, unit=' pts', fmt='+.1f'):
        return f'{scale * (value - base):{fmt}}{unit}'

    mb, jb = movie_rows['baseline'], joke_rows['baseline']
    movie_table = md_table(
        ['GRU setting', 'Test accuracy', 'Macro F1', 'Best epoch', 'Parameters'],
        [[r['label'], f"{100 * r['acc']:.2f}%", f"{r['f1']:.4f}", r['epoch'], f"{r['params']:,}"]
         for r in movie_rows.values()])
    joke_table = md_table(
        ['GRU setting', 'Test perplexity', 'Next-word accuracy', 'Best epoch', 'Parameters'],
        [[r['label'], f"{r['ppl']:.2f}", f"{100 * r['acc']:.2f}%", r['epoch'], f"{r['params']:,}"]
         for r in joke_rows.values()])

    best_movie = max(movie_rows.values(), key=lambda r: r['acc'])
    best_joke = min(joke_rows.values(), key=lambda r: r['ppl'])
    m = movie_rows
    j = joke_rows
    movie_text = (
        f"Starting from the GRU classifier in 3.2 and changing one setting at a time (all other settings, "
        f"the data split and seed 2 unchanged; the baseline was re-trained in the same environment), "
        f"a hidden size of 64 gave {100 * m['hidden_64']['acc']:.2f}% test accuracy "
        f"({delta(m['hidden_64']['acc'], mb['acc'])} vs. the baseline's {100 * mb['acc']:.2f}%), a hidden size of 256 gave "
        f"{100 * m['hidden_256']['acc']:.2f}% ({delta(m['hidden_256']['acc'], mb['acc'])}), and a second GRU layer gave "
        f"{100 * m['layers_2']['acc']:.2f}% ({delta(m['layers_2']['acc'], mb['acc'])}). "
        f"The best setting was: {best_movie['label']}. Each setting was trained once, so differences "
        f"of a few tenths of a point are within run-to-run variation.")
    joke_text = (
        f"Starting from the GRU joke generator (256 hidden units, one layer) and changing one setting at a time, "
        f"a hidden size of 128 gave a test perplexity of {j['hidden_128']['ppl']:.2f} "
        f"({j['hidden_128']['ppl'] - jb['ppl']:+.2f} vs. the baseline's {jb['ppl']:.2f}), a hidden size of 512 gave "
        f"{j['hidden_512']['ppl']:.2f} ({j['hidden_512']['ppl'] - jb['ppl']:+.2f}), and a second GRU layer gave "
        f"{j['layers_2']['ppl']:.2f} ({j['layers_2']['ppl'] - jb['ppl']:+.2f}); lower is better. "
        f"The best setting was: {best_joke['label']}. With only 1,269 training jokes, each run was trained once, "
        f"so small differences should be read with caution.")
    snippet = (
        '### 3.7.3 GRU Movie Classification: Hidden Size and Depth\n\n' + movie_table + '\n\n' + movie_text +
        '\n\n### 3.7.4 GRU Joke Generation: Hidden Size and Depth\n\n' + joke_table + '\n\n' + joke_text + '\n')
    (out / 'report_3.7_snippet.md').write_text(snippet)

    # small folder for GitHub (no checkpoints, no 25,000-row prediction files)
    upload = out / 'github_upload' / 'result' / 'stage_4_ablation'
    shutil.rmtree(out / 'github_upload', ignore_errors=True)
    for name in MOVIE_CONFIGS:
        d = upload / 'movie_gru' / name
        d.mkdir(parents=True)
        for f in ('metrics.json', 'history.csv'):
            shutil.copy2(out / 'movie' / name / 'full/gru' / f, d / f)
    for name in JOKE_CONFIGS:
        d = upload / 'joke_gru' / name
        d.mkdir(parents=True)
        for f in ('metrics.json', 'history.csv', 'generated.txt', 'learning_curve.png'):
            if (out / 'joke' / name / f).exists():
                shutil.copy2(out / 'joke' / name / f, d / f)
    (upload / 'README.md').write_text(
        '# Stage 4 ablations: GRU hidden size and depth (report 3.7.3 and 3.7.4)\n\n'
        'Produced by `script/stage_4_script/run_stage4_ablations.py` with the team\'s unchanged Stage-4 code: '
        '`script_movie_compare.py --models GRU` (movie) and `script_joke_generation.py --cell GRU` (jokes). '
        'Each configuration changes one setting of the GRU baseline; the baseline was re-trained in the same '
        'environment. Checkpoints and per-review predictions are not included.\n\n' + snippet)
    code_dir = out / 'github_upload' / 'script' / 'stage_4_script'
    code_dir.mkdir(parents=True)
    shutil.copy2(Path(__file__), code_dir / 'run_stage4_ablations.py')
    print(snippet)
    print('saved:', out / 'report_3.7_snippet.md')
    print('GitHub upload folder:', out / 'github_upload')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=['prepare', 'movie', 'joke', 'report'])
    parser.add_argument('--repo', type=Path, required=True, help='Stage-4 checkout of the team repository')
    parser.add_argument('--out', type=Path, required=True, help='where finished runs are kept (Google Drive)')
    parser.add_argument('--raw', type=Path, help='unzipped course stage_4_data (prepare step only)')
    parser.add_argument('--nltk-data', type=Path, default=None)
    parser.add_argument('--epochs', type=int, default=None, help='testing only: override the epoch count')
    parser.add_argument('--allow-cpu', action='store_true', help='testing only: let the joke script run on CPU')
    args = parser.parse_args()
    args.repo, args.out = args.repo.resolve(), args.out.resolve()
    {'prepare': prepare, 'movie': movie, 'joke': joke, 'report': report}[args.step](args)
