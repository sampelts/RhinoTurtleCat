'''
Stage 4 joke generation: train an RNN / LSTM / GRU language model on the jokes,
then generate jokes from three starting words.

usage (from the repository root):
    python script/stage_4_script/script_joke_generation.py                 # vanilla RNN
    python script/stage_4_script/script_joke_generation.py --cell LSTM
    python script/stage_4_script/script_joke_generation.py --cell GRU --run-name gru_h512 --hidden-size 512

GloVe embeddings (100d, read from data/pretrained_glove/), e.g. for the ablation:
    ... --cell GRU --run-name gru_random100 --embedding-dim 100           # random, same size as GloVe
    ... --cell GRU --run-name gru_glove --glove                           # GloVe, fine-tuned
    ... --cell GRU --run-name gru_glove_frozen --glove --freeze-embeddings
    ... --cell GRU --run-name gru_glove_fallback --glove --glove-fallback

Each run is saved to result/stage_4_joke/<run-name>/:
    history.csv         train loss and validation loss / perplexity / accuracy per epoch
    learning_curve.png  training and validation loss per epoch
    metrics.json        configuration, best epoch and test loss / perplexity / accuracy
    generated.txt       generated jokes for every prompt
    best.pt             trained weights (local only, ignored by git)
'''

from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
RESULT_DIR = PROJECT_ROOT / 'result' / 'stage_4_joke'

from local_code.stage_4_code.Dataset_Loader_Joke import Dataset_Loader
from local_code.stage_4_code.Method_Recurrent_Joke import Method_Recurrent_Joke
from local_code.stage_4_code.joke_tokenizer import tokenize
from local_code.stage_4_code.Glove_Loader import load_glove_matrix, DEFAULT_GLOVE_PATH, GLOVE_DIM
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# three starting words per prompt (step 4-4); every word is in the training vocabulary
PROMPTS = ['what did the', 'why did the', 'how do you', 'what do you', 'i told my']


def plot_learning_curve(history, best_epoch, path, title):
    epochs = [r['Epoch'] for r in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [r['Train_Loss'] for r in history], label='train')
    ax.plot(epochs, [r['Val_Loss'] for r in history], label='validation')
    ax.axvline(best_epoch, color='gray', linestyle='--', label=f'best epoch ({best_epoch})')
    ax.set_xlabel('epoch')
    ax.set_ylabel('cross-entropy loss')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', choices=['RNN', 'LSTM', 'GRU'], default='RNN')
    parser.add_argument('--run-name', default=None, help='output folder name (default: the cell name)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--embedding-dim', type=int, default=None)
    parser.add_argument('--hidden-size', type=int, default=None)
    parser.add_argument('--num-layers', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--glove', action='store_true', help='start the embeddings from GloVe 100d vectors')
    parser.add_argument('--glove-path', default=DEFAULT_GLOVE_PATH)
    parser.add_argument('--glove-fallback', action='store_true',
                        help="give missing contractions/possessives the average of their GloVe pieces")
    parser.add_argument('--freeze-embeddings', action='store_true', help='keep the GloVe vectors fixed')
    parser.add_argument('--temperature', type=float, default=0.8, help='sampling temperature for generation')
    parser.add_argument('--samples', type=int, default=3, help='sampled jokes per prompt (plus one greedy)')
    parser.add_argument('--allow-cpu', action='store_true', help='run even if CUDA is not available')
    args = parser.parse_args()

    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    if not torch.cuda.is_available() and not args.allow_cpu:
        sys.exit('CUDA is not available; check the PyTorch install, or pass --allow-cpu')
    overrides = {key: value for key, value in {
        'cell': args.cell, 'max_epoch': args.epochs, 'embedding_dim': args.embedding_dim,
        'hidden_size': args.hidden_size, 'num_layers': args.num_layers, 'dropout': args.dropout,
        'learning_rate': args.learning_rate, 'batch_size': args.batch_size}.items() if value is not None}
    if (args.glove_fallback or args.freeze_embeddings) and not args.glove:
        sys.exit('--glove-fallback and --freeze-embeddings need --glove')
    if args.glove:
        if overrides.get('embedding_dim', GLOVE_DIM) != GLOVE_DIM:
            sys.exit(f'GloVe vectors have {GLOVE_DIM} dimensions; drop --embedding-dim or set it to {GLOVE_DIM}')
        overrides['embedding_dim'] = GLOVE_DIM
        overrides['freeze_embeddings'] = args.freeze_embeddings
    run_dir = RESULT_DIR / (args.run_name or args.cell.lower())
    run_dir.mkdir(parents=True, exist_ok=True)
    #------------------------------------------------------

    # ---- object initialization section ------------------
    data_obj = Dataset_Loader('jokes', 'encoded joke dataset')
    sets = data_obj.load()
    glove_matrix, glove_info = None, None
    if args.glove:
        glove_matrix, glove_info = load_glove_matrix(data_obj.itos, data_obj.pad_idx, args.glove_path,
                                                     fallback=args.glove_fallback)
    method_obj = Method_Recurrent_Joke('joke ' + args.cell, 'joke generation', data_obj.vocab_size,
                                       pad_idx=data_obj.pad_idx, unk_idx=data_obj.unk_idx,
                                       pretrained_embeddings=glove_matrix, **overrides)
    method_obj.data = sets
    print('device:', method_obj.device, torch.cuda.get_device_name() if method_obj.device.type == 'cuda' else '')
    print(method_obj)
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ' + args.cell + ' ************')
    learned = method_obj.run()
    test = learned['test']
    print('************ Test Performance ************')
    print(f"best epoch: {learned['best_epoch']}, test loss: {test['loss']:.4f}, "
          f"perplexity: {test['perplexity']:.2f}, next-word accuracy: {test['accuracy']:.4f}")

    # generate jokes from three starting words
    banned = [data_obj.pad_idx, data_obj.bos_idx, data_obj.unk_idx]
    lines = []
    for prompt in PROMPTS:
        prompt_ids = [data_obj.stoi.get(token, data_obj.unk_idx) for token in tokenize(prompt.lower())]
        lines.append(f'=== {prompt}')
        ids = method_obj.generate(prompt_ids, data_obj.bos_idx, data_obj.eos_idx, banned,
                                  max_len=data_obj.max_len, greedy=True)
        lines.append('greedy:  ' + data_obj.decode(ids, skip_special=True))
        for i in range(args.samples):
            ids = method_obj.generate(prompt_ids, data_obj.bos_idx, data_obj.eos_idx, banned,
                                      max_len=data_obj.max_len, temperature=args.temperature)
            lines.append(f'sample {i + 1} (T={args.temperature}): ' + data_obj.decode(ids, skip_special=True))
    print('************ Generated Jokes ************')
    print('\n'.join(lines))

    # save everything for the report
    (run_dir / 'generated.txt').write_text('\n'.join(lines) + '\n')
    with open(run_dir / 'history.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(method_obj.history[0]))
        writer.writeheader()
        writer.writerows(method_obj.history)
    plot_learning_curve(method_obj.history, learned['best_epoch'], run_dir / 'learning_curve.png',
                        f'{args.cell} joke generator')
    with open(run_dir / 'metrics.json', 'w') as f:
        embedding = 'random' if not args.glove else (
            'glove' + ('+fallback' if args.glove_fallback else '') + (' frozen' if args.freeze_embeddings else ' fine-tuned'))
        json.dump({'config': method_obj.config, 'embedding': embedding, 'glove_coverage': glove_info,
                   'best_epoch': learned['best_epoch'], 'test': test,
                   'parameters': sum(p.numel() for p in method_obj.parameters()),
                   'device': str(method_obj.device)}, f, indent=2)
    torch.save({'model_state': method_obj.state_dict(), 'config': method_obj.config,
                'vocab_size': data_obj.vocab_size}, run_dir / 'best.pt')
    print('results saved to', run_dir)
    print('************ Finish ************')
    # ------------------------------------------------------
