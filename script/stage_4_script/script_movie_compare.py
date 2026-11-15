"""Train movie LSTM/GRU using the team's encoded data and shared RNN settings.

Run from any directory. Paths are resolved relative to the repository root.
Smoke runs are explicitly marked and kept separate from full experiment results.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import torch
from local_code.stage_4_code.Dataset_Loader_Movie import Dataset_Loader
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_4_code.Method_Recurrent_Movie import Method_Recurrent_Movie
from local_code.stage_4_code.Result_Saver_Movie import Result_Saver_Movie
from local_code.stage_4_code.Setting_Movie import Setting_Movie


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=['RNN','LSTM','GRU'],default=['LSTM','GRU'])
    parser.add_argument('--data',type=Path,default=ROOT/'data/stage_4_data/text_classification/movie_encoded.pt')
    parser.add_argument('--output',type=Path,default=ROOT/'result/stage_4_movie')
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--embedding-dim',type=int,default=128)
    parser.add_argument('--hidden-size',type=int,default=128)
    parser.add_argument('--num-layers',type=int,default=1)
    parser.add_argument('--dropout',type=float,default=0.)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--seed',type=int,default=2)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--device',choices=['cpu','cuda','mps'],default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--smoke',action='store_true',help='Use 256 train/128 val/128 test reviews; not reportable results')
    args=parser.parse_args()
    if min(args.epochs,args.embedding_dim,args.hidden_size,args.num_layers,args.batch_size,args.threads)<1:
        parser.error('epochs, dimensions, layer count, batch size and threads must be positive')
    if not 0 <= args.dropout < 1 or args.lr <= 0:
        parser.error('dropout must be in [0,1) and learning rate must be positive')
    if not args.data.is_file():
        parser.error(f'Encoded dataset not found: {args.data}. Run script_movie_prepare.py first.')
    output=args.output/('smoke' if args.smoke else 'full')
    metrics=[]
    for cell in args.models:
        config=dict(cell=cell,epochs=args.epochs,embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,num_layers=args.num_layers,dropout=args.dropout,
            batch_size=args.batch_size,lr=args.lr,seed=args.seed,threads=args.threads,device=args.device)
        if args.smoke:
            config.update(train_limit=256,val_limit=128,test_limit=128)
        folder=output/cell.lower()
        if (folder/'best.pt').exists():
            parser.error(f'{folder} already contains a checkpoint; use a new --output directory to retain previous runs')
        dataset=Dataset_Loader('IMDb','Team-provided encoded sentiment reviews')
        dataset.dataset_source_folder_path=str(args.data.resolve().parent)
        dataset.dataset_source_file_name=args.data.name
        # The dataset loader establishes vocabulary size before model initialization.
        info=torch.load(args.data,weights_only=True)
        method=Method_Recurrent_Movie(vocab_size=len(info['vocab']['itos']),cell=cell,
                    embedding_dim=args.embedding_dim,hidden_size=args.hidden_size,
                    num_layers=args.num_layers,dropout=args.dropout,pad_idx=info['config']['pad_idx'])
        setting=Setting_Movie(config)
        setting.prepare(dataset,method,Result_Saver_Movie(folder),Evaluate_Accuracy(num_classes=2,label_start=0))
        metrics.append(setting.load_run_save_evaluate())
    (output/'comparison.json').write_text(json.dumps(metrics,indent=2)+'\n')
    from local_code.stage_4_code.movie_plot_results import plot_results
    plot_results(output)


if __name__=='__main__': main()
