"""Train on train, select checkpoints on validation, then evaluate held-out test."""
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from local_code.base_class.setting import setting
from local_code.stage_4_code.Dataset_Loader_Movie import MoviesDataset


def evaluate_model(model, loader, device):
    model.eval()
    loss_sum = 0.0
    labels, logits = [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            output = model(x.to(device), lengths)
            loss_sum += nn.functional.binary_cross_entropy_with_logits(
                output, y.float().to(device), reduction='sum').item()
            labels.extend(y.tolist())
            logits.extend(output.cpu().tolist())
    truth = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(logits)
    predicted = (scores >= 0).astype(np.int64)
    return dict(loss=loss_sum/len(truth), accuracy=float(np.mean(truth==predicted))), truth, scores


def balanced_subset(ds, limit, seed):
    """A deterministic stratified subset used only for explicitly marked smoke runs."""
    if not limit:
        return ds
    if limit < 2:
        raise ValueError('Smoke limits must include at least two examples')
    generator = torch.Generator().manual_seed(seed)
    indices = []
    for label, count in ((0, limit//2), (1, limit-limit//2)):
        candidates = torch.where(ds.y == label)[0]
        indices.extend(candidates[torch.randperm(len(candidates), generator=generator)[:count]].tolist())
    return Subset(ds, sorted(indices))


class Setting_Movie(setting):
    def __init__(self, config):
        super().__init__('IMDb train validation test', 'Official held-out test split')
        self.config = config

    def load_run_save_evaluate(self):
        cfg = self.config
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        torch.set_num_threads(cfg['threads'])
        device = torch.device(cfg['device'])
        sets = self.dataset.load()
        counts = {name:len(ds) for name,ds in sets.items()}
        for name in sets:
            lengths = sets[name].lengths
            if bool((lengths < 1).any()):
                raise ValueError(f'{name} contains empty sequences; encode empty reviews as <UNK>')
            sets[name] = balanced_subset(sets[name], cfg.get(f'{name}_limit'), cfg['seed'])
        generator = torch.Generator().manual_seed(cfg['seed'])
        loaders = {name:DataLoader(ds, batch_size=cfg['batch_size'], shuffle=name=='train',
                      collate_fn=MoviesDataset.collate_fn,
                      generator=generator if name=='train' else None) for name,ds in sets.items()}
        # Reset initialization separately for each architecture, matching Madison's seed.
        torch.manual_seed(cfg['seed'])
        from local_code.stage_4_code.Method_Recurrent_Movie import MovieRecurrent
        model = MovieRecurrent(self.dataset.vocab_size, cell=cfg['cell'],
                    embedding_dim=cfg['embedding_dim'], hidden_size=cfg['hidden_size'],
                    num_layers=cfg['num_layers'], dropout=cfg['dropout'], pad_idx=self.dataset.pad_idx).to(device)
        self.method.model = model
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        history=[]; best_accuracy=-1; best_epoch=0
        checkpoint_path = self.result.folder/'best.pt'
        start=time.perf_counter()
        for epoch in range(1,cfg['epochs']+1):
            epoch_start=time.perf_counter(); model.train()
            loss_sum=0.; correct=seen=0
            for x,lengths,y in loaders['train']:
                x=x.to(device); target=y.float().to(device)
                optimizer.zero_grad(set_to_none=True)
                output=model(x,lengths)
                loss=nn.functional.binary_cross_entropy_with_logits(output,target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                loss_sum+=loss.item()*len(y)
                correct+=int(((output.detach()>=0)==target.bool()).sum().item()); seen+=len(y)
            validation,_,_=evaluate_model(model, loaders['val'], device)
            row=dict(epoch=epoch, train_loss=loss_sum/seen, train_accuracy=correct/seen,
                     val_loss=validation['loss'], val_accuracy=validation['accuracy'],
                     seconds=time.perf_counter()-epoch_start)
            history.append(row)
            if validation['accuracy'] > best_accuracy:
                best_accuracy=validation['accuracy']; best_epoch=epoch
                torch.save(dict(model_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()},
                       model_config=self.method.config, config=cfg, data_config=self.dataset.data['config'],
                       vocab=self.dataset.data['vocab'], epoch=epoch,
                       validation_accuracy=best_accuracy), checkpoint_path)
            # Save epoch history immediately, so interrupted runs retain their progress.
            import csv
            with (self.result.folder/'history.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
            print(f"{cfg['cell']} epoch {epoch}/{cfg['epochs']} train_loss={row['train_loss']:.4f} "
                  f"val_loss={row['val_loss']:.4f} val_accuracy={row['val_accuracy']:.4f} "
                  f"seconds={row['seconds']:.1f}",flush=True)
        training_seconds=time.perf_counter()-start
        best=torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.load_state_dict(best['model_state']); model.to(device)
        test,truth,scores=evaluate_model(model,loaders['test'],device)
        predicted=(scores>=0).astype(int)
        self.evaluate.data=dict(true_y=truth.tolist(),pred_y=predicted.tolist())
        metrics=self.evaluate.evaluate()
        precision,recall,f1,support=precision_recall_fscore_support(truth,predicted,labels=[0,1],zero_division=0)
        encoded_path=Path(self.dataset.dataset_source_folder_path)/self.dataset.dataset_source_file_name
        digest=hashlib.sha256()
        with encoded_path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
        metrics.update(cell=cfg['cell'], test_loss=test['loss'], best_epoch=best_epoch,
            best_validation_accuracy=best_accuracy, parameter_count=sum(p.numel() for p in model.parameters()),
            training_seconds=training_seconds, confusion_matrix=confusion_matrix(truth,predicted,labels=[0,1]).tolist(),
            per_class={label:dict(precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]),support=int(support[i]))
                       for i,label in enumerate(['negative','positive'])},
            dataset_sha256=digest.hexdigest(), dataset_config=self.dataset.data['config'],
            original_split_sizes=counts, used_split_sizes={name:len(ds) for name,ds in sets.items()},
            smoke_run=any(cfg.get(f'{name}_limit') for name in sets), config=cfg,
            environment=dict(python=platform.python_version(),torch=str(torch.__version__),device=str(device)))
        probabilities=torch.sigmoid(torch.tensor(scores,dtype=torch.float64)).tolist()
        indices=sets['test'].indices if isinstance(sets['test'],Subset) else range(len(truth))
        predictions=[dict(test_index=int(index),true_label=int(y),predicted_label=int(p),positive_probability=float(prob))
                     for index,y,p,prob in zip(indices,truth,predicted,probabilities)]
        self.result.data=dict(metrics=metrics,history=history,predictions=predictions)
        self.result.save()
        print(json.dumps({key:metrics[key] for key in ['cell','accuracy','precision_macro','recall_macro','f1_macro','best_epoch']},indent=2),flush=True)
        return metrics
