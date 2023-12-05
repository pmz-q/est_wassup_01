from dataclasses import dataclass, field
from makers import EvalMaker
from sklearn.model_selection import KFold
from torch import nn, manual_seed
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm.auto import trange
from trains import Evaluate, Train
from typing import Type

import pandas as pd


@dataclass
class KFoldCV:
  e_m: EvalMaker
  def run(self):
    manual_seed(2023)
    kfold = KFold(n_splits=self.e_m.n_split, shuffle=False)
    metrics = {'trn_rmse': [], 'val_rmse': []}
    for k,_ in self.e_m.get_metrics().items():
        metrics[f'trn_history_{k}'] = []
        metrics[f'val_history_{k}'] = []
    
    X = self.e_m.get_X()
    y = self.e_m.get_y()
    # print(X.shape)
    # print(y.shape)
    for i, (trn_idx, val_idx) in enumerate(kfold.split(X)):
      self.e_m.init_model()
      X_trn, y_trn = X[trn_idx], y[trn_idx]
      X_val, y_val = X[val_idx], y[val_idx]

      ds_trn = TensorDataset(X_trn, y_trn)
      ds_val = TensorDataset(X_val, y_val)

      dl_trn = DataLoader(ds_trn, shuffle=True, **self.e_m.dataloader_params)
      dl_val = DataLoader(ds_val, **self.e_m.dataloader_params)

      for k,_ in self.e_m.get_metrics().items():
        metrics[f'trn_history_{k}'].append([])
        metrics[f'val_history_{k}'].append([])
      
      pbar = trange(self.e_m.epochs) #trange Tqdm + range
      for _ in pbar:
        values = {f'{which}{k}':'' for which in ['trn_', 'val_'] for k in self.e_m.get_metrics().keys()}
        Train.train_one_epoch(**{**self.e_m.get_train_parameters(), 'data_loader': dl_trn})
        for k,m in self.e_m.get_metrics().items():
          trn_m = m.compute().item()
          values[f'trn_{k}'] = trn_m
          metrics[f'trn_history_{k}'][i].append(trn_m)
          m.reset()
        
        Evaluate.run(**{**self.e_m.get_eval_parameters(), 'data_loader': dl_val})
        for k,m in self.e_m.get_metrics().items():
          val_m = m.compute().item()
          values[f'val_{k}'] = val_m
          metrics[f'val_history_{k}'][i].append(val_m)
          m.reset()
        pbar.set_postfix(**values)
      metrics['trn_rmse'].append(metrics[f'trn_history_rmse'][i][-1])
      metrics['val_rmse'].append(metrics[f'val_history_rmse'][i][-1])
    return pd.DataFrame(metrics)

  def __call__(self):
    return self.run()