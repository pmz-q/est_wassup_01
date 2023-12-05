import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from typing import Type
from .train_maker import TrainMaker


class PredMaker(TrainMaker):
  def __init__(
    self,
    y_scaler: Type[BaseEstimator],
    train_target_original_csv: str,
    index_col: str,
    target_col: str,
    **kwargs
  ):
    self.__y_scaler = y_scaler
    self.__y_origin_csv = train_target_original_csv
    self.__index_col = index_col
    self.__target_col = target_col
    super().__init__(**kwargs)
  
  @property
  def y_origin_csv(self): return getattr(self, '__y_origin_csv')
  @property
  def y_output_csv(self): return getattr(self, '__output_pred')
  
  def get_idx_target_cols(self):
    """
    Returns:
        [index_col, target_col]
    """
    return [self.__index_col, self.__target_col]
  
  def get_y_scaler(self):
    scaler = self.__y_scaler
    if scaler == None:
      return None
    y_origin = pd.read_csv(self.__y_origin_csv, index_col=0)
    scaler.fit(y_origin)
    return scaler
  
  def get_tst_X(self):
    """
    Returns:
        X_index, X_Dataloader
    """
    X_tst = pd.read_csv(getattr(self, '__tst_csv'), index_col=0)
    X_tst_tensor = torch.tensor(X_tst.to_numpy(dtype=np.float32))
    ds_tst = TensorDataset(X_tst_tensor)
    return X_tst.index.tolist(), DataLoader(ds_tst, **self.dataloader_params)