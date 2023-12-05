from .dataset import Dataset

from dataclasses import dataclass
from typing import Literal
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from .utils import custom_X_preprocess_cat

import numpy as np
import pandas as pd


@dataclass(kw_only=True)
class HomeData(Dataset):
  fill_num_strategy: Literal['mean', 'min', 'max'] = 'min'
  x_scaler: BaseEstimator = None
  y_scaler: BaseEstimator = None
  
  def _scale_X(self, X_df: pd.DataFrame):
    self.x_scaler.fit(X_df)
    return self.x_scaler.transform(X_df).astype(dtype=np.float32)

  def _scale_Y(self, target_origin:iter):
    target = pd.DataFrame(target_origin)
    if self.y_scaler == None:
      return target, target
    self.y_scaler.fit(target)
    return target_origin, pd.DataFrame(self.y_scaler.transform(target).astype(dtype=np.float32))

  def _X_preprocess(self, X_df: pd.DataFrame):
    # Custom X preprocess for cat data - label encoded or cat objects
    X_df = custom_X_preprocess_cat(X_df)
    
    # Numeric
    df_num = X_df.select_dtypes(include=['number'])
    if self.fill_num_strategy == 'mean':
      fill_values = df_num.mean(axis=1)
    elif self.fill_num_strategy == 'min':
      fill_values = df_num.min(axis=1)
    elif self.fill_num_strategy == 'max':
      fill_values = df_num.max(axis=1)
    df_num.fillna(fill_values, inplace=True)
    if self.x_scaler is not None:
      df_num = pd.DataFrame(self._scale_X(df_num))
    
    # Categorical
    df_cat = X_df.select_dtypes(include=['object'])
    
    # print(df_cat)
    enc = OneHotEncoder(dtype=np.float32, sparse_output=False, drop='if_binary', handle_unknown='ignore')
    df_cat_onehot = pd.DataFrame(enc.fit_transform(df_cat))
    
    df_arr = pd.concat([df_num.reset_index(drop=True), df_cat_onehot.reset_index(drop=True)],axis=1).set_index(X_df.index)
    
    return df_arr
  
  def preprocess(self):
    """_summary_

    Returns:
        trn_X, scaled_target, target_origin, tst_X
    """
    trn_df, target, tst_df = self._get_dataset()

    # X Features
    trn_X = self._X_preprocess(trn_df)
    tst_X = self._X_preprocess(tst_df)
    
    # Y Feature
    target_origin, scaled_target = self._scale_Y(target)

    return trn_X, scaled_target, target_origin, tst_X
