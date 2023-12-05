import pandas as pd
from sklearn.base import BaseEstimator
import torch
from typing import Any, Type
from makers import PredMaker
from utils import create_path_if_not_exists


class Predict():
  def __init__(
    self, 
    y_scaler: Type[BaseEstimator],
    train_target_original_csv: str,
    index_col: str,
    target_col: str,
    **kwargs: Any
  ):
    self.p_m = PredMaker(y_scaler, train_target_original_csv, index_col, target_col, **kwargs)
  
  def inference_test_ann(self, dl_tst):
    result = []
    with torch.inference_mode():
      for X in dl_tst:
        X = X[0].to(self.p_m.device)
        output = self.p_m.model(X).squeeze().tolist()
        result.extend(output)
    return result

  def unnormalization(self, pred:iter):
    scaler = self.p_m.get_y_scaler()
    if scaler == None:
      return pred
    return scaler.inverse_transform(pd.DataFrame(pred))
  
  def run(self):
    self.p_m.model.load_state_dict(torch.load(self.p_m.output_train))
    self.p_m.model.eval()
    
    X_tst_index, X_tst_dl = self.p_m.get_tst_X()
    result = self.inference_test_ann(X_tst_dl)
    id, target_col = self.p_m.get_idx_target_cols()
    
    list_df = pd.DataFrame(zip(X_tst_index, result), columns=[id, target_col])
    
    list_df[target_col] = self.unnormalization(list_df[target_col])
    list_df[target_col] = list_df[target_col].apply(lambda x: x if x >= 0 else 0)
    create_path_if_not_exists(self.p_m.y_output_csv)
    list_df.to_csv(self.p_m.y_output_csv, index=False)