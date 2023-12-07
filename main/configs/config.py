import torch
from torch import optim
import torchmetrics
from models import ANN, RMSLELoss
from metrics import RootMeanSquaredLogError

SCHEDULER = {
  'CAWarmRestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts, # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
  'CALR': optim.lr_scheduler.CosineAnnealingLR, # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
  'lambdaLR': optim.lr_scheduler.LambdaLR, # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
}

EXPERIMENT_NAME = 'experiment_9'

config = {
  'input_files': {
    'X_csv': './data/features/train_X.csv',
    'y_csv': './data/features/train_target.csv',
    'tst_csv': './data/features/test_X.csv'
  },
  'output_files': {
    'output_train': f'./histories/{EXPERIMENT_NAME}/model.pth',
    'output_eval': f'./histories/{EXPERIMENT_NAME}/eval.csv',
    'output_pred': f'./histories/{EXPERIMENT_NAME}/pred.csv'
  },
  'model': ANN,
  'model_params': {
    'input_dim': 'auto', # Always will be determined by the data shape
    'hidden_dim': [128, 64],
    'use_drop': True,
    'drop_ratio': 0.6,
    'activation': 'relu',
    
    # Embedding params
    # embed_cols_len 은 만약 embedding 을 원치 않을 경우, 0으로 설정해주세요.
    'embed_cols_len': 1, # preprocess_config 에서 설정한 column 갯수
    'embed_dim': 10 # 늘릴 차원의 수
  },
  'train_params': {
    # 'loss': RMSLELoss(),
    'use_scheduler': True,
    'scheduler_cls': SCHEDULER['CALR'],
    'scheduler_params': {
      # 'T_0': 200,
      # 'T_mult': 2,
      'T_max': 200,
      'eta_min': 0.0000001
    },
    'loss': torch.nn.MSELoss(),
    'optim': torch.optim.Adam,
    'main_metric': 'rmse',
    'metrics': {
      'rmse': torchmetrics.MeanSquaredError(squared=False),
      'mse': torchmetrics.MeanSquaredError(),
      'rmsle': RootMeanSquaredLogError()
    },
    'device': 'cuda:0',
    'epochs': 200,
    'data_loader_params': {
      'batch_size': 32,
    },
    'optim_params': {
      'lr': 0.01,
    },
  },
  'cv_params':{
    'n_split': 5,
  }
}