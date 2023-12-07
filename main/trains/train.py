from dataclasses import dataclass
from makers import TrainMaker
from torch import nn, save
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from typing import Type
from tqdm.auto import trange
from utils.file_saver import create_path_if_not_exists


@dataclass
class Train:
  @classmethod
  def train_one_epoch(
    cls,
    model: Type[nn.Module],
    criterion: callable,
    optimizer: Type[Optimizer],
    data_loader: Type[DataLoader],
    device: str,
    metrics:Type[MetricCollection],
    scheduler:Type[LRScheduler]
  ) -> None:
    '''train one epoch
    '''
    model.train()
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)
      output = model(X)
      loss = criterion(output, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      metrics.update(output, y)
    if scheduler != None: scheduler.step()
  
  @classmethod
  def run(cls, t_m: Type[TrainMaker]) -> None:
    trn_metrics = t_m.metrics['trn']
    trn_values = {k: '' for k in trn_metrics.keys()}
    pbar = trange(t_m.epochs)
    for _ in pbar:
      train_params = t_m.get_train_parameters()
      cls.train_one_epoch(**train_params)
      for k,v in trn_metrics.compute().items(): trn_values[k]=t_m.calc_multi_output_weight(v)
      trn_metrics.reset()
      pbar.set_postfix(trn_values)
    create_path_if_not_exists(t_m.output_train)
    save(t_m.model.state_dict(), t_m.output_train)
  
  @classmethod
  def __call__(cls, train_maker: Type[TrainMaker]) -> None:
    cls.run(train_maker)