from dataclasses import dataclass
from makers import TrainMaker
from torch import nn, save
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric
from typing import Type
from tqdm.auto import trange
from utils.file_saver import create_path_if_not_exists


@dataclass
class Train:
  @classmethod
  def train_one_epoch(cls, model: Type[nn.Module], criterion: callable, optimizer: Type[Optimizer], data_loader: Type[DataLoader], device: str, metrics:dict) -> None:
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
      for _,m in metrics.items():
        m.update(output, y)
  
  @classmethod
  def run(cls, t_m: Type[TrainMaker]) -> None:
    values = {k:[] for k in t_m.get_metrics().keys()}
    pbar = trange(t_m.epochs)
    for _ in pbar:
      train_params = t_m.get_train_parameters()
      cls.train_one_epoch(**train_params)
      for k,m in t_m.get_metrics().items():
        values[k].append(m.compute().item())
      pbar.set_postfix(**{k:v[-1] for k,v in values.items()})
    create_path_if_not_exists(t_m.output_train)
    save(t_m.model.state_dict(), t_m.output_train)
  
  @classmethod
  def __call__(cls, train_maker: Type[TrainMaker]) -> None:
    cls.run(train_maker)