import torch.nn as nn
import torch.nn.functional as F

activation_list = {"sigmoid": nn.Sigmoid(), "relu": nn.ReLU(), "tanh": nn.Tanh(), "prelu": nn.PReLU()}

class ANN(nn.Module):
  def __init__(self, input_dim: int=5, hidden_dim: list=[128, 128, 64, 32], activation: str="sigmoid", use_drop:bool = True, drop_ratio: float=0.0):
    super().__init__()
    dims = [input_dim] + hidden_dim 
    self.Identity = nn.Identity()
    self.dropout = nn.Dropout(drop_ratio)
    self.activation = activation_list[activation]
    
    model = [[nn.Linear(dims[i], dims[i+1]), self.dropout if use_drop else self.Identity, self.activation] for i in range(len(dims) - 1)]
    # output_layer = [nn.Linear(dims[-1], 1), nn.ReLU()] # 대구 교통사고 데이터 기준 음수 값이 나오는 경우가 있어서 제거
    output_layer = [nn.Linear(dims[-1], 1), nn.Identity()]
    self.module_list= nn.ModuleList(sum(model, []) + output_layer)
  def forward(self, x):
    for layer in self.module_list:
         x = layer(x)
    return x