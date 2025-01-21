import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class ConvLSTM(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(12, 64, 3, padding=1)
    self.bn1 = nn.BatchNorm2d(64)
    self.pool1 = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
    self.bn2 = nn.BatchNorm2d(128)
    self.pool2 = nn.MaxPool2d(2)
    self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
    self.bn3 = nn.BatchNorm2d(256)
    self.pool3 = nn.MaxPool2d(2)
    self.lstm = nn.LSTM(257, 512, 2, batch_first=True, bidirectional=True, dropout=0.5)
    self.fc = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 4))

  def forward(self, moves: torch.Tensor, evals: torch.Tensor) -> torch.Tensor:
    BS, T, C, H, W = moves.shape
    x = moves.view(BS * T, C, H, W)
    x = self.pool1(F.relu(self.bn1(self.conv1(x))))
    x = self.pool2(F.relu(self.bn2(self.conv2(x))))
    x = self.pool3(F.relu(self.bn3(self.conv3(x))))
    x = F.adaptive_avg_pool2d(x, 1).view(BS, T, -1)
    x = torch.cat([x, evals.unsqueeze(-1)], dim=-1)
    x, _ = self.lstm(x)
    return self.fc(x[:, -1])


class Dense(nn.Module):
  def __init__(self, input_dim: int, hidden: list[int]):
    super().__init__()
    layers = []
    for h in hidden:
      layers.append(nn.Linear(input_dim, h))
      layers.append(nn.ReLU())
      input_dim = h
    layers.append(nn.Linear(input_dim, 4))
    self.layers = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(x.size(0), -1)
    return self.layers(x)


class Conv(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv3d(6, 64, kernel_size=(5, 3, 3), stride=(2, 1, 1))
    self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 1, 1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.permute(0, 2, 1, 3, 4)
    x = F.relu(self.conv1(x))
    return F.relu(self.conv2(x))
