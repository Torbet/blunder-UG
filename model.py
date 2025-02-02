import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)


class Transformer(nn.Module):
  def __init__(self, in_channels: int = 12, hidden_dim: int = 256, nhead: int = 8, num_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1):
    super().__init__()

    d_model = hidden_dim * 3

    self.move_encoder = nn.Sequential(
      nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
      nn.BatchNorm3d(32),
      nn.ReLU(),
      nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
      nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
      nn.BatchNorm3d(64),
      nn.ReLU(),
      nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
      nn.Conv3d(64, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
      nn.BatchNorm3d(hidden_dim),
      nn.ReLU(),
    )

    # TODO: Experiment with MaxPool1d

    self.eval_encoder = nn.Sequential(
      nn.Conv1d(1, 32, kernel_size=3, padding=1),
      nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.Conv1d(32, 64, kernel_size=3, padding=1),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
      nn.BatchNorm1d(hidden_dim),
      nn.ReLU(),
    )

    # TODO: Experiment with MaxPool1d

    self.time_encoder = nn.Sequential(
      nn.Conv1d(1, 32, kernel_size=3, padding=1),
      nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.Conv1d(32, 64, kernel_size=3, padding=1),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
      nn.BatchNorm1d(hidden_dim),
      nn.ReLU(),
    )

    self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=40)

    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    self.classifier = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 4))

  def forward(self, moves: torch.Tensor, evals: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    BS, T, C, H, W = moves.shape

    moves = moves.permute(0, 2, 1, 3, 4)
    moves = self.move_encoder(moves)
    moves = moves.permute(0, 2, 1, 3, 4).reshape(BS, T, -1)

    evals = self.eval_encoder(evals.unsqueeze(1)).permute(0, 2, 1)
    times = self.time_encoder(times.unsqueeze(1)).permute(0, 2, 1)

    combined = torch.cat([moves, evals, times], dim=-1)
    combined = self.positional_encoding(combined)

    output = self.transformer_encoder(combined)
    output = torch.mean(output, dim=1)

    return self.classifier(output)


class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.pe[:, : x.size(1)]
    return self.dropout(x)


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
