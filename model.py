import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)


class Transformer(nn.Module):
  def __init__(
    self,
    in_channels: int = 12,
    hidden_dim: int = 256,
    nhead: int = 8,
    num_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    num_moves: int = 60,
    evals: bool = True,
    times: bool = True,
  ):
    super().__init__()

    self.evals = evals
    self.times = times

    d_model = hidden_dim * (1 + evals + times)

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
    if self.evals:
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
    if self.times:
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

    self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=num_moves)

    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    self.classifier = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 4))

  def forward(self, moves: torch.Tensor, evals: torch.Tensor = None, times: torch.Tensor = None) -> torch.Tensor:
    BS, T, C, H, W = moves.shape

    moves = moves.permute(0, 2, 1, 3, 4)
    moves = self.move_encoder(moves)
    moves = moves.permute(0, 2, 1, 3, 4).reshape(BS, T, -1)

    combined = moves

    if self.evals:
      evals = self.eval_encoder(evals.unsqueeze(1)).permute(0, 2, 1)
      combined = torch.cat([combined, evals], dim=-1)
    if self.times:
      times = self.time_encoder(times.unsqueeze(1)).permute(0, 2, 1)
      combined = torch.cat([combined, times], dim=-1)

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
  def __init__(self, in_channels: int = 12, evals: bool = True, times: bool = True):
    super().__init__()
    self.evals = evals
    self.times = times

    self.conv1 = nn.Conv2d(in_channels, 64, 3)
    self.conv2 = nn.Conv2d(64, 128, 3)
    self.conv3 = nn.Conv2d(128, 256, 3)
    dim = 1024 + evals + times
    self.lstm = nn.LSTM(dim, 256, batch_first=True, bidirectional=True)
    # self.fc = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 4))
    self.fc = nn.Linear(512, 4)

  def forward(self, moves: torch.Tensor, evals: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    BS, T, C, H, W = moves.shape
    x = moves.view(BS * T, C, H, W)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(BS, T, -1)
    if self.evals:
      x = torch.cat([x, evals.unsqueeze(-1)], dim=-1)
    if self.times:
      x = torch.cat([x, times.unsqueeze(-1)], dim=-1)
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
    x = x.reshape(x.size(0), -1)
    return self.layers(x)


class Conv(nn.Module):
  def __init__(self, channels: int = 12):
    super().__init__()
    self.conv1 = nn.Conv3d(channels, 64, kernel_size=(5, 3, 3), stride=(2, 1, 1))
    self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 1, 1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.permute(0, 2, 1, 3, 4)
    x = F.relu(self.conv1(x))
    return F.relu(self.conv2(x))


class Dense1(nn.Module):
  def __init__(self, channels: int = 12):
    super().__init__()
    dim = 15360 * (channels // 6)
    self.dense = Dense(dim, [512])

  def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
    return self.dense(x)


class Dense3(nn.Module):
  def __init__(self, channels: int = 12):
    super().__init__()
    dim = 15360 * (channels // 6)
    self.dense = Dense(dim, [512, 512, 64])

  def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
    return self.dense(x)


class Dense6(nn.Module):
  def __init__(self, channels: int = 12):
    super().__init__()
    dim = 15360 * (channels // 6)
    self.dense = Dense(dim, [2048, 2048, 512, 512, 128, 128])

  def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
    return self.dense(x)


class Conv1(nn.Module):
  def __init__(self, channels: int = 12):
    super().__init__()
    self.conv = Conv(channels)
    self.dense = Dense1()

  def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
    x = self.conv(x)
    x = x.reshape(x.size(0), -1)
    return self.dense(x)
