import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(0)
torch.cuda.manual_seed(0)


class Transformer(nn.Module):
  def __init__(
    self,
    channels: int = 12,
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

    # d_model is computed as hidden_dim multiplied by number of modalities
    d_model = hidden_dim * (1 + int(evals) + int(times))

    # Move encoder: processes 3D (spatial + temporal) move data.
    self.move_encoder = nn.Sequential(
      nn.Conv3d(channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
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

    # Evaluation encoder: processes per-move evaluation values.
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

    # Time encoder: processes per-move time information.
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

    # CLS token: a learnable parameter to aggregate sequence information.
    self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    # Learnable positional embeddings: note the extra position for the CLS token.
    self.pos_embedding = nn.Parameter(torch.randn(1, num_moves + 1, d_model))
    self.dropout = nn.Dropout(dropout)

    # Transformer encoder: processes the entire sequence.
    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    # Classifier head: produces final 4-class output.
    self.classifier = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 4))

  def forward(self, moves: torch.Tensor, evals: torch.Tensor = None, times: torch.Tensor = None) -> torch.Tensor:
    # moves: (BS, T, C, H, W)
    BS, T, C, H, W = moves.shape

    # Process moves with the convolutional encoder.
    # Rearranging so that time is in the channel dimension for Conv3d: (BS, C, T, H, W)
    moves = moves.permute(0, 2, 1, 3, 4)
    moves = self.move_encoder(moves)
    # Rearranging back: (BS, T, features)
    # Note: After pooling, the spatial dims may be reduced; we flatten these along with the channel.
    moves = moves.permute(0, 2, 1, 3, 4).reshape(BS, T, -1)

    # Start with move features.
    combined = moves

    # Process evals if provided.
    if self.evals and evals is not None:
      # evals expected shape: (BS, T). After unsqueeze: (BS, 1, T)
      evals_enc = self.eval_encoder(evals.unsqueeze(1)).permute(0, 2, 1)
      combined = torch.cat([combined, evals_enc], dim=-1)

    # Process times if provided.
    if self.times and times is not None:
      # times expected shape: (BS, T). After unsqueeze: (BS, 1, T)
      times_enc = self.time_encoder(times.unsqueeze(1)).permute(0, 2, 1)
      combined = torch.cat([combined, times_enc], dim=-1)

    # Prepend the CLS token to each sequence.
    cls_tokens = self.cls_token.expand(BS, -1, -1)  # (BS, 1, d_model)
    combined = torch.cat([cls_tokens, combined], dim=1)  # (BS, T + 1, d_model)

    # Add learnable positional embeddings and apply dropout.
    combined = combined + self.pos_embedding
    combined = self.dropout(combined)

    # Pass the sequence through the transformer encoder.
    output = self.transformer_encoder(combined)
    # Use the CLS token (first token) as the aggregated representation.
    cls_output = output[:, 0]

    return self.classifier(cls_output)


class ConvLSTM(nn.Module):
  def __init__(self, channels: int = 12, evals: bool = True, times: bool = True):
    super().__init__()
    self.evals = evals
    self.times = times

    self.conv1 = nn.Conv2d(channels, 64, 3)
    self.conv2 = nn.Conv2d(64, 128, 3)
    self.conv3 = nn.Conv2d(128, 256, 3)
    dim = 1024 + evals + times
    self.lstm = nn.LSTM(dim, 512, batch_first=True)
    # self.fc = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 4))
    self.fc = nn.Linear(512, 4)

  def forward(self, moves: torch.Tensor, evals: torch.Tensor = None, times: torch.Tensor = None) -> torch.Tensor:
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


class Dense1(nn.Module):
  def __init__(self, channels: int = 12):
    super().__init__()
    in_dim = 15360 * (channels // 6)
    self.l1 = nn.Linear(in_dim, 512)
    self.l2 = nn.Linear(512, 4)

  def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
    x = x.view(x.size(0), -1)
    x = F.relu(self.l1(x))
    return self.l2(x)


class Dense3(nn.Module):
  def __init__(self, channels: int = 12):
    super().__init__()
    in_dim = 15360 * (channels // 6)
    # hiden layers: 512, 512, 64
    self.l1 = nn.Linear(in_dim, 512)
    self.l2 = nn.Linear(512, 512)
    self.l3 = nn.Linear(512, 64)
    self.l4 = nn.Linear(64, 4)

  def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
    x = x.view(x.size(0), -1)
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    x = F.relu(self.l3(x))
    return self.l4(x)


class Dense6(nn.Module):
  def __init__(self, channels: int = 12):
    super().__init__()
    in_dim = 15360 * (channels // 6)
    # hidden layers: 2048, 2048, 512, 512, 128, 128
    self.l1 = nn.Linear(in_dim, 2048)
    self.l2 = nn.Linear(2048, 2048)
    self.l3 = nn.Linear(2048, 512)
    self.l4 = nn.Linear(512, 512)
    self.l5 = nn.Linear(512, 128)
    self.l6 = nn.Linear(128, 128)
    self.l7 = nn.Linear(128, 4)

  def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
    x = x.view(x.size(0), -1)
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    x = F.relu(self.l3(x))
    x = F.relu(self.l4(x))
    x = F.relu(self.l5(x))
    x = F.relu(self.l6(x))
    return self.l7(x)


class Conv1(nn.Module):
  def __init__(self, channels: int = 12):
    super().__init__()
    self.conv1 = nn.Conv3d(channels, 64, kernel_size=(5, 3, 3), stride=(2, 1, 1))
    self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 1, 1))
    in_dim = 16384 * (channels // 6)
    self.l1 = nn.Linear(in_dim, 512)
    self.l2 = nn.Linear(512, 4)

  def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
    x = x.permute(0, 2, 1, 3, 4)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = x.view(x.size(0), -1)
    x = F.relu(self.l1(x))
    return self.l2(x)


class Conv3(nn.Module):
  def __init__(self, channels: int = 12):
    super().__init__()
    self.conv1 = nn.Conv3d(channels, 64, kernel_size=(5, 3, 3), stride=(2, 1, 1))
    self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 1, 1))
    in_dim = 16384 * (channels // 6)
    self.l1 = nn.Linear(in_dim, 512)
    self.l2 = nn.Linear(512, 512)
    self.l3 = nn.Linear(512, 64)
    self.l4 = nn.Linear(64, 4)

  def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
    x = x.permute(0, 2, 1, 3, 4)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = x.view(x.size(0), -1)
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    x = F.relu(self.l3(x))
    return self.l4(x)


class Conv6(nn.Module):
  def __init__(self, channels: int = 12):
    super().__init__()
    self.conv1 = nn.Conv3d(channels, 64, kernel_size=(5, 3, 3), stride=(2, 1, 1))
    self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 1, 1))
    in_dim = 16384 * (channels // 6)
    self.l1 = nn.Linear(in_dim, 2048)
    self.l2 = nn.Linear(2048, 2048)
    self.l3 = nn.Linear(2048, 512)
    self.l4 = nn.Linear(512, 512)
    self.l5 = nn.Linear(512, 128)
    self.l6 = nn.Linear(128, 128)
    self.l7 = nn.Linear(128, 4)

  def forward(self, x: torch.Tensor, *_) -> torch.Tensor:
    x = x.permute(0, 2, 1, 3, 4)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = x.view(x.size(0), -1)
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    x = F.relu(self.l3(x))
    x = F.relu(self.l4(x))
    x = F.relu(self.l5(x))
    x = F.relu(self.l6(x))
    return self.l7(x)
