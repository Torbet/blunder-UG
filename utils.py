import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import chess
from stockfish import Stockfish
from lczero.backends import Backend, Weights, GameState
import os

np.random.seed(0)
torch.manual_seed(0)


class SyntheticDataset(Dataset):
  def __init__(self, limit: int, num_moves: int, engine_prob: float, device: torch.device = torch.device('cpu')):
    self.device = device

    data = np.load(os.path.join(os.path.dirname(__file__), f'data/generated/{limit}_{num_moves}_{engine_prob}.npz'))
    shuffle = np.random.permutation(4 * limit)
    self.moves = data['moves'][shuffle]
    self.evals = data['evals'][shuffle]
    self.move_labels = data['move_labels'][shuffle]
    self.game_labels = data['game_labels'][shuffle]

  def __len__(self):
    return len(self.moves)

  def __getitem__(self, idx):
    return (
      torch.tensor(self.moves[idx], device=self.device, dtype=torch.float32),
      torch.tensor(self.evals[idx], device=self.device, dtype=torch.float32),
      torch.tensor(self.move_labels[idx], device=self.device, dtype=torch.long),
      torch.tensor(self.game_labels[idx], device=self.device, dtype=torch.long),
    )


class ProcessedDataset(Dataset):
  def __init__(self, limit: int, num_moves: int, channels: int, device: torch.device = torch.device('cpu')):
    self.device = device

    data = np.load(os.path.join(os.path.dirname(__file__), f'data/processed/{limit}_{num_moves}_{channels}.npz'))
    shuffle = np.random.permutation(4 * limit)
    self.moves = data['moves'][shuffle]
    self.evals = data['evals'][shuffle]
    self.times = data['times'][shuffle]
    self.game_labels = data['game_labels'][shuffle]

  def __len__(self):
    return len(self.moves)

  def __getitem__(self, idx):
    return (
      torch.tensor(self.moves[idx], device=self.device, dtype=torch.float32),
      torch.tensor(self.evals[idx], device=self.device, dtype=torch.float32),
      torch.tensor(self.times[idx], device=self.device, dtype=torch.float32),
      torch.tensor(self.game_labels[idx], device=self.device, dtype=torch.long),
    )


def split_data(dataset: Dataset, batch_size: int = 64) -> tuple[DataLoader, DataLoader, DataLoader]:
  n = len(dataset)
  t = int(0.8 * n)
  sizes = (t, (n - t) // 2, (n - t) // 2)
  train, val, test = random_split(dataset, sizes)
  return (DataLoader(ds, batch_size=batch_size) for ds in (train, val, test))


stockfish = Stockfish()


def evaluate_board(board: chess.Board) -> float:
  stockfish.set_depth(8)
  stockfish.set_fen_position(board.fen())
  evaluation = stockfish.get_evaluation()
  if evaluation['type'] == 'mate':
    return 1000 if evaluation['value'] > 0 else -1000
  return evaluation['value']


def stockfish_move(board: chess.Board) -> chess.Move:
  stockfish.set_depth(16)
  stockfish.set_fen_position(board.fen())
  return chess.Move.from_uci(stockfish.get_best_move())


maia = Backend(Weights(os.path.join(os.path.dirname(__file__), 'data/misc/maia_weights.pb.gz')))


def maia_move(board: chess.Board) -> chess.Move:
  state = GameState(fen=board.fen())
  output = maia.evaluate(state.as_input(maia))[0]
  moves = sorted(list(zip(state.moves(), output.p_softmax(*state.policy_indices()))), key=lambda x: x[1], reverse=True)
  return chess.Move.from_uci(moves[0][0])


def parse_board_12(board: chess.Board) -> np.ndarray:
  b = np.zeros((12, 8, 8), dtype=int)
  for i in range(64):
    if (piece := board.piece_at(i)) is not None:
      channel = piece.piece_type - 1 if piece.color == chess.WHITE else piece.piece_type + 5
      b[channel, i // 8, i % 8] = 1
  return b


def parse_board_6(board: chess.Board) -> np.ndarray:
  b = np.zeros((6, 8, 8), dtype=int)
  for i in range(64):
    if (piece := board.piece_at(i)) is not None:
      sign = 1 if piece.color == chess.WHITE else -1
      channel = piece.piece_type - 1
      b[channel, i // 8, i % 8] = sign
  return b
