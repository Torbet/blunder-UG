import argparse
import numpy as np
import chess.pgn
import os, shutil
from tqdm import trange
from utils import parse_board_12, parse_board_6, evaluate_board

# suppress logging (variant: untimed)
import logging

logging.getLogger('chess.pgn').setLevel(logging.CRITICAL)

np.random.seed(0)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--limit', type=int, default=1000)
parser.add_argument('--num-moves', type=int, default=40)
parser.add_argument('--years', nargs='+', type=int, default=[2019, 2020, 2021, 2022, 2023])
parser.add_argument('--elo', nargs=2, type=int, default=[1800, 2200])
parser.add_argument('--channels', type=int, default=12, choices=[6, 12])
args = parser.parse_args()
print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

limit = args.limit  # number of games per class
num_moves = args.num_moves  # number of moves per game
years = sorted(args.years)  # years to process
elo = sorted(args.elo)  # elo range
channels = args.channels  # number of channels


def filter_games(limit: int, years: list[int], elo: list[int]):
  print('Filtering games...\n')
  paths = [f'data/raw/{year}_{mode}.pgn' for year in years[::-1] for mode in ['HvH', 'HvE', 'EvE']]
  games = {mode: [] for mode in ['HvH', 'HvE', 'EvH', 'EvE']}

  for path in paths:
    with open(path, 'r') as f:
      while game := chess.pgn.read_game(f):
        headers = game.headers
        white_comp, black_comp = [headers.get(f'{c}IsComp') is not None for c in ['White', 'Black']]
        mode = f'{["H", "E"][int(white_comp)]}v{["H", "E"][int(black_comp)]}'
        white_elo, black_elo = [int(headers.get(f'{c}Elo', 0)) for c in ['White', 'Black']]
        ply_count = int(headers.get('PlyCount', 0))

        if 'HvE' in path and len(games['HvE']) >= limit and len(games['EvH']) >= limit:
          break
        if any(m in path and len(games[m]) >= limit for m in ['HvH', 'EvE']):
          break
        if len(games[mode]) >= limit:
          continue

        if headers.get('Variant') == 'untimed':
          continue
        if not white_comp and not (elo[0] <= white_elo <= elo[1]):
          continue
        if not black_comp and not (elo[0] <= black_elo <= elo[1]):
          continue
        if ply_count < num_moves + 20:
          continue

        games[mode].append(game)
        print('\r' + ' '.join(f'{mode}: {len(games[mode])}' for mode in games), end='')

  filtered_path = f'data/filtered/{limit}_{"-".join(str(e) for e in elo)}'
  if os.path.exists(filtered_path):
    shutil.rmtree(filtered_path)
  os.makedirs(filtered_path)
  for mode in games:
    with open(f'{filtered_path}/{mode}.pgn', 'w+') as f:
      for game in games[mode]:
        f.write(str(game) + '\n\n')

  print(f'\nFiltered data saved to {filtered_path}\n')


def process_games(limit: int, num_moves: int, filtered_path: str, channels: int):
  parse_board = parse_board_12 if channels == 12 else parse_board_6
  print('Processing games...\n')
  moves = np.zeros((4 * limit, num_moves, channels, 8, 8), dtype=int)
  evals = np.zeros((4 * limit, num_moves), dtype=float)
  times = np.zeros((4 * limit, num_moves), dtype=float)
  game_labels = np.zeros(4 * limit, dtype=int)

  for l, c in enumerate(['HvH', 'HvE', 'EvH', 'EvE']):
    with open(f'{filtered_path}/{c}.pgn', 'r') as f:
      for g in trange(limit, desc=c):
        game_labels[l * limit + g] = l
        game = chess.pgn.read_game(f)
        board = game.board()

        for i, node in enumerate(game.mainline()):
          if i == num_moves + 10:
            break
          board.push(node.move)
          if i >= 10:
            moves[l * limit + g, i - 10] = parse_board(board)
            evals[l * limit + g, i - 10] = evaluate_board(board)
            times[l * limit + g, i - 10] = node.emt()

  shuffle = np.random.permutation(4 * limit)
  moves, evals, times, game_labels = moves[shuffle], evals[shuffle], times[shuffle], game_labels[shuffle]
  processed_path = f'data/processed/{limit}_{num_moves}_{channels}.npz'
  np.savez_compressed(processed_path, moves=moves, evals=evals, times=times, game_labels=game_labels)
  print(f'Processed data saved to {processed_path}')


if __name__ == '__main__':
  filtered_path = f'data/filtered/{limit}_{"-".join(str(e) for e in elo)}'
  if not os.path.exists(filtered_path):
    filter_games(limit, years, elo)
  process_games(limit, num_moves, filtered_path, channels)
