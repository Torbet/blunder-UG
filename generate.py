import argparse
import numpy as np
import chess
import csv
from utils import parse_board_12, parse_board_6, evaluate_board, maia_move, stockfish_move, generate_move_times
from tqdm import trange

np.random.seed(0)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--limit', type=int, default=1000)
parser.add_argument('--num-moves', type=int, default=60)
parser.add_argument('--engine-prob', type=float, default=0.3)
parser.add_argument('--channels', type=int, default=12, choices=[6, 12])
args = parser.parse_args()
print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

limit = args.limit  # number of games per class
num_moves = args.num_moves  # number of moves per game
engine_prob = args.engine_prob  # probability of engine move
channels = args.channels  # number of channels
parse_board = parse_board_12 if channels == 12 else parse_board_6
ratings = [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]

with open('data/misc/openings.tsv', 'r') as f:
  reader = csv.reader(f, delimiter='\t')
  openings = [l[4] for l in reader if 5 <= len(l[3].split(' ')) <= 10]

moves = np.zeros((4 * limit, num_moves, channels, 8, 8), dtype=int)
evals = np.zeros((4 * limit, num_moves), dtype=float)
times = np.zeros((4 * limit, num_moves), dtype=float)
move_labels = np.zeros((4 * limit, num_moves), dtype=int)
game_labels = np.zeros(4 * limit, dtype=int)


for l, c in enumerate(['HvH', 'HvE', 'EvH', 'EvE']):
  """
    0: maia vs maia
    1: maia vs stockfish
    2: stockfish vs maia
    3: stockfish vs stockfish
  """
  for g in (t := trange(limit)):
    idx = 4 * g + l
    game_labels[idx] = l
    board = chess.Board(openings[np.random.randint(len(openings))])
    for m in range(num_moves):
      if board.is_game_over():
        break

      r = np.random.rand()
      rating = np.random.choice(ratings)
      move_label = int(r < engine_prob)

      match l:
        case 0:
          move, label = maia_move(board, rating), 0
        case 1:
          if board.turn == chess.WHITE:
            move, label = maia_move(board, rating), 0
          else:
            move = stockfish_move(board) if r < engine_prob else maia_move(board, rating)
        case 2:
          if board.turn == chess.WHITE:
            move = stockfish_move(board) if r < engine_prob else maia_move(board, rating)
          else:
            move, label = maia_move(board), 0
        case 3:
          move = stockfish_move(board) if r < engine_prob else maia_move(board, rating)

      board.push(move)
      moves[idx, m] = parse_board(board)
      evals[idx, m] = evaluate_board(board)
      move_labels[idx, m] = move_label

    times[idx] = generate_move_times(evals[idx], move_labels[idx])
    t.set_description(c)

shuffle = np.random.permutation(4 * limit)
moves, evals, times, move_labels, game_labels = moves[shuffle], evals[shuffle], times[shuffle], move_labels[shuffle], game_labels[shuffle]

output_path = f'data/generated/{limit}_{num_moves}_{engine_prob}_{channels}.npz'
np.savez_compressed(output_path, moves=moves, evals=evals, times=times, move_labels=move_labels, game_labels=game_labels)
print(f'Data saved to {output_path}')
