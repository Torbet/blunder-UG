import numpy as np
import chess
import sqlite3
from utils import parse_board_12, parse_board_6, evaluate_board, generate_move_times

channels = 6
parse_board = parse_board_12 if channels == 12 else parse_board_6

conn = sqlite3.connect('games.db')
c = conn.cursor()
c.execute('SELECT * FROM games')
games = c.fetchall()
conn.close()

n_games = len(games)
n_moves = max(len(game[2].split('_')) for game in games)

moves = np.zeros((n_games, n_moves, channels, 8, 8), dtype=int)
evals = np.zeros((n_games, n_moves), dtype=float)
times = np.zeros((n_games, n_moves), dtype=float)
move_labels = np.zeros((n_games, n_moves), dtype=int)
game_labels = np.zeros(n_games, dtype=int)

for i, game in enumerate(games):
  # id, rating, fens, labels
  rating = game[1]
  fens = game[2].split('_')
  labels = game[3].split('_')
  side = chess.Board(fens[0]).turn

  for j, (fen, label) in enumerate(zip(fens, labels)):
    board = chess.Board(fen)
    moves[i, j] = parse_board(board)
    evals[i, j] = evaluate_board(board)
    move_labels[i, j] = int(label)
  times[i] = generate_move_times(evals[i], move_labels[i])
  if any(move_labels[i, 1::2] == 1):
    game_labels[i] = 2 if side == chess.WHITE else 1
  else:
    game_labels[i] = 0

np.savez_compressed(f'data/human/games_{channels}.npz', moves=moves, evals=evals, times=times, move_labels=move_labels, game_labels=game_labels)
