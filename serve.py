import argparse
import random

# import torch
# import torch.nn.functional as F
from flask import Flask, render_template, request, session
import chess
import csv
import sqlite3
import os
from model import ConvLSTM, Transformer
from utils import maia_move, stockfish_move, parse_board_12, evaluate_board

parser = argparse.ArgumentParser(description='')
parser.add_argument('--limit', type=int, default=10000)
parser.add_argument('--num-moves', type=int, default=60)
parser.add_argument('--engine-prob', type=float, default=0.5)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()
print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

with open('data/misc/openings.tsv', 'r') as f:
  reader = csv.reader(f, delimiter='\t')
  openings = [l[4] for l in reader if 5 <= len(l[3].split(' ')) <= 10]

"""
model = Transformer(times=False)
weights = f'results/weights/Transformer_{args.limit}_{args.num_moves}_{args.engine_prob}_{args.batch_size}_{args.lr}_{args.epochs}.pt'
model.load_state_dict(torch.load(weights, weights_only=True))
model.eval()
"""


def store_data(fens, labels):
  con = sqlite3.connect('web/games.db')
  cur = con.cursor()
  cur.execute('CREATE TABLE IF NOT EXISTS games (id INTEGER PRIMARY KEY, fens TEXT, labels TEXT)')
  cur.execute('INSERT INTO games (fens, labels) VALUES (?, ?)', ('_'.join(fens), '_'.join(map(str, labels))))
  con.commit()
  con.close()


app = Flask(__name__, static_folder='web/static', template_folder='web/templates')
app.secret_key = 'super secret key'
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 600


@app.route('/')
def index():
  return render_template('index.html')


@app.route('/play')
def play():
  board = chess.Board(openings[random.randint(0, len(openings) - 1)])
  while board.turn != chess.WHITE:
    board = chess.Board(openings[random.randint(0, len(openings) - 1)])
  session['fens'] = [board.fen()]
  session['labels'] = [0]
  return render_template('play.html', fen=board.fen())


@app.route('/img/<path:path>')
def img(path):
  name = path.split('/')[-1]
  return app.send_static_file(f'pieces/{name}')


@app.route('/maia/<path:fen>')
def maia(fen):
  labels = session['labels']
  labels.append(0)
  session['labels'] = labels
  board = chess.Board(fen)
  if board.is_game_over():
    return 'Game Over'
  board.push(maia_move(board, request.args.get('rating', default=1200, type=int)))
  return board.fen()


@app.route('/stockfish/<path:fen>')
def stockfish(fen):
  labels = session['labels']
  labels.append(1)
  session['labels'] = labels
  board = chess.Board(fen)
  board.push(stockfish_move(board))
  return board.fen()


@app.route('/move/<path:fen>')
def move(fen):
  print(session)
  board = chess.Board(fen)
  # add fen
  fens = session['fens']
  fens.append(board.fen())
  session['fens'] = fens

  if len(session['fens']) > len(session['labels']):
    labels = session['labels']
    labels.append(0)
    session['labels'] = labels
  if board.is_game_over():
    store_data(session['fens'], session['labels'])
    print('Game Over')
    return '0'
  return str(evaluate_board(board))


if __name__ == '__main__':
  ssl_path = '/etc/letsencrypt/live/chess.torbet.co'
  if os.path.exists(ssl_path):
    app.run(
      host='0.0.0.0',
      port=443,
      ssl_context=(f'{ssl_path}/fullchain.pem', f'{ssl_path}/privkey.pem'),
    )
  else:
    app.run(host='0.0.0.0', debug=True)
