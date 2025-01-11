import argparse
import torch
import torch.nn.functional as F
from flask import Flask, render_template
import chess
from model import ConvLSTM
from utils import maia_move, stockfish_move, parse_board_12, evaluate_board

parser = argparse.ArgumentParser(description='')
parser.add_argument('--limit', type=int, default=1000)
parser.add_argument('--num-moves', type=int, default=40)
parser.add_argument('--engine-prob', type=float, default=0.3)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()
print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

model = ConvLSTM()
weights = f'results/weights/{args.limit}_{args.num_moves}_{args.engine_prob}_{args.batch_size}_{args.lr}_{args.epochs}.pt'
model.load_state_dict(torch.load(weights, weights_only=True))
model.eval()

total_moves = 0
moves = torch.zeros((60, 12, 8, 8), dtype=torch.float32)
evals = torch.zeros(60, dtype=torch.float32)
preds = torch.zeros((60, 4), dtype=torch.float32)

app = Flask(__name__, static_folder='web/static', template_folder='web/templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 600


@app.route('/')
def index():
  return render_template('index.html')


@app.route('/img/<path:path>')
def img(path):
  name = path.split('/')[-1]
  return app.send_static_file(f'pieces/{name}')


@app.route('/maia/<path:fen>')
def maia(fen):
  board = chess.Board(fen)
  board.push(maia_move(board))
  return board.fen()


@app.route('/stockfish/<path:fen>')
def stockfish(fen):
  board = chess.Board(fen)
  board.push(stockfish_move(board))
  return board.fen()


@app.route('/move/<path:fen>')
def move(fen):
  global total_moves
  board = chess.Board(fen)
  moves[total_moves] = torch.tensor(parse_board_12(board))
  evals[total_moves] = torch.tensor(evaluate_board(board))
  preds[total_moves] = F.softmax(model(moves.unsqueeze(0), evals.unsqueeze(0))[0], dim=0)
  total_moves += 1
  return str(preds[total_moves - 1, 2].item())


if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)
