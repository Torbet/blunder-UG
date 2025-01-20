import sys

sys.path.append('..')
import unittest
import numpy as np
import chess
from utils import parse_board_6, parse_board_12, SyntheticDataset, ProcessedDataset, split_data, stockfish_move, maia_move, evaluate_board


class TestUtils(unittest.TestCase):
  def test_parse_board_6(self):
    board = chess.Board()
    parsed = parse_board_6(board)
    self.assertEqual(parsed.shape, (6, 8, 8))
    self.assertEqual(parsed[4, 0, 3], 1)  # White king
    self.assertEqual(parsed[5, 0, 4], 1)  # White queen
    self.assertEqual(parsed[4, 7, 3], -1)  # Black king
    self.assertEqual(parsed[5, 7, 4], -1)  # Black queen

  def test_parse_board_12(self):
    board = chess.Board()
    parsed = parse_board_12(board)
    self.assertEqual(parsed.shape, (12, 8, 8))
    self.assertEqual(parsed[4, 0, 3], 1)  # White king
    self.assertEqual(parsed[5, 0, 4], 1)  # White queen
    self.assertEqual(parsed[10, 7, 3], 1)  # Black king
    self.assertEqual(parsed[11, 7, 4], 1)  # Black queen

  def test_board_parsing_edge_cases(self):
    # Test empty board
    empty_board = chess.Board.empty()
    parsed_empty_12 = parse_board_12(empty_board)
    self.assertTrue(np.all(parsed_empty_12 == 0))

    # Test endgame positions
    board = chess.Board()
    board.set_fen('8/4k3/8/8/8/8/4K3/8 w - - 0 1')  # King vs King
    parsed = parse_board_12(board)
    self.assertEqual(np.sum(parsed), 2)  # Only two pieces

  def test_synthetic_dataset(self):
    dataset = SyntheticDataset(100, 60, 0.3, 'cpu')
    self.assertEqual(len(dataset), 4 * 100)

    moves, evals, move_labels, game_labels = dataset[0]
    self.assertEqual(moves.shape, (60, 12, 8, 8))
    self.assertEqual(evals.shape, (60,))
    self.assertEqual(move_labels.shape, (60,))
    self.assertEqual(game_labels.shape, ())

  def test_processed_dataset(self):
    dataset_12 = ProcessedDataset(100, 40, 12, 'cpu')
    dataset_6 = ProcessedDataset(100, 40, 6, 'cpu')
    self.assertEqual(len(dataset_12), 4 * 100)
    self.assertEqual(len(dataset_6), 4 * 100)

    moves, evals, times, game_labels = dataset_12[0]
    self.assertEqual(moves.shape, (40, 12, 8, 8))
    self.assertEqual(evals.shape, (40,))
    self.assertEqual(times.shape, (40,))
    self.assertEqual(game_labels.shape, ())

    moves, evals, times, game_labels = dataset_6[0]
    self.assertEqual(moves.shape, (40, 6, 8, 8))
    self.assertEqual(evals.shape, (40,))
    self.assertEqual(times.shape, (40,))
    self.assertEqual(game_labels.shape, ())

  def test_engine_moves(self):
    board = chess.Board()
    self.assertTrue(board.is_legal(stockfish_move(board)))
    self.assertTrue(board.is_legal(maia_move(board)))

  def test_eval_board(self):
    board = chess.Board()
    evaluation = evaluate_board(board)
    self.assertGreater(evaluation, -1)
    board.push_san('e4')
    self.assertNotEqual(evaluation, evaluate_board(board))

  def test_split_data(self):
    dataset = SyntheticDataset(100, 60, 0.3, 'cpu')
    train, val, test = split_data(dataset, 10)
    self.assertEqual(len(train), 32)
    self.assertEqual(len(val), 4)
    self.assertEqual(len(test), 4)


if __name__ == '__main__':
  unittest.main()
