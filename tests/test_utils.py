import sys

sys.path.append('..')
import unittest
import chess
from utils import parse_board_6, parse_board_12, SyntheticDataset, ProcessedDataset, split_data


class TestUtils(unittest.TestCase):
  def test_parse_board_6(self):
    board = chess.Board()
    parsed = parse_board_6(board)
    self.assertEqual(parsed.shape, (6, 8, 8))

  def test_parse_board_12(self):
    board = chess.Board()
    parsed = parse_board_12(board)
    self.assertEqual(parsed.shape, (12, 8, 8))

  def test_synthetic_dataset(self):
    dataset = SyntheticDataset(100, 40, 0.3, 'cpu')
    self.assertEqual(len(dataset), 4 * 100)

    moves, evals, move_labels, game_labels = dataset[0]
    self.assertEqual(moves.shape, (40, 12, 8, 8))
    self.assertEqual(evals.shape, (40,))
    self.assertEqual(move_labels.shape, (40,))
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


if __name__ == '__main__':
  unittest.main()
