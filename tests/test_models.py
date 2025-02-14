import sys

sys.path.append('..')

import unittest
import torch
import torch.nn as nn
from model import ConvLSTM, Dense
import numpy as np

torch.manual_seed(0)
np.random.seed(0)


class TestModels(unittest.TestCase):
  def setUp(self):
    self.batch_size = 4
    self.time_steps = 40
    self.channels = 6  # Changed to match Conv model input channels
    self.board_size = 8
    self.device = torch.device('cpu')

    # Create sample inputs with correct channel ordering for Conv3d
    self.moves = torch.randn(self.batch_size, self.channels, self.time_steps, self.board_size, self.board_size)
    self.evals = torch.randn(self.batch_size, self.time_steps)
    self.game_labels = torch.randint(0, 4, (self.batch_size,))

    # Create separate input for ConvLSTM which expects 12 channels
    self.moves_12 = torch.randn(self.batch_size, self.time_steps, 12, self.board_size, self.board_size)

  def test_convlstm_architecture(self):
    """Test ConvLSTM model architecture and output shape"""
    model = ConvLSTM(times=False).to(self.device)
    output = model(self.moves_12, self.evals)

    self.assertEqual(output.shape, (self.batch_size, 4))

    self.assertIsInstance(model.conv1, nn.Conv2d)
    self.assertIsInstance(model.lstm, nn.LSTM)
    self.assertEqual(model.lstm.bidirectional, True)

    loss = output.sum()
    loss.backward()
    for param in model.parameters():
      self.assertIsNotNone(param.grad)

  def test_dense_architecture(self):
    """Test Dense model architecture and output shape"""
    hidden_layers = [512, 256, 128]
    # Reshape moves for Dense input
    moves_reshaped = self.moves.permute(0, 2, 1, 3, 4)
    input_dim = self.time_steps * self.channels * self.board_size * self.board_size
    model = Dense(input_dim, hidden_layers).to(self.device)
    output = model(moves_reshaped)

    self.assertEqual(output.shape, (self.batch_size, 4))

    current_dim = input_dim
    for i, layer in enumerate(model.layers[:-1:2]):
      self.assertEqual(layer.in_features, current_dim)
      self.assertEqual(layer.out_features, hidden_layers[i])
      current_dim = hidden_layers[i]

  def test_model_training_step(self):
    """Test single training step for each model"""
    # Test ConvLSTM
    model = ConvLSTM(times=False).to(self.device)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()
    output = model(self.moves_12, self.evals)
    loss = nn.CrossEntropyLoss()(output, self.game_labels)
    loss.backward()
    self._verify_gradients(model)

    # Test Dense
    moves_reshaped = self.moves.permute(0, 2, 1, 3, 4)
    input_dim = self.time_steps * self.channels * self.board_size * self.board_size
    model = Dense(input_dim, [512]).to(self.device)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()
    output = model(moves_reshaped)
    loss = nn.CrossEntropyLoss()(output, self.game_labels)
    loss.backward()
    self._verify_gradients(model)

  def test_convlstm_bidirectional(self):
    """Test bidirectional LSTM behavior"""
    model = ConvLSTM(times=False).to(self.device)

    forward_moves = self.moves_12
    backward_moves = torch.flip(self.moves_12, [1])
    forward_output = model(forward_moves, self.evals)
    backward_output = model(backward_moves, torch.flip(self.evals, [1]))

    self.assertTrue(torch.any(forward_output != backward_output))

  def _verify_gradients(self, model):
    """Helper method to verify gradient properties"""
    for param in model.parameters():
      self.assertIsNotNone(param.grad)
      self.assertFalse(torch.isnan(param.grad).any())


if __name__ == '__main__':
  unittest.main()
