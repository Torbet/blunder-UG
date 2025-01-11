import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import SyntheticDataset, ProcessedDataset, split_data
from model import ConvLSTM

np.random.seed(0)
torch.manual_seed(0)

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


def train(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer) -> tuple[float, float]:
  model.train()
  for moves, evals, move_labels, game_labels in (t := tqdm(loader)):
    optimizer.zero_grad()
    output = model(moves, evals)
    loss = F.cross_entropy(output, game_labels, label_smoothing=0.1)
    loss.backward()
    optimizer.step()
    t.set_description(f'Loss: {loss.item():.2f}')
  return evaluate(model, loader)


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
  model.eval()
  loss, correct, total = 0, 0, 0
  with torch.no_grad():
    for moves, evals, move_labels, game_labels in loader:
      output = model(moves, evals)
      loss += F.cross_entropy(output, game_labels, label_smoothing=0.1, reduction='sum').item()
      correct += (output.argmax(dim=1) == game_labels).sum().item()
      total += len(game_labels)
  return loss / total, correct / total


if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = ConvLSTM().to(device)
  dataset = SyntheticDataset(args.limit, args.num_moves, args.engine_prob, device)
  # dataset = ProcessedDataset(args.limit, args.num_moves, 12, device)
  train_loader, val_loader, test_loader = split_data(dataset, args.batch_size)
  optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

  results = {}
  for epoch in range(args.epochs):
    train_loss, train_accuracy = train(model, train_loader, optimizer)
    scheduler.step()
    val_loss, val_accuracy = evaluate(model, val_loader)
    print(
      f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.2f}, Train Accuracy: {train_accuracy:.2f}, Val Loss: {val_loss:.2f}, Val Accuracy: {val_accuracy:.2f}'
    )
    results[epoch] = (train_loss, train_accuracy, val_loss, val_accuracy)

  test_loss, accuracy = evaluate(model, test_loader)
  print(f'Test Loss: {test_loss:.2f}, Test Accuracy: {accuracy:.2f}')

  if args.save:
    name = f'{args.limit}_{args.num_moves}_{args.engine_prob}_{args.batch_size}_{args.lr}_{args.epochs}'
    torch.save(model.state_dict(), f'results/weights/{name}.pt')
    plt.plot(list(results.keys()), [r[0] for r in results.values()], label='Train Loss')
    plt.plot(list(results.keys()), [r[1] for r in results.values()], label='Train Accuracy')
    plt.plot(list(results.keys()), [r[2] for r in results.values()], label='Val Loss', linestyle='--')
    plt.plot(list(results.keys()), [r[3] for r in results.values()], label='Val Accuracy', linestyle='--')
    plt.legend()
    plt.savefig(f'results/plots/{name}.png')
