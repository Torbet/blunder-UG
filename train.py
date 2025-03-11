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
from model import ConvLSTM, Transformer, Dense1, Dense3, Dense6, Conv1, Conv3, Conv6
from sklearn import metrics
import csv

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

parser = argparse.ArgumentParser(description='')
parser.add_argument(
  '--model', type=str, default='transformer', choices=['conv1', 'conv3', 'conv6', 'dense1', 'dense3', 'dense6', 'convlstm', 'transformer']
)
parser.add_argument('--evals', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--times', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--limit', type=int, default=1000)
parser.add_argument('--data', type=str, default='processed', choices=['synthetic', 'processed'])
parser.add_argument('--num-moves', type=int, default=40)
parser.add_argument('--engine-prob', type=float, default=0.3)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--channels', type=int, default=6, choices=[6, 12])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()
print(' '.join(f'{k}={v}' for k, v in vars(args).items()))


def train(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer) -> dict[str, float]:
  model.train()
  for moves, evals, times, game_labels in (t := tqdm(loader)):
    optimizer.zero_grad()
    output = model(moves, evals, times)
    loss = F.cross_entropy(output, game_labels, label_smoothing=0.1)
    loss.backward()
    optimizer.step()
    t.set_description(f'Loss: {loss.item():.4f}')
  return evaluate(model, loader)


def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
  model.eval()
  loss, labels, preds, probs = 0, [], [], []
  with torch.no_grad():
    for moves, evals, times, game_labels in (t := tqdm(loader)):
      output = model(moves, evals, times)
      labels.extend(game_labels.tolist())
      preds.extend(output.argmax(dim=1).tolist())
      probs.append(F.softmax(output, dim=1).cpu().numpy())  # Store softmax probabilities
      loss += F.cross_entropy(output, game_labels, reduction='sum').item()
      t.set_description(f'Loss: {loss / len(loader.dataset):.4f}')
  probs = np.vstack(probs)
  return {
    'loss': loss / len(loader.dataset),
    'accuracy': metrics.accuracy_score(labels, preds),
    **{k: v for k, v in zip(['precision', 'recall', 'f1'], metrics.precision_recall_fscore_support(labels, preds, average='weighted'))},
    'confusion': metrics.confusion_matrix(labels, preds),
    'auc': metrics.roc_auc_score(labels, probs, average='weighted', multi_class='ovr'),  # Use probs instead of preds
  }


if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = {
    'dense1': Dense1(channels=args.channels).to(device),
    'dense3': Dense3(channels=args.channels).to(device),
    'dense6': Dense6(channels=args.channels).to(device),
    'conv1': Conv1(channels=args.channels).to(device),
    'conv3': Conv3(channels=args.channels).to(device),
    'conv6': Conv6(channels=args.channels).to(device),
    'convlstm': ConvLSTM(channels=args.channels, evals=args.evals, times=args.times).to(device),
    'transformer': Transformer(channels=args.channels, evals=args.evals, times=args.times).to(device),
  }[args.model]
  name = model.__class__.__name__
  if isinstance(model, Transformer) or isinstance(model, ConvLSTM):
    if model.evals or model.times:
      name += f'({'evals' if model.evals else ''}{', ' if model.evals and model.times else ''}{'times' if model.times else ''})'
  print(f'Model: {name}')
  if args.data == 'synthetic':
    dataset = SyntheticDataset(args.limit, args.num_moves, args.engine_prob, device)
  else:
    dataset = ProcessedDataset(args.limit, args.num_moves, args.channels, device)
  train_loader, val_loader, test_loader = split_data(dataset, args.batch_size)
  optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)

  results = {}
  for epoch in range(args.epochs):
    print(f'Epoch {epoch + 1}')
    train_results = train(model, train_loader, optimizer)
    val_results = evaluate(model, val_loader)
    results[epoch] = {'train': train_results, 'val': val_results}
    print(f'Train Loss: {train_results["loss"]:.4f}, Train Accuracy: {train_results["accuracy"]:.4f}')
    print(f'Train Precision: {train_results["precision"]:.4f}, Train Recall: {train_results["recall"]:.4f}, Train F1: {train_results["f1"]:.4f}')
    print(f'Val Loss: {val_results["loss"]:.4f}, Val Accuracy: {val_results["accuracy"]:.4f}')
    print(f'Val Precision: {val_results["precision"]:.4f}, Val Recall: {val_results["recall"]:.4f}, Val F1: {val_results["f1"]:.4f}')
    print(f'Val Confusion Matrix:\n{val_results["confusion"]}')
    print(f'Val AUC: {val_results["auc"]:.4f}')
    # scheduler.step()

  test_results = evaluate(model, test_loader)
  print(f'Test Loss: {test_results["loss"]:.4f}, Test Accuracy: {test_results["accuracy"]:.4f}')
  print(f'Test Precision: {test_results["precision"]:.4f}, Test Recall: {test_results["recall"]:.4f}, Test F1: {test_results["f1"]:.4f}')
  print(f'Test Confusion Matrix:\n{test_results["confusion"]}')
  print(f'Test AUC: {test_results["auc"]:.4f}')

  if args.save:
    name = f'{name}_{args.data}_{args.limit}_{args.num_moves}_{args.engine_prob}_{args.batch_size}_{args.lr}_{args.weight_decay}_{args.epochs}'
    torch.save(model.state_dict(), f'results/weights/{name}.pt')

    keys = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc']
    with open(f'results/{name}.csv', 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(['epoch', 'type'] + keys)
      for epoch in range(args.epochs):
        for key in ['train', 'val']:
          writer.writerow([epoch, key] + [results[epoch][key][k] for k in keys])
      writer.writerow(['+', 'test'] + [test_results[k] for k in keys])

    for key in keys:
      plt.plot([results[epoch]['train'][key] for epoch in range(args.epochs)], label=f'Train {key}')
      plt.plot([results[epoch]['val'][key] for epoch in range(args.epochs)], label=f'Val {key}', linestyle='dashed')
      plt.title(key)
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title(f'{name} Metrics')
    plt.legend()
    plt.savefig(f'results/plots/{name}.png')
