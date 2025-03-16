import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from model import ConvLSTM, Transformer
from utils import HumanDataset
import os, glob, csv

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, default='transformer', choices=['convlstm', 'transformer'])
parser.add_argument('--channels', type=int, default=6, choices=[6, 12])
parser.add_argument('--evals', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--times', action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()
print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
  model.eval()
  loss, labels, preds, probs = 0, [], [], []
  with torch.no_grad():
    for moves, evals, times, game_labels in (t := tqdm(loader)):
      output = model(moves, evals, times)
      output = output[:, :-1]
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
  num_moves = 60
  dataset = HumanDataset(num_moves=num_moves, channels=args.channels, device=device)
  loader = torch.utils.data.DataLoader(dataset, batch_size=1)
  model = {
    'convlstm': ConvLSTM(channels=args.channels, evals=args.evals, times=args.times).to(device),
    'transformer': Transformer(channels=args.channels, num_moves=num_moves, evals=args.evals, times=args.times).to(device),
  }[args.model].to(device)
  name = model.__class__.__name__
  if isinstance(model, Transformer) or isinstance(model, ConvLSTM):
    if model.evals or model.times:
      name += f'({'evals' if model.evals else ''}{', ' if model.evals and model.times else ''}{'times' if model.times else ''})'

  # get models with all learning rates and weight decays, variables are name and channels, make sure data is 'generated'
  pattern = os.path.join('results/weights', f'{name}_generated_{args.channels}*.pt')
  files = glob.glob(pattern)

  for file in files:
    print(f'Evaluating {file}')
    model.load_state_dict(torch.load(file, weights_only=True, map_location=device))
    m = evaluate(model, loader)
    dataset = 'generated' if 'generated' in file else 'processed'
    new_name = file.replace('weights/', '').replace('.pt', '.csv').replace('generated', 'human').replace('processed', 'human')
    with open(new_name, 'w') as f:
      # epoch, type, loss, accuracy, precision, recall, f1, auc
      # ignore confusion matrix
      writer = csv.writer(f)
      writer.writerow(['epoch', 'type', 'loss', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'dataset'])
      writer.writerow([0, 'test', m['loss'], m['accuracy'], m['precision'], m['recall'], m['f1'], m['auc'], dataset])

  print('Done')
