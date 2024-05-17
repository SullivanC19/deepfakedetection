import torch
import torch.utils
import torch.utils.data
import numpy as np

from .constants import TRAIN_ITERS, TRAIN_BATCH_SIZE

def get_acc(y_pred, y):
  acc = 0
  y_pred = np.array([0 if pred < 0.5 else 1 for pred in y_pred])
  for i in range(len(y_pred)):
    if (y[i] == 0 and y_pred[i] == 0) or (y[i] == 1 and y_pred[i] == 1):
      acc += 1
  acc /= len(y_pred)
  return acc

def train(model: torch.nn.Sequential, dataset: torch.utils.data.Dataset):
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
  losses = []
  accuracies = []

  for epoch in range(TRAIN_ITERS):
    print(f"Epoch {epoch + 1}/{TRAIN_ITERS}")
    acc = 0
    num_batches = 0
    for _, data in enumerate(dataloader):
      x, y = data
      y_pred = model(x)
      acc += get_acc(y_pred, y.reshape(-1, 1))
      loss = criterion(y_pred, y.reshape(-1, 1))
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      losses.append(loss.item())
      print(f"Loss: {loss.item()}")
      print(f"Accuracy: {get_acc(y_pred, y.reshape(-1, 1))}")
      num_batches += 1
    accuracies.append(acc/num_batches)
  return losses, accuracies