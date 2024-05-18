import torch
import torch.utils
import torch.utils.data
import numpy as np
from sklearn import metrics
from sklearn.linear_model import SGDClassifier

from .constants import TRAIN_ITERS, TRAIN_BATCH_SIZE

def get_acc(y_pred, y):
  acc = 0
  y_pred = np.array([0 if pred < 0.5 else 1 for pred in y_pred])
  for i in range(len(y_pred)):
    if (y[i] == 0 and y_pred[i] == 0) or (y[i] == 1 and y_pred[i] == 1):
      acc += 1
  return acc

def train_cnn(model: torch.nn.Sequential, dataset: torch.utils.data.Dataset):
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
  losses = []
  accuracies = []

  for epoch in range(TRAIN_ITERS):
    print(f"Epoch {epoch + 1}/{TRAIN_ITERS}")
    acc = 0
    num_samples = 0
    for _, data in enumerate(dataloader):
      x, y = data
      y_pred = model(x.to(torch.float32).requires_grad_(True))
      acc += get_acc(y_pred, y.to(torch.float32).requires_grad_(True).reshape(-1, 1))
      loss = criterion(y_pred, y.to(torch.float32).requires_grad_(True).reshape(-1, 1))
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      losses.append(loss.item())
      print(f"Loss: {loss.item()}")
      print(f"Accuracy: {get_acc(y_pred, y.reshape(-1, 1))/len(y_pred)}")
      num_samples += len(y_pred)
    accuracies.append(acc/num_samples)
  print(f"Final train accuracy: {accuracies[-1]}")
  return losses, accuracies

def test_cnn(model: torch.nn.Sequential, dataset: torch.utils.data.Dataset):
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
  acc = 0
  num_samples = 0
  for _, data in enumerate(dataloader):
    x, y = data
    y_pred = model(x.to(torch.float32).requires_grad_(True))
    acc += get_acc(y_pred, y.to(torch.float32).requires_grad_(True).reshape(-1, 1))
    num_samples += len(y_pred)
  return acc/num_samples

def train_svm(dataset: torch.utils.data.Dataset):
  svm =  SGDClassifier(max_iter=1000, tol=1e-3)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
  losses = []
  accuracies = []

  for epoch in range(TRAIN_ITERS):
    print(f"Epoch {epoch + 1}/{TRAIN_ITERS}")
    acc = 0
    num_samples = 0
    for _, data in enumerate(dataloader):
      x, y = data
      x = x.detach().numpy().reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
      y = y.detach().numpy()
      svm.partial_fit(x, y, np.unique(y))
      y_pred = svm.predict(x)
      acc += metrics.accuracy_score(y, y_pred)*len(y_pred)
      losses.append(metrics.hinge_loss(y, y_pred))
      print(f"Loss: {metrics.hinge_loss(y, y_pred)}")
      print(f"Accuracy: {metrics.accuracy_score(y, y_pred)}")
      num_samples += len(y_pred)
    accuracies.append(acc/num_samples)
  print(f"Final train accuracy: {accuracies[-1]}")
  return losses, accuracies

def test_svm(dataset: torch.utils.data.Dataset):
  svm = SGDClassifier(max_iter=1000, tol=1e-3)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
  num_samples = 0
  for _, data in enumerate(dataloader):
    x, y = data
    x = x.detach().numpy().reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
    y = y.detach().numpy()
    svm.partial_fit(x, y, np.unique(y))
    y_pred = svm.predict(x)
    acc += metrics.accuracy_score(y, y_pred)*len(y_pred)
    num_samples += len(y_pred)
  return acc/num_samples