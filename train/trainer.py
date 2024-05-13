import torch
import torch.utils
import torch.utils.data

from .constants import TRAIN_ITERS, TRAIN_BATCH_SIZE

def train(model: torch.nn.Sequential, dataset: torch.utils.data.Dataset):
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

  for epoch in range(TRAIN_ITERS):
    print(f"Epoch {epoch + 1}/{TRAIN_ITERS}")
    for _, data in enumerate(dataloader):
      x, y = data
      y_pred = model(x)
      print(y_pred)
      print(y)
      loss = criterion(y_pred, y.reshape(-1, 1))
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      print(f"Loss: {loss.item()}")