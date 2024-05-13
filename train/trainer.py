import torch
import torch.utils
import torch.utils.data

from .constants import TRAIN_ITERS, TRAIN_BATCH_SIZE

def train(model: torch.nn.Sequential, dataset: torch.utils.data.Dataset):
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters())
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

  for epoch in range(TRAIN_ITERS):
    print(f"Epoch {epoch + 1}/{TRAIN_ITERS}")
    for _, data in enumerate(dataloader):
      x, y = data
      y_pred = torch.argmax(model(x), dim=1)
      loss = criterion(y_pred.float(), y.float())
      optimizer.zero_grad()
      # loss.backward()
      optimizer.step()
      print(f"Loss: {loss.item()}")