import torch
import torch.nn as nn
import torch.utils.data as dt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .constants import TRAIN_EPOCHS, BATCH_SIZE, LOG_DIR, ITERS_PER_VALIDATION

def train_model(model: nn.Module, train_data: dt.Dataset, val_data: dt.Dataset, criterion: nn.Module):
  writer = SummaryWriter(log_dir=LOG_DIR)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  dataloader = dt.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)

  model.train()
  for epoch in range(TRAIN_EPOCHS):
    print(f"Epoch {epoch + 1}/{TRAIN_EPOCHS}")
    for i, data in tqdm(list(enumerate(dataloader))):
      optimizer.zero_grad()
      x, y = data
      scores = model(x)
      y_pred = scores > 0.5
      acc = torch.count_nonzero(y_pred == y) / len(y_pred)
      loss = criterion(y_pred, y)
      loss.backward()
      optimizer.step()

      writer.add_scalar("loss", loss.item(), global_step=i)
      writer.add_scalar("train-acc", acc, global_step=i)

      if i % ITERS_PER_VALIDATION == 0:
        val_acc = test_model(model, val_data)
        writer.add_scalar("val-acc", val_acc, global_step=i)

    lr_scheduler.step()
    writer.flush()
    

def test_model(model: nn.Module, dataset: dt.Dataset):
  model.eval()
  dataloader = dt.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
  with torch.no_grad():
    acc = 0
    num_samples = 0
    for (x, y) in dataloader:
      scores = model(x)
      y_pred = scores > 0.5
      acc += torch.count_nonzero(y_pred == y) / len(y_pred)
      num_samples += len(x)
  model.train()
  return acc / num_samples
