import os
import torch
import torch.nn as nn
import torch.utils.data as dt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .constants import TRAIN_EPOCHS, BATCH_SIZE, SAVED_MODELS_DIR, get_timestamp, get_saved_model_path, get_log_dir

def train_model(model_name: str, model: nn.Module, train_data: dt.Dataset, val_data: dt.Dataset, criterion: nn.Module):
  timestamp = get_timestamp()
  model_save_path = get_saved_model_path(model_name, timestamp)
  log_dir = get_log_dir(model_name, timestamp)
  os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
  
  writer = SummaryWriter(log_dir=log_dir)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  dataloader = dt.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)

  global_step = 0
  model.train()
  for epoch in range(TRAIN_EPOCHS):
    print(f"Epoch {epoch + 1}/{TRAIN_EPOCHS}")
    torch.save(model.state_dict(), model_save_path)   

    for data in tqdm(dataloader):
      optimizer.zero_grad()
      x, y = data
      scores = model(x)
      y_pred = scores > 0.5
      acc = torch.count_nonzero(y_pred == y) / len(y_pred)
      loss = criterion(scores, y)
      loss.backward()
      optimizer.step()

      writer.add_scalar(f"{model_name}/loss", loss.item(), global_step=global_step)
      writer.add_scalar(f"{model_name}/train_acc", acc, global_step=global_step)

      global_step += 1

    val_acc = test_model(model, val_data)
    writer.add_scalar("val_acc", val_acc, global_step=global_step)

    lr_scheduler.step()
    writer.flush()

  torch.save(model.state_dict(), model_save_path)   
  writer.close() 

def test_model(model: nn.Module, dataset: dt.Dataset):
  model.eval()
  dataloader = dt.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
  with torch.no_grad():
    acc = 0
    num_samples = 0
    for (x, y) in tqdm(dataloader):
      scores = model(x)
      y_pred = scores > 0.5
      acc += torch.count_nonzero(y_pred == y)
      num_samples += len(y)
  model.train()
  return acc / num_samples

