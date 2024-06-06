import os
import torch
import torch.nn as nn
import torch.utils.data as dt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .constants import (
  TRAIN_EPOCHS,
  BATCH_SIZE,
  get_timestamp,
  get_saved_model_path,
  get_log_dir,
)

def train_model(model_name: str, model: nn.Module, train_data: dt.Dataset, val_data: dt.Dataset):
  timestamp = get_timestamp()
  model_save_path = get_saved_model_path(model_name, timestamp)
  log_dir = get_log_dir(model_name, timestamp)
  os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
  
  writer = SummaryWriter(log_dir=log_dir)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  criterion = nn.BCELoss()
  dataloader = dt.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)

  last_val_acc = 0
  val_acc_decreasing_counter = 0
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

      writer.add_scalar("loss", loss.item(), global_step=global_step)
      writer.add_scalar("acc/train", acc, global_step=global_step)

      global_step += 1

    val_acc = test_model(model, val_data)
    if val_acc < last_val_acc:
      val_acc_decreasing_counter += 1
    else:
      val_acc_decreasing_counter = 0

    writer.add_scalar("acc/val", val_acc, global_step=global_step)
    lr_scheduler.step()
    writer.flush()

    if val_acc_decreasing_counter >= 3:
      print("Stopping early due to decreasing validation accuracy")
      break

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

