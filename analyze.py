import numpy as np
import torch
import torch.nn as nn
from torchvision import utils
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2

from constants import SPECS
from data.constants import IMAGE_SIZE
from data.dataset import FaceImageDataset
from models.lin import lin_model
from models.cnn import cnn_model
from train.trainer import test_model

from constants import LIN_MODELS, CNN_MODELS, DATASET_SPECS

np.random.seed(0)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def plot_image(img, path):
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  if not os.path.exists(os.path.dirname(path)):
    os.makedirs(os.path.dirname(path))
  cv2.imwrite(path, img)

def squash_tensor(tensor):
  return torch.clip(tensor / 100, 0, 1)

def tensor_to_image(tensor):
  image = tensor.numpy().transpose((1, 2, 0))
  image = (image * 255).astype('uint8')
  return image

def plot_sample_images():
  fake_images = []
  real_images = []
  dataset = FaceImageDataset('train')
  while len(fake_images) < 10 or len(real_images) < 10:
    i = np.random.randint(len(dataset))
    image, label = dataset[i]
    if label == 0:
      if len(fake_images) < 10:
        fake_images.append(i)
    else:
      if len(real_images) < 10:
        real_images.append(i)
  
  for spec in DATASET_SPECS:
    name, args = spec[0], spec[1:]
    dataset = FaceImageDataset(*args)
    for idx, i in enumerate(fake_images):
      image, _ = dataset[i]
      if name == 'fft':
        image = squash_tensor(image)
      plot_image(tensor_to_image(image), f'plots/sample-images/{name}/fake/{idx}.png')

    for i in real_images:
      image, _ = dataset[i]
      if name == 'fft':
        image = squash_tensor(image)
      plot_image(tensor_to_image(image), f'plots/sample-images/{name}/real/{i}.png')

def plot_lin_weights():
  model = nn.DataParallel(lin_model())
  for name, path in LIN_MODELS:
    model.load_state_dict(torch.load(path))
    model.eval()
    W, b = model.parameters()
    W = W.detach().numpy()
    b = b.detach().numpy()
    W = W.reshape((3, IMAGE_SIZE, IMAGE_SIZE)).transpose((1, 2, 0))
    W = sigmoid(W * 20)
    W = (W * 255).astype('uint8')
    plot_image(W, f'plots/lin-weights/{name}.png')

def evaluate_models():
  df = pd.DataFrame(columns=['model', 'accuracy'])
  for source in ['local', 'tpdne', 'celeba']:
    for spec in SPECS:
      model_name, f_model, do_fft, _, mean, std = spec
      print(f"Evaluating model {model_name} on dataset {source}...")
      data = FaceImageDataset('test', source=source, do_fft=do_fft, do_augment=False, mean=mean, std=std)
      model = f_model()
      model = nn.DataParallel(model)
      f_saved_model = os.listdir(f"saved_models/{model_name}")[0] 
      model.load_state_dict(torch.load(f"saved_models/{model_name}/{f_saved_model}"))
      acc = test_model(model, data)
      print(f"Accuracy: {acc.item()}")
      df = df._append({'model': model_name, 'accuracy': acc.item(), 'source': source}, ignore_index=True)

  if not os.path.exists('plots/test-models'):
    os.makedirs('plots/test-models')
  df.to_csv('plots/test-models/model-accuracy.csv')

def plot_acc():
  for source in ['local', 'tpdne', 'celeba']:
    df = pd.read_csv('plots/test-models/model-accuracy.csv')
    df = df[(df['source'] == source)]
    df['aug'] = df['model'].str.contains('aug')
    df['fft'] = df['model'].str.contains('fft')
    df['group'] = df['model'].str.replace('-aug', '')

    models = df['group'].unique()

    _, ax = plt.subplots(layout='constrained')
    for row in df.iterrows():
      _, row = row
      x = list(models).index(row['group']) * 3 + row['aug'] + 0.5 * row['fft']
      ax.bar(x, row['accuracy'], color='g' if row['fft'] else ('r' if row['aug'] else 'b'))
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Model accuracy on {source} dataset')
    ax.set_xticks([0.5, 3.5, 6.5, 9.5, 12.5], models)
  
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='Augmented Data')
    blue_patch = mpatches.Patch(color='blue', label='Not Augmented Data')
    ax.legend(handles=[red_patch, blue_patch])

    ax.set_ylim(0, 1)
    plt.savefig(f'plots/test-models/acc-{source}.png')

def plot_cnn_filters():
  model = nn.DataParallel(cnn_model())
  for name, path in CNN_MODELS:
    model.load_state_dict(torch.load(path))
    model.eval()
    tensor = model.module[0].weight.data.clone()    
    nrow = 8
    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=1)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.title(f"{name} Shallow CNN 7x7 filters")
    if not os.path.exists('plots/cnn-filters'):
      os.makedirs('plots/cnn-filters')
    plt.savefig(f'plots/cnn-filters/{name}.png')

def plot_image_gradients():
  data = FaceImageDataset('valid', do_fft=False, do_augment=False)
  base_image = None
  label = None
  while label != 0:
    i = np.random.randint(len(data))
    base_image, label = data[i]
  for spec in SPECS:
    image = torch.clone(base_image).expand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    image.requires_grad_()
    model_name, f_model, do_fft, _, _, _ = spec
    if do_fft:
      continue
    model = f_model()
    model = nn.DataParallel(model)
    f_saved_model = os.listdir(f"saved_models/{model_name}")[0] 
    model.load_state_dict(torch.load(f"saved_models/{model_name}/{f_saved_model}"))
    model.train()
    model.zero_grad()

    out = model(image)
    grad = torch.autograd.grad(out, image, create_graph=True)[0][0]
    grad = torch.abs(grad.detach())
    plot_image(tensor_to_image(grad * 50), f'plots/image-gradients/{model_name}.png')
  plot_image(tensor_to_image(base_image), 'plots/image-gradients/base.png')

def analyze():
  plot_lin_weights()
  plot_sample_images()
  evaluate_models()
  plot_acc()
  plot_cnn_filters()
  plot_image_gradients()

if __name__ == '__main__':
  analyze()