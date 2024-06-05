import numpy as np
import matplotlib.pyplot as plt

# import torch
from data.preprocessing import CustomDataset
# from torch.utils.data import DataLoader
from torchvision import transforms
from models.cnn import cnn_model
from train.trainer import train_cnn, test_cnn, train_svm, test_svm
from sklearn.linear_model import SGDClassifier

def main():
  # Define the path to your data csv 
  data_folder = './140k-real-and-fake-faces'
  train_dataset = 'dev_train.csv'
  val_dataset = 'dev_valid.csv'
  test_dataset = 'dev_test.csv'

  # Define the transformations to apply to the images
  transform = transforms.Compose([
      transforms.Resize((256, 256)),  # Resize the image to 256x256
      transforms.ToTensor(),  # Convert PIL image to tensor
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image pixels
  ])

  # Create an instance of your custom dataset
  train_data = CustomDataset(data_folder, train_dataset, transform=transform)
  val_data = CustomDataset(data_folder, val_dataset, transform=transform)
  test_data = CustomDataset(data_folder, test_dataset, transform=transform)

  # Create a DataLoader to load the data in batches
  # batch_size = 128
  # data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

  # Iterate over the data loader to access batches of preprocessed images
  # for batch_idx, batch in enumerate(data_loader):
    # Process each batch as needed
    # batch is a tensor of shape (batch_size, channels, height, width)
    # For example, you can perform some operations on the batch like printing its shape
    # print(f"Batch {batch_idx + 1} shape:", batch[0].shape, len(batch[1]))

    # # Convert the batch tensor to a NumPy array
    # batch_np = batch[0].numpy()

    # # Iterate over images in the batch
    # for i in range(len(batch_np)):
    #   image_np = np.transpose(batch_np[i], (1, 2, 0))  # Change from (C, H, W) to (H, W, C)
    #   # Display the image using matplotlib
    #   plt.imshow(image_np)
    #   plt.title(f"Batch {batch_idx + 1}, Image {i + 1}")
    #   plt.axis('off')  # Turn off axis labels
    #   plt.show()
    #   print('label: ', batch[1][i])
    
    # Optionally, break the loop after processing a few batches
    # if batch_idx == 4:
      # break

  print('Starting baseline')
  svm =  SGDClassifier(max_iter=1000, tol=1e-3)
  b_losses, bt_accs, bv_accs = train_svm(svm, train_data, val_data)
  print('Starting training')
  channel_dims = [3, 256, 128, 64, 32, 1]
  model = cnn_model(channel_dims, 2)
  losses, t_accs, v_accs = train_cnn(model, train_data, val_data)
  
  plt.title("SVM Training loss")
  plt.plot(b_losses)
  plt.xlabel("Iteration")
  plt.grid(linestyle='--', linewidth=0.5)
  plt.show()

  plt.title("CNN Training loss")
  plt.plot(losses)
  plt.xlabel("Iteration")
  plt.grid(linestyle='--', linewidth=0.5)
  plt.show()

  plt.title("Training accuracy")
  plt.plot(np.arange(1, len(bt_accs)+1), bt_accs, label='SVM train')
  plt.plot(np.arange(1, len(bv_accs)+1), bv_accs, label='SVM val')
  plt.plot(np.arange(1, len(t_accs)+1), t_accs, label='CNN train')
  plt.plot(np.arange(1, len(v_accs)+1), v_accs, label='CNN val')
  plt.xlabel("Epoch")
  plt.grid(linestyle='--', linewidth=0.5)
  plt.legend(loc='best')
  plt.show()

  print(f'Baseline SVM test accuracy: {test_svm(svm, test_data)}')
  print(f'Shallow CNN accuracy: {test_cnn(model, test_data)}')

if __name__ == '__main__':
  main()