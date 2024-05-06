import pandas as pd
import os
import opendatasets as od

# dataset = "https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images"
dataset = "https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces"
od.download(dataset)