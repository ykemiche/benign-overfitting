from torch.utils.data import Dataset
from scipy.io.arff import loadarff
import pandas as pd
import torch
import numpy as np

class MyDataset(Dataset):
 
  def __init__(self,x,y):

    self.x_train=torch.tensor(x.values.astype(np.float32),dtype=torch.float32)
    self.y_train=torch.tensor(y.values.astype(np.float32),dtype=torch.float32)
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]
    