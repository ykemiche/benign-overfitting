from ast import arg
import torch
import os
import time
from models.neural_network import *
import numpy as np
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler
import math
import argparse
from glob import glob
import pandas as pd
from tqdm import tqdm
from utils import *
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from Dataset import *

# os.environ["CUDA_VISIBLE_DEVICES"] ="1,2"
CUDA_LAUNCH_BLOCKING=1
parser = argparse.ArgumentParser(description='')
# 1279
# "cuda:0" if torch.cuda.is_available() else
parser.add_argument('--input_size', dest='input_size', type=int,default=11, help='')
parser.add_argument('--output_size', dest='output_size', type=int, default=1, help='')

parser.add_argument('--epochs', dest='epochs', type=int, default=500, help='# of epoch')

parser.add_argument('--start_size', dest='start_size', type=int, default=1, help='')
parser.add_argument('--max_size', dest='max_size', type=int, default=900, help='maximum incremental size')
parser.add_argument('--noise_percentage', dest='noise_percentage', type=int, default=0, help='')
parser.add_argument('--nb_trials', dest='nb_trials', type=int, default=1, help='')
parser.add_argument('--split_size', dest='split_size', type=float, default=0.2, help='')
parser.add_argument('--batch_size', dest='batch_size', type=float, default=32, help='')

# parser.add_argument('--device', dest='device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), help='device')
parser.add_argument('--device', dest='device', type=str, default=torch.cuda.set_device(0))

parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='lr')

parser.add_argument('--model_name', dest='model_name', type=str, default="nn", help='')
parser.add_argument('--optim_name', dest='optim_name', type=str, default="adam", help='')
parser.add_argument('--loss_name', dest='loss_name', type=str, default="mse", help='')

parser.add_argument('--dataset_name', dest='dataset_name', type=str, default="BNG_wine_quality", help='')

parser.add_argument('--save_dir', dest='save_dir', type=str, default="/home/infres/ext-6343/venv_boverfitting_gpu3/benign-overfitting/saved_results", help='device')

parser.add_argument('--json_name', dest='json_name', type=str, default="results.json", help='')

args = parser.parse_args()

destination_folder=args.save_dir+"/"+args.dataset_name+"_"+args.model_name+"_"+str(args.epochs)+"_"+str(args.max_size)+"_"+str(args.nb_trials)+"_"+args.loss_name+"_"+args.optim_name+"_"+str(args.lr)+"_"+str(args.noise_percentage)
destination_folder_plots=destination_folder+"/plots"

dest = os.path.exists(destination_folder)
plots=os.path.exists(destination_folder_plots)

if not dest:
  # Create a new directory because it does not exist
  os.makedirs(destination_folder)

if not plots:
  # Create a new directory because it does not exist
  os.makedirs(destination_folder_plots)

json_file=destination_folder+"/"+args.json_name

if __name__ == '__main__':
  torch.cuda.empty_cache()

  ############################ DATA ##########################################
  data = loadarff('/home/infres/ext-6343/venv_boverfitting/benign-overfitting/data/BNG_wine_quality.arff')
  raw_dataset = pd.DataFrame(data[0])
  dataset = raw_dataset.copy()
  # dataset=dataset.sample(frac=1)[0:10000]

  dataset = dataset.dropna()
  # normalization
  dataset.iloc[:,:11]=(dataset.iloc[:,:11]-dataset.iloc[:,:11].mean())/dataset.iloc[:,:11].std()

  train_features = dataset.copy()
  train_labels = train_features.pop('quality')

  X_train, X_test, y_train, y_test = train_test_split(train_features,train_labels)

  train_dataset=MyDataset(X_train,y_train)
  test_dataset=MyDataset(X_test,y_test)

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True,drop_last=True)
  validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False,drop_last=True)

  X_train_list = torch.Tensor(X_train.values.astype(np.float32))
  X_train_list=X_train_list.to(args.device)

  y_train_list = torch.Tensor(y_train.values.astype(np.float32)).view(y_train.shape)
  y_train_list=y_train_list.to(args.device)

  X_test_list = torch.Tensor(X_test.values.astype(np.float32))
  X_test_list=X_test_list.to(args.device)

  y_test_list = torch.Tensor(y_test.values.astype(np.float32)).view(y_test.shape)
  y_test_list=y_test_list.to(args.device)


  ############################ TRAINING ##########################################

  saved_values={}
  keys_errors = ["Train_Errors","Test_Errors"]
  keys_trials=[i for  i in range(args.nb_trials)]
  saved_values={key: {key: np.zeros((args.max_size-args.start_size,args.epochs)) for key in keys_errors} for key in keys_trials}

  for key in saved_values:

    saved_values[key]["Test_kernel_Errors"]=np.zeros((args.max_size-args.start_size))
    saved_values[key]["Train_kernel_Errors"]=np.zeros((args.max_size-args.start_size))



  for k in range(args.start_size,args.max_size):
    print(f" Width K : {k+1}/{args.max_size}: ")
    for trial in range(args.nb_trials):

      model=NN(args.input_size,args.output_size,k)
      model.to(args.device)

      fnet, params ,buffers =make_functional_with_buffers(model)

      # loss
      loss_fn = torch.nn.MSELoss() 
      # optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

      # optimizer
      optim = torch.optim.Adam(model.parameters(), lr=args.lr)

      # training_from_df(X_train,y_train,X_test,y_test,trial,args.epochs,model,loss_fn,optim,k-args.start_size,params,buffers,fnet,saved_values,json_file)

      train_from_loader(train_loader,validation_loader,X_train_list, X_test_list, y_train_list,y_test_list,trial,args.epochs,model,loss_fn,optim,k-args.start_size,args.device,saved_values,json_file,fnet, params ,buffers)
