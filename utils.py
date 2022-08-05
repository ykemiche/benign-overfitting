
import torch
from tqdm import tqdm
import json
import numpy as np
from ntk.ntk import *
from functorch import make_functional,make_functional_with_buffers, vmap, vjp, jvp, jacrev


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def training_from_df(X_train,y_train,X_test,y_test,trial,epochs,model,loss_func,optimizer,k,params,buffers,fnet,saved_values,json_path):

  scores = []

  with open(json_path, 'w') as f:
    # for epoch in tqdm(range(epochs)):
    for epoch in range(epochs):

        s_predicted = model.forward(X_train)
        loss = loss_func(s_predicted.reshape(y_train.shape[0]), y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        saved_values[trial]["Train_Errors"][k-1,epoch]=loss.item()

        #Validation
        with torch.no_grad():
          s_val_predicted = model.forward(X_test)
          val_loss = loss_func(s_val_predicted.reshape(y_test.shape[0]), y_test)
          saved_values[trial]["Test_Errors"][k-1,epoch]=val_loss.item()
    #  Test     
    try:

      test_kernel_predicted=compute_kxkny(X_train,X_test,y_train,params,buffers,fnet)
      test_kernel_loss=loss_func(test_kernel_predicted.reshape(y_test.shape[0]), y_test)
      saved_values[trial]["Test_kernel_Errors"][k-1]=test_kernel_loss.item()
    
    except :
      saved_values[trial]["Test_kernel_Errors"][k-1]=1

    # Train
    try:
      train_kernel_predicted=compute_kxkny(X_train,X_train,y_train,params,buffers,fnet)
      train_kernel_loss=loss_func(train_kernel_predicted.reshape(y_train.shape[0]), y_train)

      saved_values[trial]["Train_kernel_Errors"][k-1]=train_kernel_loss.item()
    
    except:
      saved_values[trial]["Train_kernel_Errors"][k-1]=1

    json.dump(saved_values, f, indent=4,cls=NumpyEncoder)


def train_from_loader(train_loader,test_loader,X_train_list, X_test_list, y_train_list, y_test_list,trial,epochs,model,loss_fn,optim,k,device,saved_values,json_path,fnet, params ,buffers):
  # model.train()

  for epoch in tqdm(range(epochs)):
    train_loss = 0

    # Train data with nn
    for i,(X_train,Y_train) in enumerate(train_loader):

      X_train,Y_train = X_train.to(device),Y_train.to(device)
      output = model.forward(X_train)

      loss = loss_fn(torch.squeeze(output),Y_train)

      optim.zero_grad()
      loss.backward()
      optim.step()
      train_loss += loss.data

    saved_values[trial]["Train_Errors"][k-1,epoch]=train_loss/len(train_loader)

    # Validation data with nn
    loss_test=0
    model.eval()
    with torch.no_grad():
      for j,(X_test,Y_test) in enumerate(test_loader):
          X_test, Y_test = X_test.to(device), Y_test.to(device)

          output = model.forward(X_test)
          loss = loss_fn(torch.squeeze(output) ,Y_test)
          loss_test += loss.data
    saved_values[trial]["Test_Errors"][k-1,epoch]=loss_test/len(test_loader)

  torch.cuda.empty_cache()
  
  # ntk test
  try:
    test_kernel_predicted=compute_kxkny(X_train_list,X_test_list,y_train_list,params,buffers,fnet)
    test_kernel_loss=loss_fn(test_kernel_predicted.reshape(y_test_list.shape[0]), y_test_list)
    saved_values[trial]["Test_kernel_Errors"][k-1]=test_kernel_loss.item()

  except Exception as e:
    saved_values[trial]["Test_kernel_Errors"][k-1]=1
    # print(saved_values[trial]["Test_kernel_Errors"][k-1])

  # ntk train
  try:
    train_kernel_predicted=compute_kxkny(X_train_list,X_train_list,y_train_list,params,buffers,fnet)
    train_kernel_loss=loss_fn(train_kernel_predicted.reshape(y_train_list.shape[0]), y_train_list)
    saved_values[trial]["Train_kernel_Errors"][k-1]=train_kernel_loss.item()

  except:
    saved_values[trial]["Train_kernel_Errors"][k-1]=1

  with open(json_path, 'w') as f:
    json.dump(saved_values, f, indent=4,cls=NumpyEncoder)
    f.close()
