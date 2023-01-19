#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 16:08:02 2022

@author: nemo
"""

# %% import library

import inspect, os, sys, platform
import numpy as np
import h5py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import pytorch_model_summary

import fnmatch
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

import scipy.io

sys.path.append('/MNT_disk5/SBLIM/Deeplearning_pipeline/')

# Deep learning structure.
import model_torch_v_0_1_1 as model_tor


import matplotlib
# matplotlib.use('Agg')   # without plot
import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from math import sqrt
sc = StandardScaler()

######
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
# from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier

# %% Code implement message
print("*----------------------------------------------------------------*")
print("This script was created by NEMOLAB, Seokbeen Lim")
print("................")
print(".............")
print("..........")
print("........")
print("......")
print("...")
print(".")
print("...")
print("......")
print("........")
print("..........")
print(".............")
print("................")
print("Pytorch library based deep learning code for study")
print("*----------------------------------------------------------------*")

# %% figure size
plt.rcParams["figure.figsize"] = [9,6]
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 12

# %% deeplearning environment check.
print("*----------------------------------------------------------------*")

''' 2. 딥러닝 모델을 설계할 때 활용하는 장비 확인 '''
# cuda, cpu check.
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)


# platform information
if platform.machine() == 'x86_64':
    SERVER_NUM = 0    # 0: Medical Building,    1: Jang-Heung
    print("This script currently implement Intel core MCU based server: medical building ")
elif platform.machine() == 'ppc64le':
    SERVER_NUM = 1    # 0: Medical Building,    1: Jang-Heung
    print("This script currently implement IBM core base [ppc64le] based server: medical building")


# model_torch or model_tensorflow version check.
model_torch_version = model_tor.__version__()
print("Scirpt purpose: deeplearning example script - MNIST ROTATE Regression script")
print("Model structure information")
print("model_torch.py Version: "+model_tor.__version__())

print("*----------------------------------------------------------------*")
# %% Parameter set up
print("*------------------------ Parameter set up --------------------------*")

#--------------------------------------------------------#
   ############## pytorch parameter ################
#--------------------------------------------------------#

# GPU number selection
os.environ["CUDA_VISIBLE_DEVICES"]="2"

# Training parameter
BATCH_SIZE = 64
EPOCHS = 1500
LEARNING_RATE = 0.0001

# Model name
# Please read README file in the procpath, there are other model name or structure information
model_name = "CNN_simple_Serial_mod2"

# Trainig condition
# for regression

# input image processing
inputimage_layer = 1  # 0: No inputimage processing
                      # 1: 'Zero-center' normalization: mean distraction normalization.

dataresize = 0   # 1: data resize process activation
                 # 0: no resize
dataresize_type = 1  # 1: data resize Type 1:
                     # 2: data resize Type 2: 
                     # 3: data resize Type 3:

if ((dataresize == 1) and (dataresize_type == 1)):
    img_size_square_resize = 224
elif ((dataresize == 1) and (dataresize_type == 2)):
    img_size_square_resize = 256
elif ((dataresize == 1) and (dataresize_type == 3)):
    img_size_square_resize = 128
else:
    print("[Pre-process info] No data resize processing")
                         

best_model_weight_call_option = 0  # 0: No call weight
                                   # 1: Yes call best model weights (at classification model)
# Random gaussian distribution index
Gaussian_ditrib_count = 10

# K-fold cross validation parameter
n_folds_outer = 5  # K-fold number.

# if we was loaded pretrained weight for model regression, choose two types model weight update option when model training.
freeze_model_weight = 0 # 0: DOn't freeze weight for grad update, 1: Freeze weight for grad update

# Optimizer choice 
Optimizer_index = 2    # 1: sgdm
                       # 2: Adam
                       # 3: RMSprop

# Criterion choice
Criterion_index = 1    # 1: MSELoss

# Loss regulization index 
regulization_use = 0   # 1: Regulization used.
                       # 0: No Regulization
regulization_index = 3   # 1: l1 regulization
                         # 2: l2 regulization
                         # 3: l1 + l2 regulization (Elastic net)
l1_weight_ind = 0.3
l2_weight_ind = 0.7                          

# model result or figure save option
fnprefix = 'Pytorch_' + model_name + '_' + 'B' + str(BATCH_SIZE) + '_' + 'Epoch' + str(EPOCHS) + '_' + 'Lr' + str(LEARNING_RATE) +\
    '_Opt_' + 'InL_' + str(inputimage_layer) + '_dr_' + str(dataresize) + '_drt_' + str(dataresize_type) + '_bmwcall_'+str(best_model_weight_call_option) +\
        '_Gd_' + str(Gaussian_ditrib_count)+ '_Opti_'+ str(Optimizer_index)+ '_Crti_'+ str(Criterion_index) + '_Regul_' + str(regulization_use) + '_Kf_' + str(n_folds_outer)

# model regression out dimension
output_dim = 1

# For regression, lastlayer should be linear layer
# this index is able to set linear layer of model last layer
Regression_lastlayer_change_call_option = 1     # 0: NO change lastlayer
                                                # 1: Yes change lastlayer for regression

#--------------------------------------------------------#
#--------------------------------------------------------#


##########################################################


#---------------------------------------------------------------#
   ############## data load path or parameter ################
#---------------------------------------------------------------#

# Data Path
datapath = '/MNT_disk5/SBLIM/Deeplearning_pipeline/Data/Example/'
codepath = '/MNT_disk5/SBLIM/Deeplearning_pipeline/'
procpath = '/MNT_disk5/SBLIM/Deeplearning_pipeline/Data/Processpath_Example'

# Best model call option check and implementing best model weights
if best_model_weight_call_option == 1:
    Best_model_path = '/MNT_disk5/SBLIM/Deeplearning_pipeline/Data/Processpath_Example/'
    Best_model_folder = ''
else:
    print('Does not call best weight path, foler name')
    


# Data name
dataname = 'MNIST_Rotate.mat'
test_dataset_name = 'MNIST_Rotate_test.mat'

if best_model_weight_call_option == 1:
    Best_model_name = 'Best_model_pytorch.pt'
    Best_model_structure = Best_model_path + Best_model_folder + Best_model_name
else:
    print('Does not call best weight model name, total path and name')
    

#---------------------------------------------------------------#
#---------------------------------------------------------------#

print("*----------------------------------------------------------------*")

# %% Defined data load, model trainig, and evaluate function 

print("*------------------------ Defined training function --------------------------*")

''' Pytorch dataset custom data set.. '''
from torch.utils.data import Dataset

# Dataset Inheritance
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.X_Data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.X_Data)
    
    def __getitem__(self, idx):
        X_Data_tensor = torch.FloatTensor(self.X_Data[idx])
        # Y_Data = torch.LongTensor(self.labels[idx])        # Label doesnot dtype to float..
        Y_Data = torch.FloatTensor(self.labels[idx])        # Label does dtype to float if regressor model.. (FOr MSELoss)
        Y_Data = torch.squeeze(Y_Data)
        return X_Data_tensor, Y_Data

''' 8. CNN 모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        # print(output)
        # print(label)
        # print(output.shape)
        # print(label.shape)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                epoch, batch_idx * len(image), 
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                loss.item()))

def train_regulization(model, train_loader, optimizer, log_interval, regulization_index, l1_weight_index, l2_weight_index):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        # print(output)
        # print(label)
        # print(output.shape)
        # print(label.shape)
        loss = criterion(output, label)
        
        # Specify L1 and L2 weight regulization (Elastic net)
        # Specify L1 and L2 weights
        l1_weight = l1_weight_index
        l2_weight = l2_weight_index
        
        # Compute L1 and L2 loss component
        parameters = []
        for parameter in model.parameters():
            parameters.append(parameter.view(-1))
        l1 = l1_weight * model.compute_l1_loss(torch.cat(parameters))
        l2 = l2_weight * model.compute_l2_loss(torch.cat(parameters))
        
        # Add L1 and L2 loss components
        if regulization_index == 1:
            loss += l1
        elif regulization_index == 2:
            loss += l2
        elif regulization_index == 3:
            loss += l1
            loss += l2
        
        # Perform backward pass       
        loss.backward()
        
        # Perform optimization
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                epoch, batch_idx * len(image), 
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                loss.item()))



''' 9. 학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            # prediction = output.max(1, keepdim = True)[1]
            # correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss /= (len(test_loader.dataset) / BATCH_SIZE)
    # doesn't need in classification model.
    # test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss

print("*----------------------------------------------------------------*")

#%% Other function here
print("*------------------------ Defined other function --------------------------*")

def plot_histogram(Data, title_string, xlabel_string, ylabel_string, save_info):
    
    plt.title(title_string)
    plt.hist(Data, edgecolor='black')
    plt.xlabel(xlabel_string)
    plt.ylabel(ylabel_string)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    fig2 = plt.gcf()
    plt.show()
    plt.draw()
    fig2.savefig(save_info)
    plt.gcf().clear()
    
    success_ind = 1
    
    return success_ind

def plot_image_check(Data, label, title_string, save_info):
    
    numImage = Data.shape[0]
    idx = np.random.permutation(numImage)
    idx = idx[:19]
    
    plt.title(title_string)
    plt.rc('xtick', labelsize=3)
    plt.rc('ytick', labelsize=3)
    for i in range(idx.shape[0]):
        plt.subplot(4, 5, i+1)
        plt.imshow(np.squeeze(Data[idx[i], :, :, :]))
        plt.rc('xtick', labelsize=3)
        plt.rc('ytick', labelsize=3)
        plt.title("Label: " + str(label[idx[i]]), fontsize=10)
        
    
    fig2 = plt.gcf()
    plt.show()
    plt.draw()
    fig2.savefig(save_info)
    plt.gcf().clear()
    
    success_ind = 1
    
    return success_ind


print("*----------------------------------------------------------------*")

#%% Data load and augmentation

print("*------------------------ Data load and augmentation --------------------------*")

#---------------------------------------------------------------#
         ############## data load ################
#---------------------------------------------------------------#


os.chdir(datapath)
mat_contents = h5py.File(dataname,'r') # .mat should be saved -v7.3 setting.
mat_contents_list = list(mat_contents.keys())

print('data set .mat contentes below:  ')
print(mat_contents_list)

X_data_for_mat = 'X_data'
Y_data_for_mat = 'Y_data'

# Raw-data for trainig data set.
X_Data = np.transpose(mat_contents[''.join(fnmatch.filter(mat_contents_list, X_data_for_mat))])
Y_Data = np.transpose(mat_contents[''.join(fnmatch.filter(mat_contents_list, Y_data_for_mat))])


# save variable for pre-processed data set
X_Data_stack = X_Data
Y_Data_stack = Y_Data
Y_Data_backup = Y_Data_stack

print('\t [info] Data load process finish....!')
#---------------------------------------------------------------#
#---------------------------------------------------------------#


##########################################################


#---------------------------------------------------------------#
         ############## data information ################
#---------------------------------------------------------------#

print('\t [info] Data information calculated.....!')
print('\t ..... (1) Data min-max information .....')
Max_value_X_Data_stack = np.max(X_Data_stack)
Min_value_X_Data_stack = np.min(X_Data_stack)

print('\tX_Data_stack Max value :', Max_value_X_Data_stack)
print('\tX_Data_stack Min value :', Min_value_X_Data_stack)


if inputimage_layer == 1:
    # Data-> Mean subtraction.
    
    X_Data_mean = np.mean(X_Data_stack, axis=0)
    X_Data_stack = X_Data_stack - X_Data_mean
    
    X_Data_stack_norm_mean_backup = X_Data_stack #norm_mean backup
    
    Max_value_X_Data_stack = np.max(X_Data_stack)
    Min_value_X_Data_stack = np.min(X_Data_stack)
    print('\t[Inputlayer norm] Zero center norm.... X_Data_stack Max value :', Max_value_X_Data_stack)
    print('\t[Inputlayer norm] Zero center norm.... X_Data_stack Min value :', Min_value_X_Data_stack)
    
    num_var_Y_stack = np.unique(Y_Data_stack)
    
    max_value_Y_Data_stack = np.max(num_var_Y_stack)
    min_value_Y_Data_stack = np.min(num_var_Y_stack)

    print('\t Y_Data_stack Max value :', max_value_Y_Data_stack)
    print('\t Y_Data_stack Min value :', min_value_Y_Data_stack)
    
    
    Y_Data_standarzation = (Y_Data_stack - min_value_Y_Data_stack)/(max_value_Y_Data_stack-min_value_Y_Data_stack)
    
    Y_Data_standarzation = Y_Data_standarzation * 2 - 1  # -1 ~ 1 scaling
    
    max_value_Y_Data_standarzation = np.max(Y_Data_standarzation)
    min_value_Y_Data_standarzation = np.min(Y_Data_standarzation)
    
    print('\t[Inputlayer norm] Label standarzation norm.... Y_Data_standarzation Max value :', max_value_Y_Data_standarzation)
    print('\t[Inputlayer norm] Label standarzation norm.... Y_Data_standarzation Min value :', min_value_Y_Data_standarzation)
    
    
    Y_Data_stack = Y_Data_standarzation

else:
    print('\t[Inputlayer norm] No processing for input image.....!')

#---------------------------------------------------------------#
#---------------------------------------------------------------#


##########################################################


#------------------------------------------------------------------------#
     ############## image resize for training model ################
#------------------------------------------------------------------------#

if dataresize == 1:
    
        print('\tData resize process start...!')
        X_Data_resize = np.zeros((0, img_size_square_resize, img_size_square_resize))
        for loop_resize in range(len(X_Data_stack)):
            Temp_image = X_Data_stack[loop_resize, :, :]
            Temp_image_resize = cv2.resize(Temp_image, (img_size_square_resize, img_size_square_resize))
            Temp_image_resize = np.reshape(Temp_image_resize, (1, img_size_square_resize, img_size_square_resize))
            X_Data_resize = np.concatenate((X_Data_resize, Temp_image_resize))
                
        print('\tData resize process end...!')
            
        X_Data_expand_dim = np.expand_dims(X_Data_resize, axis=1)   # if gray scale in image data set, it should expand dimension for deep learning process.
        
        X_Data_stack = X_Data_expand_dim
        Y_Data_stack = Y_Data_stack
else:
        X_Data_expand_dim = np.expand_dims(X_Data_stack, axis=1)   # if gray scale in image data set, it should expand dimension for deep learning process.
        
        X_Data_stack = X_Data_expand_dim
        Y_Data_stack = Y_Data_stack    
#------------------------------------------------------------------------#
#------------------------------------------------------------------------#

##########################################################

#--------------------------------------------------------------------------------#
  ############## Gaussian noise inserted in pre-processed data ################
#--------------------------------------------------------------------------------#

print('\t [info] Gaussian noise inserted process start.....!')

import random

mean_noise_distrib = np.array([0])
std_noise_distrib = np.array([0.01])

X_Data_noise_resize = X_Data_stack
Y_Data_stack_resize = Y_Data_stack
Y_Data_backup_resize = Y_Data_backup

X_Data_noise_stack = np.zeros((0, X_Data_stack.shape[1], X_Data_stack.shape[2],X_Data_stack.shape[3]))

for idx in range(Gaussian_ditrib_count):
    for jdx in range(X_Data_stack.shape[0]):
        Random_noise_temp = np.random.normal(mean_noise_distrib[0], std_noise_distrib[0], size=(1,  X_Data_stack.shape[1], X_Data_stack.shape[2],X_Data_stack.shape[3]))
        X_Data_stack_plus_noise = X_Data_stack[jdx,:,:] + Random_noise_temp
        
        X_Data_noise_stack = np.concatenate((X_Data_noise_stack, X_Data_stack_plus_noise))
    
    X_Data_noise_resize = np.concatenate((X_Data_noise_resize, X_Data_noise_stack))
    Y_Data_stack_resize = np.concatenate((Y_Data_stack_resize, Y_Data_stack))
    Y_Data_backup_resize = np.concatenate((Y_Data_backup_resize, Y_Data_backup))
    
    X_Data_noise_stack = np.zeros((0, X_Data_stack.shape[1], X_Data_stack.shape[2],X_Data_stack.shape[3]))   # initialization.

X_Data_stack = X_Data_noise_resize
Y_Data_stack = Y_Data_stack_resize
Y_Data_backup = Y_Data_backup_resize


print('\t [info] Gaussian noise inserted process end.....!')

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#

##########################################################


print("*----------------------------------------------------------------*")


# %% model train script.

print("*------------------------ Model train --------------------------*")

cross_fold = StratifiedKFold(n_splits=n_folds_outer, random_state=1, shuffle=True)  # TRain, Val n_folds set..
cross_fold_count = 0

for train_index, test_index in cross_fold.split(X_Data_stack, Y_Data_backup):   # Y Data should be (length, 1)

#---------------------------------------------------------------------------------#
  ############## Data split for train/validation set each K-fold ################
#---------------------------------------------------------------------------------#
    
    print('\t Cross fold:  %d, started...!' %(cross_fold_count+1))
    X_remain, X_Test = X_Data_stack[train_index], X_Data_stack[test_index]
    Y_remain, Y_Test = Y_Data_stack[train_index,:], Y_Data_stack[test_index,:]
    
    print('\tX_remain.shape :', X_remain.shape)
    print('\tY_remain shape :', Y_remain.shape)
    print('\tX_Test.shape :', X_Test.shape)
    print('\tY_Test shape :', Y_Test.shape)
    
    X_Remain = X_remain
    
    # Dataload function for training model using pytorch library
    
    train_loader = torch.utils.data.DataLoader(dataset = CustomDataset(X_Remain, Y_remain),
                                            batch_size = BATCH_SIZE,
                                            shuffle = True)
    
    test_loader = torch.utils.data.DataLoader(dataset = CustomDataset(X_Test, Y_Test),
                                              batch_size = BATCH_SIZE,
                                              shuffle = False)
    
    model_test_loader = torch.utils.data.DataLoader(dataset = CustomDataset(X_Test, Y_Test),
                                              batch_size = BATCH_SIZE,
                                              shuffle = False)
    
    
    
#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#

##########################################################    
    
#---------------------------------------------------------------------------------#
     ############## Training Information data save part ################
#---------------------------------------------------------------------------------#    
    
    if not os.path.exists(procpath):
        os.makedirs(procpath)
    os.chdir(procpath)
        
    procpath2 = str(datetime.now().year)+'_'+str(datetime.now().month)+'_'+str(datetime.now().day)+\
        '_'+fnprefix+'_'+dataname[0:-4]+"_Reg_"+"Fold"+str(cross_fold_count+1)
    
        
    if SERVER_NUM == 1:    # ppc64le, Jang-Heung
        procpath2 = procpath2+'_titan' 
    
    
    if not os.path.exists(procpath2):
            os.makedirs(procpath2)
    os.chdir(procpath2)
    
    procpath2 = os.getcwd()
    print('\tProcessing path: ', procpath2)
    
    os.chdir(procpath)
       
    currentcode = inspect.getfile(inspect.currentframe())
    print('\n [Info] Running code: ', currentcode)
            
    ########################################################    
    os.chdir(procpath2)  # Change directory


    # Model structure save...
    if SERVER_NUM == 0:
        import shutil
        shutil.copyfile(currentcode, '%s/%s' % (procpath2,currentcode.split('/')[-1]))
        # plot_model(model, show_shapes= 'True', to_file= modelname+'.png')
        
    # with open(modelname+'.txt', 'w') as f2:
        # model.summary(print_fn=lambda x: f2.write(x+'\n'))
    
    # plot data.
    
    X_Data_Preprocessing = plot_image_check(X_Data_stack, Y_Data_stack, "X_Data_Preprocessing", "X_Data_Preprocessing_subplot.png")
    X_Data_K_fold = plot_image_check(X_Remain, Y_remain, "X_Data_K_fold_"  + str(cross_fold_count), "X_Data_K_fold_" + str(cross_fold_count)+"_hist.png")
    
    Before_K_fold_hist_ind = plot_histogram(Y_Data_stack, "Before_K_fold", "Rotation_angle", "Counts", "Before_K_fold_hist.png")
    K_fold_hist_ind = plot_histogram(Y_remain, "K_fold_"  + str(cross_fold_count) , "Rotation_angle", "Counts", "K_fold_" + str(cross_fold_count)+"_hist.png")

#%%

#---------------------------------------------------------------------------------#
     ############## Model, optimizer, objective function set ################
#---------------------------------------------------------------------------------#   

    ''' 7. Optimizer, Objective Function 설정하기 '''
    model = model_tor.get_model(model_name, X_Remain.shape, X_Remain.shape[1], output_dim)
    model.cuda()
    
    if best_model_weight_call_option == 1:
        model = torch.load(Best_model_structure)
    else:
        print('\t [info]  Does not call torch best model weights...!')
    
    # for regression
    # model.feature[-1] = F.relu()
    
    # Freeze model weights
    if freeze_model_weight == 1:
        for param in model.parameters():
            param.requires_grad = False
                
        # If it want to fc1 layer
        model.fc1.weight.requires_grad = True  
    
    if Regression_lastlayer_change_call_option == 1:
        model.output_activation = nn.Linear(output_dim, output_dim)
    else:
        print('\t [info]  Does not change last layer for regression...!')
    
    if Optimizer_index == 1:
        optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum=0.9)
    elif Optimizer_index == 2:
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    elif Optimizer_index == 3:
        optimizer = torch.optim.RMSprop(model.parameters(), lr = LEARNING_RATE, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)    
    
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.MultiLabelSoftMarginLoss()
    if Criterion_index == 1:
        criterion = nn.MSELoss()
    
    print(model)
    
    with open('Model_structure_Reg_output_dim'+str(output_dim)+'_lr_' \
              +str(LEARNING_RATE)+'_bs_'+str(BATCH_SIZE)+'_opt_'+'ADAM' + '_Fold_' + str(cross_fold_count+1) + '.txt', 'w') as f:
        print(model, file=f)
        
    with open('Model_state_dict.txt', 'w') as f:
        # Model state_dict printout
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size(), file=f)
    
    
    
    with open('Model_summary_print_model_summarylib.txt', 'w') as f:
        # Model summary printout
        model.cpu() 
        print(pytorch_model_summary.summary(model, torch.zeros(1, X_Remain.shape[1], X_Remain.shape[2], X_Remain.shape[3]), show_input=True), file=f)
        
    model.cuda()

# pytorch cuda error device-side assert triggered error -> y label should start 0 ~ 2

#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#

##########################################################    
#%%  

#---------------------------------------------------------------------------------#
             ############## Model training part ################
#---------------------------------------------------------------------------------#   

    train_loss_best = 1E6    # temp variable.
    
    train_losses = []
    test_losses = []
    train_accuracys = []
    test_accuracys = []
    root_train_losses = []
    root_test_losses = []
    
    ''' 10. CNN 학습 실행하며 Train, Test set의 Loss 및 Test set Accuracy 확인하기 '''
    for epoch in range(1, EPOCHS + 1):
        if regulization_use == 1:
            train_regulization(model, train_loader, optimizer, log_interval = 100, regulization_index=regulization_index
                               ,l1_weight_index=l1_weight_ind, l2_weight_index=l2_weight_ind)
        else:
            train(model, train_loader, optimizer, log_interval = 100)
        train_loss = evaluate(model, train_loader)
        test_loss = evaluate(model, test_loader)
        
        root_train_loss = sqrt(train_loss)
        root_test_loss = sqrt(test_loss)
        print("\n[EPOCH: {}], \tTrain Loss: {:.4f} \n".format(
            epoch, train_loss))
        print("[EPOCH: {}], \tTest Loss: {:.4f} \n".format(
            epoch, test_loss))
        print("\n[EPOCH: {}], \tRoot train Loss: {:.4f} \n".format(
            epoch, root_train_loss))
        print("[EPOCH: {}], \tRoot test Loss: {:.4f} \n".format(
            epoch, root_test_loss))
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        root_train_losses.append(root_train_loss)
        root_test_losses.append(root_test_loss)
        # train_accuracys.append(train_accuracy)
        # test_accuracys.append(test_accuracy)
         
        if train_loss <= train_loss_best:
            torch.save(model, 'Best_model_pytorch.pt')
            torch.save(model, 'Best_model_pytorch_parameters.pt')
            train_loss_best = train_loss

#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#

##########################################################    
#%% Plot result.

#---------------------------------------------------------------------------------#
             ############## Plot result model training result ################
#---------------------------------------------------------------------------------#   


    plt.title("Train and validation Loss")
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend()
    fig2 = plt.gcf()
    plt.show()
    plt.draw()
    fig2.savefig('Loss curve.png')   # for regression.
    plt.gcf().clear()
    
    plt.title("Root train and validation Loss")
    plt.plot(root_train_losses, label="train")
    plt.plot(root_test_losses, label="test")
    plt.xlabel("epoch")
    plt.ylabel("Root[loss]")
    plt.legend()
    fig2 = plt.gcf()
    plt.show()
    plt.draw()
    fig2.savefig('Root loss curve.png')   # for regression.
    plt.gcf().clear()
    
    
    
    # plt.title("Train and validation Accuracy")
    # plt.plot(train_accuracys, label="train")
    # plt.plot(test_accuracys, label="validation")
    # plt.xlabel("epoch")
    # plt.ylabel("Accuracy [%]")
    # plt.legend()
    # fig2 = plt.gcf()
    # plt.show()
    # plt.draw()
    # fig2.savefig('Accuracy curve.png')
    # plt.gcf().clear()

#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#

##########################################################   

#%% model information save

#---------------------------------------------------------------------------------#
         ############## Model information save part ################
#---------------------------------------------------------------------------------#   


    scipy.io.savemat('Model_accuracy_fortrainset_and_valdataset_'+dataname,dict(train_accuracys=train_accuracys
                                             , test_accuracys=test_accuracys
                                             , train_losses=train_losses
                                             , test_losses=test_losses))    
    
#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#

##########################################################   


#%% model evaluation part during training model



#%%
    print('\t Cross fold:  %d, End...!' %(cross_fold_count+1))
    cross_fold_count = cross_fold_count + 1
    os.chdir(datapath)
    
    
    
    
#%% model evaluation part after finishing model train



