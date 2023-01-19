#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:32:37 2022

@author: nemo
"""

import inspect, os, sys, platform
import numpy as np
import h5py
import cv2
import fnmatch
import scipy.io

import matplotlib
# matplotlib.use('Agg')      # without plot
import matplotlib.pyplot as plt


# Pytorch lib
import torch
import torch.nn as nn
import torch.nn.functional as F


from torchvision import transforms, datasets
from torchvision.transforms import Compose, Resize, ToTensor

# For Transformer
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce



from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from datetime import datetime

from math import sqrt
sc = StandardScaler()



# Version of this code script

version = "0.1.1"      # 1.) Use any string you like to specify/define your version of the model pipeline
def __version__():
    #print(version)
    return version


#2.) Insert the name of your model in this list. All those names here, will appear later in the GUI or code script.

predefined_model_1 = ["CNN_simple_Serial", "CNN_simple_Serial_mod1","CNN_simple_Serial_mod2", "VGGnet_16", "VGGnet_19"]
predefined_model_2 = ["Inception", "GoogLenet"]
predefined_model_3 = ["Resnet_18","Resnet_34","Resnet_50","Resnet_101","Resnet_152", 
                      "Densenet_121", "Densenet_121_custom" "Densenet_161", "Densenet_169", "Densenet_201"]
predefined_model_4 = ["Transformer", "VisualTransformer"]

# Coord-conv layer
predefined_model_5 = ["CNN_simple_Serial_coordconv", "VGGnet_16_coordconv", "VGGnet_19_coordconv"]     # basemodel(predefined_model_1) + Coordination convolution layer
predefined_model_6 = ["Inception_coordconv", "GoogLenet_coordconv"]     # basemodel(predefined_model_1) + Coordination convolution layer
predefined_model_7 = ["Resnet_coordconv", "Densenet_coordconv"]     # basemodel(predefined_model_1) + Coordination convolution layer

# Attention mechanism (CBAM or SE)
predefined_model_8 = ["CNN_simple_Serial_att_i_l", "CNN_simple_Serial_att_ac1_l",
                      "CNN_simple_Serial_att_ac2_l", "CNN_simple_Serial_att_ac3_l",
                      "CNN_simple_Serial_att_all_l"]           # basemodel(predefined_model_1) + Attention layer (each part)
predefined_model_9 = ["VGGnet_16_att_i_l", "VGGnet_16_att_ac1_l",
                      "VGGnet_16_att_ac2_l", "VGGnet_16_att_ac3_l",
                      "VGGnet_16_att_all_l"]           # basemodel(predefined_model_1) + Attention layer (each part)
predefined_model_10 = ["VGGnet_19_att_i_l", "VGGnet_19_att_ac1_l",
                      "VGGnet_19_att_ac2_l", "VGGnet_19_att_ac3_l",
                      "VGGnet_19_att_all_l"]           # basemodel(predefined_model_1) + Attention layer (each part)
predefined_model_11 = ["Inception_att_i_l", "Inception_att_l_l", 
                       "Inception_att_all_l"]           # basemodel(predefined_model_1) + Attention layer (each part)
predefined_model_12 = ["GoogLenet_att_i_l", "GoogLenet_att_l_l", 
                       "GoogLenet_att_all_l"]           # basemodel(predefined_model_1) + Attention layer (each part)
predefined_model_13 = ["Resnet_att_i_l", "Resnet_att_l_l", 
                       "Resnet_att_all_l"]           # basemodel(predefined_model_1) + Attention layer (each part)
predefined_model_14 = ["Densenet_att_i_l", "Densenet_att_l_l", 
                       "Densenet_att_all_l"]           # basemodel(predefined_model_1) + Attention layer (each part)

# Coord-conv layer + Attention mechnism (CBAM  or SE)
predefined_model_15 = ["CNN_simple_Serial_coordconv_att_i_l", "CNN_simple_Serial_coordconv_att_ac1_l",
                      "CNN_simple_Serial_coordconv_att_ac2_l", "CNN_simple_Serial_coordconv_att_ac3_l",
                      "CNN_simple_Serial_coordconv_att_all_l"]           # basemodel(predefined_model_1) + Coordination convolution layer + Attention layer (each part)
predefined_model_16 = ["VGGnet_16_coordconv_att_i_l", "VGGnet_16_coordconv_att_ac1_l",
                      "VGGnet_16_coordconv_att_ac2_l", "VGGnet_16_coordconv_att_ac3_l",
                      "VGGnet_16_coordconv_att_all_l"]           # basemodel(predefined_model_1) + Coordination convolution layer + Attention layer (each part)
predefined_model_17 = ["VGGnet_19_coordconv_att_i_l", "VGGnet_19_coordconv_att_ac1_l",
                      "VGGnet_19_coordconv_att_ac2_l", "VGGnet_19_coordconv_att_ac3_l",
                      "VGGnet_19_coordconv_att_all_l"]           # basemodel(predefined_model_1) + Coordination convolution layer + Attention layer (each part)
predefined_model_18 = ["Inception_coordconv_att_i_l", "Inception_coordconv_att_l_l", 
                       "Inception_coordconv_att_all_l"]           # basemodel(predefined_model_1) + Coordination convolution layer + Attention layer (each part)
predefined_model_19 = ["GoogLenet_coordconv_att_i_l", "GoogLenet_coordconv_att_l_l", 
                       "GoogLenet_coordconv_att_all_l"]           # basemodel(predefined_model_1) + Coordination convolution layer + Attention layer (each part)
predefined_model_20 = ["Resnet_coordconv_att_i_l", "Resnet_coordconv_att_l_l", 
                       "Resnet_coordconv_att_all_l"]           # basemodel(predefined_model_1) + Coordination convolution layer + Attention layer (each part)
predefined_model_21 = ["Densenet_coordconv_att_i_l", "Densenet_coordconv_att_l_l", 
                       "Densenet_coordconv_att_all_l"]           # basemodel(predefined_model_1) + Coordination convolution layer + Attention layer (each part)




predefined_models = predefined_model_1+predefined_model_2+predefined_model_3+predefined_model_4+\
    predefined_model_5+predefined_model_6+predefined_model_7+predefined_model_8+predefined_model_9+\
        predefined_model_10+predefined_model_11+predefined_model_12+predefined_model_13+predefined_model_14+\
            predefined_model_15+predefined_model_16+predefined_model_17+predefined_model_18+predefined_model_19+\
                predefined_model_20+predefined_model_21
                

        
def get_predefined_models():  # this function is used inside code scripts.
    return predefined_models


#3.) update the get_model function

def get_model(modelname, in_dim, input_channels, output_dimension):
    
    if modelname=="CNN_simple_Serial":
        model = CNN_simple_Serial(input_channels, output_dimension)
    elif modelname=="CNN_simple_Serial_mod1":
        model = CNN_simple_Serial_mod1(input_channels, output_dimension)
    elif modelname=="CNN_simple_Serial_mod2":
        model = CNN_simple_Serial_mod2(input_channels, output_dimension)
    elif modelname=="VGGnet_16":
        model = VGGnet_16(input_channels, output_dimension)
    elif modelname=="VGGnet_19":
        model = VGGnet_19(input_channels, output_dimension)
    elif modelname=="Inception":
        model = Inception(aux_logits=True, in_channels=input_channels, out_dim=output_dimension, init_weights=True)
    elif modelname=="GoogLenet":
        model = GoogLenet(aux_logits=True, in_channels=input_channels, out_dim=output_dimension, init_weights=True)
    elif modelname=="Resnet_18":
        model = Resnet(Resnet_Basicblock, [2, 2, 2, 2], in_channels=input_channels, out_dim=output_dimension, init_weights=True)
    elif modelname=="Resnet_34":
        model = Resnet(Resnet_Basicblock, [3, 4, 6, 3], in_channels=input_channels, out_dim=output_dimension, init_weights=True)
    elif modelname=="Resnet_50":
        model = Resnet(Resnet_Bottleneck, [3, 4, 6, 3], in_channels=input_channels, out_dim=output_dimension, init_weights=True)
    elif modelname=="Resnet_101":
        model = Resnet(Resnet_Bottleneck, [3, 4, 23, 3], in_channels=input_channels, out_dim=output_dimension, init_weights=True)
    elif modelname=="Resnet_152":
        model = Resnet(Resnet_Bottleneck, [3, 8, 36, 3], in_channels=input_channels, out_dim=output_dimension, init_weights=True)
    elif modelname=="Densenet_121":
        model = Densenet([6, 12, 24, 16], growth_rate=12, reduction=0.5, channels=input_channels, out_dim=output_dimension, init_weights=True)
    elif modelname=="Densenet_121_custom":
        model = Densenet([6, 12], growth_rate=12, reduction=0.5, channels=input_channels, out_dim=output_dimension, init_weights=True)
    elif modelname=="Densenet_161":
        model = Densenet([6, 12, 36, 24], growth_rate=12, reduction=0.5, channels=input_channels, out_dim=output_dimension, init_weights=True)
    elif modelname=="Densenet_169":
        model = Densenet([6, 12, 32, 32], growth_rate=12, reduction=0.5, channels=input_channels, out_dim=output_dimension, init_weights=True)
    elif modelname=="Densenet_201":
        model = Densenet([6, 12, 48, 32], growth_rate=12, reduction=0.5, channels=input_channels, out_dim=output_dimension, init_weights=True)
    elif modelname=="Transformer":
        model = ViT(in_channels=input_channels, n_classes=output_dimension)
    elif modelname=="VisualTransformer":
        model = ViT(in_channels=input_channels, n_classes=output_dimension)
    
    
    # elif modelname=="CNN_simple_Serial_coordconv":
    #     model = CNN_simple_Serial_coordconv(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_16_coordconv":
    #     model = VGGnet_16_coordconv(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_19_coordconv":
    #     model = VGGnet_19_coordconv(in_dim, channels, out_dim)
    # elif modelname=="Inception_coordconv":
    #     model = Inception_coordconv(in_dim, channels, out_dim)
    # elif modelname=="GoogLenet_coordconv":
    #     model = GoogLenet_coordconv(in_dim, channels, out_dim)
    # elif modelname=="Resnet_coordconv":
    #     model = Resnet_coordconv(in_dim, channels, out_dim)
    # elif modelname=="Densenet_coordconv":
    #     model = Densenet_coordconv(in_dim, channels, out_dim)
        
    
    # elif modelname=="CNN_simple_Serial_att_i_l":
    #     model = CNN_simple_Serial_att_i_l(in_dim, channels, out_dim)
    # elif modelname=="CNN_simple_Serial_att_ac1_l":
    #     model = CNN_simple_Serial_att_ac1_l(in_dim, channels, out_dim)
    # elif modelname=="CNN_simple_Serial_att_ac2_l":
    #     model = CNN_simple_Serial_att_ac2_l(in_dim, channels, out_dim)
    # elif modelname=="CNN_simple_Serial_att_ac3_l":
    #     model = CNN_simple_Serial_att_ac3_l(in_dim, channels, out_dim)
    # elif modelname=="CNN_simple_Serial_att_all_l":
    #     model = CNN_simple_Serial_att_all_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_16_att_i_l":
    #     model = VGGnet_16_att_i_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_16_att_ac1_l":
    #     model = VGGnet_16_att_ac1_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_16_att_ac2_l":
    #     model = VGGnet_16_att_ac2_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_16_att_ac3_l":
    #     model = VGGnet_16_att_ac3_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_16_att_all_l":
    #     model = VGGnet_16_att_all_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_19_att_i_l":
    #     model = VGGnet_19_att_i_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_19_att_ac1_l":
    #     model = VGGnet_19_att_ac1_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_19_att_ac2_l":
    #     model = VGGnet_19_att_ac2_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_19_att_ac3_l":
    #     model = VGGnet_19_att_ac3_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_19_att_all_l":
    #     model = VGGnet_19_att_all_l(in_dim, channels, out_dim)
    # elif modelname=="Inception_att_i_l":
    #     model = Inception_att_i_l(in_dim, channels, out_dim)
    # elif modelname=="Inception_att_l_l":
    #     model = Inception_att_l_l(in_dim, channels, out_dim)
    # elif modelname=="Inception_att_all_l":
    #     model = Inception_att_all_l(in_dim, channels, out_dim)
    # elif modelname=="GoogLenet_att_i_l":
    #     model = GoogLenet_att_i_l(in_dim, channels, out_dim)
    # elif modelname=="GoogLenet_att_l_l":
    #     model = GoogLenet_att_l_l(in_dim, channels, out_dim)
    # elif modelname=="GoogLenet_att_all_l":
    #     model = GoogLenet_att_all_l(in_dim, channels, out_dim)
    # elif modelname=="Resnet_att_i_l":
    #     model = Resnet_att_i_l(in_dim, channels, out_dim)
    # elif modelname=="Resnet_att_l_l":
    #     model = Resnet_att_l_l(in_dim, channels, out_dim)
    # elif modelname=="Resnet_att_all_l":
    #     model = Resnet_att_all_l(in_dim, channels, out_dim)
    # elif modelname=="Densenet_att_i_l":
    #     model = Densenet_att_i_l(in_dim, channels, out_dim)
    # elif modelname=="Densenet_att_l_l":
    #     model = Densenet_att_l_l(in_dim, channels, out_dim)
    # elif modelname=="Densenet_att_all_l":
    #     model = Densenet_att_all_l(in_dim, channels, out_dim)
        
        
    # elif modelname=="CNN_simple_Serial_coordconv_att_i_l":
    #     model = CNN_simple_Serial_coordconv_att_i_l(in_dim, channels, out_dim)
    # elif modelname=="CNN_simple_Serial_coordconv_att_ac1_l":
    #     model = CNN_simple_Serial_coordconv_att_ac1_l(in_dim, channels, out_dim)
    # elif modelname=="CNN_simple_Serial_coordconv_att_ac2_l":
    #     model = CNN_simple_Serial_coordconv_att_ac2_l(in_dim, channels, out_dim)
    # elif modelname=="CNN_simple_Serial_coordconv_att_ac3_l":
    #     model = CNN_simple_Serial_coordconv_att_ac3_l(in_dim, channels, out_dim)
    # elif modelname=="CNN_simple_Serial_coordconv_att_all_l":
    #     model = CNN_simple_Serial_coordconv_att_all_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_16_coordconv_att_i_l":
    #     model = VGGnet_16_coordconv_att_i_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_16_coordconv_att_ac1_l":
    #     model = VGGnet_16_coordconv_att_ac1_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_16_coordconv_att_ac2_l":
    #     model = VGGnet_16_coordconv_att_ac2_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_16_coordconv_att_ac3_l":
    #     model = VGGnet_16_coordconv_att_ac3_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_16_coordconv_att_all_l":
    #     model = VGGnet_16_coordconv_att_all_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_19_coordconv_att_i_l":
    #     model = VGGnet_19_coordconv_att_i_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_19_coordconv_att_ac1_l":
    #     model = VGGnet_19_coordconv_att_ac1_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_19_coordconv_att_ac2_l":
    #     model = VGGnet_19_coordconv_att_ac2_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_19_coordconv_att_ac3_l":
    #     model = VGGnet_19_coordconv_att_ac3_l(in_dim, channels, out_dim)
    # elif modelname=="VGGnet_19_coordconv_att_all_l":
    #     model = VGGnet_19_coordconv_att_all_l(in_dim, channels, out_dim)
    # elif modelname=="Inception_coordconv_att_i_l":
    #     model = Inception_coordconv_att_i_l(in_dim, channels, out_dim)
    # elif modelname=="Inception_coordconv_att_l_l":
    #     model = Inception_coordconv_att_l_l(in_dim, channels, out_dim)
    # elif modelname=="Inception_coordconv_att_all_l":
    #     model = Inception_coordconv_att_all_l(in_dim, channels, out_dim)
    # elif modelname=="GoogLenet_coordconv_att_i_l":
    #     model = GoogLenet_coordconv_att_i_l(in_dim, channels, out_dim)
    # elif modelname=="GoogLenet_coordconv_att_l_l":
    #     model = GoogLenet_coordconv_att_l_l(in_dim, channels, out_dim)
    # elif modelname=="GoogLenet_coordconv_att_all_l":
    #     model = GoogLenet_coordconv_att_all_l(in_dim, channels, out_dim)
    # elif modelname=="Resnet_coordconv_att_i_l":
    #     model = Resnet_coordconv_att_i_l(in_dim, channels, out_dim)
    # elif modelname=="Resnet_coordconv_att_l_l":
    #     model = Resnet_coordconv_att_l_l(in_dim, channels, out_dim)
    # elif modelname=="Resnet_coordconv_att_all_l":
    #     model = Resnet_coordconv_att_all_l(in_dim, channels, out_dim)
    # elif modelname=="Densenet_coordconv_att_i_l":
    #     model = Densenet_coordconv_att_i_l(in_dim, channels, out_dim)
    # elif modelname=="Densenet_coordconv_att_l_l":
    #     model = Densenet_coordconv_att_l_l(in_dim, channels, out_dim)
    # elif modelname=="Densenet_coordconv_att_all_l":
    #     model = Densenet_coordconv_att_all_l(in_dim, channels, out_dim)
        
    return model

              



#4. Define model architecture:

#Rules:
#- specify the name of the first layer as "inputTensor" (see examples below)
#- specify the name of the last layer as "outputTensor" (see examples below)
#- Reference: AIDeveloper currently only supports single-input (image) - single-ouput (prediction) models.
#- Code script for pytorch lib. should supports single-input - single-output models. 
#- but, it will be upgrade for multi-input and multi-output.
#- have a look at the example neural nets below if you like (they will not
#- appear in Reference(AIDeveloper) since they are not described in "predefined_models" and "get_model")

#############################################################################
############################## Neural net ###################################
#############################################################################

###################################################################################
############################## Predefined model 1 #################################
###################################################################################

class CNN_simple_Serial(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(CNN_simple_Serial, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.pool_beforeGAP = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(32, out_dim)
        self.output_activation = nn.LogSoftmax()
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool_beforeGAP(x)
        
        # x = self.conv4(x)
        # x = F.relu(x)
        # x = self.pool_beforeGAP(x)
        
        x = self.GAP(x)       # Golbal info, Global average pooling
        
        x = x.view(x.size(0), -1)   # Flatten or reshape x.
        
        x = self.fc1(x)
        x = self.output_activation(x)
        return x
    
class CNN_simple_Serial_mod1(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(CNN_simple_Serial_mod1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
        self.BTN1 = nn.BatchNorm2d(8)
        self.BTN2 = nn.BatchNorm2d(16)
        self.BTN3 = nn.BatchNorm2d(32)
        self.BTN4 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.pool_beforeGAP = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(3*3*32, out_dim)      # last layer 3 x 3 x 32 = 288
        self.output_activation = nn.LogSoftmax()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.BTN1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.BTN2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.BTN3(x)
        x = F.relu(x)
        x = self.pool_beforeGAP(x)
        
        x = self.conv4(x)
        x = self.BTN4(x)
        x = F.relu(x)
        # x = self.pool_beforeGAP(x)
        
        # x = self.GAP(x)       # Golbal info, Global average pooling
        
        
        x = x.view(x.size(0), -1)   # Flatten or reshape x.
        
        x = self.dropout(x)    # Dropout

        x = self.fc1(x)
        x = self.output_activation(x)
        return x

class CNN_simple_Serial_mod2(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(CNN_simple_Serial_mod2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 8, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
        self.BTN1 = nn.BatchNorm2d(8)
        self.BTN2 = nn.BatchNorm2d(16)
        self.BTN3 = nn.BatchNorm2d(32)
        self.BTN4 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.pool_beforeGAP = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(3*3*32, out_dim)      # last layer 3 x 3 x 32 = 288
        self.output_activation = nn.LogSoftmax()
        
        self._initialize_weights()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.BTN1(x)
        x = F.relu(x)
        x = self.avgpool(x)
        
        x = self.conv2(x)
        x = self.BTN2(x)
        x = F.relu(x)
        x = self.avgpool(x)
        
        x = self.conv3(x)
        x = self.BTN3(x)
        x = F.relu(x)
        x = self.avgpool(x)
        
        x = self.conv4(x)
        x = self.BTN4(x)
        x = F.relu(x)
        # x = self.pool_beforeGAP(x)
        
        # x = self.GAP(x)       # Golbal info, Global average pooling
        
        
        x = x.view(x.size(0), -1)   # Flatten or reshape x.
        # x = x.view(x.size()[0])   # Flatten or reshape x.
        
        x = self.dropout(x)    # Dropout

        x = self.fc1(x)
        x = self.output_activation(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()
    
    def compute_l2_loss(self, w):
        return torch.square(w).sum()
    
class VGGnet_16(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(VGGnet_16, self).__init__()
        
        conv1_filter_num = 64
        conv2_filter_num = 128
        conv3_filter_num = 256
        conv4_filter_num = 512
        fc12_layer = 4096
        fc3_layer = out_dim
                
        
        self.conv1_1 = nn.Conv2d(in_channels = in_channels, out_channels = conv1_filter_num, kernel_size = 3, padding = 1)
        self.conv1_2 = nn.Conv2d(in_channels = conv1_filter_num, out_channels = conv1_filter_num, kernel_size = 3, padding = 1)
        
        self.conv2_1 = nn.Conv2d(in_channels = conv1_filter_num, out_channels = conv2_filter_num, kernel_size = 3, padding = 1)
        self.conv2_2 = nn.Conv2d(in_channels = conv2_filter_num, out_channels = conv2_filter_num, kernel_size = 3, padding = 1)
        
        self.conv3_1 = nn.Conv2d(in_channels = conv2_filter_num, out_channels = conv3_filter_num, kernel_size = 3, padding = 1)
        self.conv3_2 = nn.Conv2d(in_channels = conv3_filter_num, out_channels = conv3_filter_num, kernel_size = 3, padding = 1)
        self.conv3_3 = nn.Conv2d(in_channels = conv3_filter_num, out_channels = conv3_filter_num, kernel_size = 3, padding = 1)
        
        self.conv4_1 = nn.Conv2d(in_channels = conv3_filter_num, out_channels = conv4_filter_num, kernel_size = 3, padding = 1)
        self.conv4_2 = nn.Conv2d(in_channels = conv4_filter_num, out_channels = conv4_filter_num, kernel_size = 3, padding = 1)
        self.conv4_3 = nn.Conv2d(in_channels = conv4_filter_num, out_channels = conv4_filter_num, kernel_size = 3, padding = 1)
                
        self.conv5_1 = nn.Conv2d(in_channels = conv4_filter_num, out_channels = conv4_filter_num, kernel_size = 3, padding = 1)
        self.conv5_2 = nn.Conv2d(in_channels = conv4_filter_num, out_channels = conv4_filter_num, kernel_size = 3, padding = 1)
        self.conv5_3 = nn.Conv2d(in_channels = conv4_filter_num, out_channels = conv4_filter_num, kernel_size = 3, padding = 1)
        
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.pool_beforeGAP = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(fc12_layer, fc12_layer)
        self.fc2 = nn.Linear(fc12_layer, fc12_layer)
        self.fc3 = nn.Linear(fc12_layer, out_dim)
        self.output_activation = nn.LogSoftmax()
        
    def forward(self, x):
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = F.relu(x)
        x = self.conv3_3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv4_1(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = F.relu(x)
        x = self.conv4_3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv5_1(x)
        x = F.relu(x)
        x = self.conv5_2(x)
        x = F.relu(x)
        x = self.conv5_3(x)
        x = F.relu(x)
        x = self.pool_beforeGAP(x)
        
        x = self.GAP(x)       # Golbal info, Global average pooling
        
        x = x.view(x.size(0), -1)   # Flatten or reshape x.
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.output_activation(x)
        
        return x
        
        
    
class VGGnet_19(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(VGGnet_19, self).__init__()
        
        conv1_filter_num = 64
        conv2_filter_num = 128
        conv3_filter_num = 256
        conv4_filter_num = 512
        fc12_layer = 4096
        fc3_layer = out_dim
                
        
        self.conv1_1 = nn.Conv2d(in_channels = in_channels, out_channels = conv1_filter_num, kernel_size = 3, padding = 1)
        self.conv1_2 = nn.Conv2d(in_channels = conv1_filter_num, out_channels = conv1_filter_num, kernel_size = 3, padding = 1)
        
        self.conv2_1 = nn.Conv2d(in_channels = conv1_filter_num, out_channels = conv2_filter_num, kernel_size = 3, padding = 1)
        self.conv2_2 = nn.Conv2d(in_channels = conv2_filter_num, out_channels = conv2_filter_num, kernel_size = 3, padding = 1)
        
        self.conv3_1 = nn.Conv2d(in_channels = conv2_filter_num, out_channels = conv3_filter_num, kernel_size = 3, padding = 1)
        self.conv3_2 = nn.Conv2d(in_channels = conv3_filter_num, out_channels = conv3_filter_num, kernel_size = 3, padding = 1)
        self.conv3_3 = nn.Conv2d(in_channels = conv3_filter_num, out_channels = conv3_filter_num, kernel_size = 3, padding = 1)
        self.conv3_4 = nn.Conv2d(in_channels = conv3_filter_num, out_channels = conv3_filter_num, kernel_size = 3, padding = 1)
        
        self.conv4_1 = nn.Conv2d(in_channels = conv3_filter_num, out_channels = conv4_filter_num, kernel_size = 3, padding = 1)
        self.conv4_2 = nn.Conv2d(in_channels = conv4_filter_num, out_channels = conv4_filter_num, kernel_size = 3, padding = 1)
        self.conv4_3 = nn.Conv2d(in_channels = conv4_filter_num, out_channels = conv4_filter_num, kernel_size = 3, padding = 1)
        self.conv4_4 = nn.Conv2d(in_channels = conv4_filter_num, out_channels = conv4_filter_num, kernel_size = 3, padding = 1)
                
        self.conv5_1 = nn.Conv2d(in_channels = conv4_filter_num, out_channels = conv4_filter_num, kernel_size = 3, padding = 1)
        self.conv5_2 = nn.Conv2d(in_channels = conv4_filter_num, out_channels = conv4_filter_num, kernel_size = 3, padding = 1)
        self.conv5_3 = nn.Conv2d(in_channels = conv4_filter_num, out_channels = conv4_filter_num, kernel_size = 3, padding = 1)
        self.conv5_4 = nn.Conv2d(in_channels = conv4_filter_num, out_channels = conv4_filter_num, kernel_size = 3, padding = 1)
        
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.pool_beforeGAP = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(fc12_layer, fc12_layer)
        self.fc2 = nn.Linear(fc12_layer, fc12_layer)
        self.fc3 = nn.Linear(fc12_layer, out_dim)
        self.output_activation = nn.LogSoftmax()
        
    def forward(self, x):
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = F.relu(x)
        x = self.conv3_3(x)
        x = F.relu(x)
        x = self.conv3_4(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv4_1(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = F.relu(x)
        x = self.conv4_3(x)
        x = F.relu(x)
        x = self.conv4_4(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv5_1(x)
        x = F.relu(x)
        x = self.conv5_2(x)
        x = F.relu(x)
        x = self.conv5_3(x)
        x = F.relu(x)
        x = self.conv5_4(x)
        x = F.relu(x)
        x = self.pool_beforeGAP(x)
        
        x = self.GAP(x)       # Golbal info, Global average pooling
        
        x = x.view(x.size(0), -1)   # Flatten or reshape x.
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.output_activation(x)
        
        return x    
    
    
###################################################################################
############################## Predefined model 2 #################################
###################################################################################  

#  Inception (same as GoogLenet, it should be modified.)
class Inception(nn.Module):
    def __init__(self, aux_logits, in_channels, out_dim, init_weights):
        super(GoogLenet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits
        
        # conv_block takes in_channels, out_channels, kernel_size, stride, padding
        # Inception block takes out(1x1), red(3x3), out(3,3), red(5x5), out(5x5), out(1x1) pool.
        
        self.conv1 = conv_block(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, 2, 1)
        #nn.MaxPool2d -> (kernel_size, stride, padding)
        self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, 2, 1)
        
        # Inception block takes out (1x1), red (3x3), out (3x3), red (5x5), out (5x5), out (1x1) pool.
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, 2, 1)
        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        
        # Auxiliary classifier
        
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        
        # Auxiliary classifier
        
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, 2, 1)
        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, out_dim)
        self.output_activation = nn.LogSoftmax()
        
        if self.aux_logits:
            self.aux1 = InceptionAux(512, out_dim)
            self.aux2 = InceptionAux(528, out_dim)
        else:
            self.aux1 = self.aux2 = None
            
        # weight initialization
        
        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
        
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.output_activation(x)
        
        if self.aux_logits and self.training:
            return x, aux1, aux2
        else:
            return x
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class GoogLenet(nn.Module):
    def __init__(self, aux_logits, in_channels, out_dim, init_weights):
        super(GoogLenet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits
        
        # conv_block takes in_channels, out_channels, kernel_size, stride, padding
        # Inception block takes out(1x1), red(3x3), out(3,3), red(5x5), out(5x5), out(1x1) pool.
        
        self.conv1 = conv_block(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, 2, 1)
        #nn.MaxPool2d -> (kernel_size, stride, padding)
        self.conv2 = conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, 2, 1)
        
        # Inception block takes out (1x1), red (3x3), out (3x3), red (5x5), out (5x5), out (1x1) pool.
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, 2, 1)
        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        
        # Auxiliary classifier
        
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        
        # Auxiliary classifier
        
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, 2, 1)
        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, out_dim)
        self.output_activation = nn.LogSoftmax()
        
        if self.aux_logits:
            self.aux1 = InceptionAux(512, out_dim)
            self.aux2 = InceptionAux(528, out_dim)
        else:
            self.aux1 = self.aux2 = None
            
        # weight initialization
        
        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
        
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.output_activation(x)
        
        if self.aux_logits and self.training:
            return x, aux1, aux2
        else:
            return x
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# For GoogLenet/Inception.

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.conv_layer(x)
    
class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()
        
        # out 1x1 conv block
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)
        
        # out 3x3 conv block
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1),
        )
        
        # out 5x5 conv block
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2),
        )
        
        # out_1x1pool conv block
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1)
        )
        
    def forward(self, x):
        # 0 dimension is batch, 
        # branch output concatenate based 1 dimension (column) filter number
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
        return x
    

# the loss of auxiliary classifier multiplied 0.3 value,
# After then, this value was added the last loss
# It is the effect of normalization.

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        
        self.conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            conv_block(in_channels, 128, kernel_size=1),
        )
        
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )
        
        self.output_activation_Aux = nn.LogSoftmax()
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.output_activation_Aux(x)
        
        return x


###################################################################################
############################## Predefined model 3 #################################
###################################################################################  

# Resnet

class Resnet_Basicblock(nn.Module):
    # Basic block for resnet  - 18 and resnet 34
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # BatchNorm include bias. conv2d set bias = false.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * Resnet_Basicblock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * Resnet_Basicblock.expansion)
        )
        
        # Identity mapping, It was used when number of filter and the feature map size of input and output was same
        
        self.shortcut = nn.Sequential()   # summation part.
        
        self.relu = nn.ReLU()
        
        # Projection mapping using 1x1 conv
        
        if stride != 1 or in_channels != Resnet_Basicblock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Resnet_Basicblock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Resnet_Basicblock.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x
    
class Resnet_Bottleneck(nn.Module):
    # Bottleneck block for resnet over 50 layers....
    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.residual_function == nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,  stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * Resnet_Bottleneck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * Resnet_Bottleneck.expansion)
        )
        
        self.shortcut = nn.Sequential()
        
        self.relu = nn.ReLU()
        
        if stride != 1 or in_channels != out_channels * Resnet_Bottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Resnet_Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Resnet_Bottleneck.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x
    
    
class Resnet(nn.Module):
    def __init__(self, block, num_block, in_channels, out_dim, init_weights=True):
        super().__init__()
        
        self.in_channels_for_inputlayer = 64
        self.num_classes=out_dim
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # block: Resnet_basicblock, Resnet_Bottleneck block
        
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)
        
        self.output_activation = nn.LogSoftmax()
        
        # weights initialization
        if init_weights:
            self._initialize_weights()
            
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # append [1] array.. number of num_blocks -1. ; Stride may affect output size.
        # print(strides)
        # layers = []
        layers  = nn.Sequential()
        for i, stride in enumerate(strides):
            # print(i, stride)
            layers.add_module('Block_layer_{}'.format(i), block(self.in_channels_for_inputlayer, out_channels, stride))
            self.in_channels_for_inputlayer = out_channels * block.expansion            
        return layers
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.output_activation(x)
        
        return x
    
    # define weight initialization function
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                


# Densenet structure

class Dense_BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        
        inner_channels = 4 * growth_rate
        
        
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, inner_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, growth_rate, 3, stride=1, padding=1, bias=False),
        )        
        self.shortcut = nn.Sequential()  # bottleneck (concat part)
        
    def forward(self, x):
        return torch.cat([self.shortcut(x), self.residual(x)], 1)
    

# Transition Block: reduce feature map s ize and n umber of channels
class Dense_Transition(nn.Module):
    def __init__(self, in_channels,  out_channels):
        super().__init__()
        
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(2, stride=2)
        )
        
    def forward(self, x):
        return self.down_sample(x)
    
# Densenet
class Densenet(nn.Module):
    def __init__(self, nblocks, growth_rate=12, reduction=0.5, channels=3, out_dim=3, init_weights=True):
        super().__init__()
        
                
        self.growth_rate = growth_rate
        inner_channels =  2 * growth_rate    # output channels of conv1 b efore entering Dense Block
        
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, inner_channels, 7, stride=2, padding=3),
            nn.MaxPool2d(3, 2, padding=1)
        )
        
        self.features = nn.Sequential()
        
        for i in range(len(nblocks)-1):
            self.features.add_module('dense_block_{}'.format(i), self._make_dense_block(nblocks[i], inner_channels))
            inner_channels += growth_rate * nblocks[i]
            out_channels = int(reduction * inner_channels)
            self.features.add_module('transition_layer_{}'.format(i), Dense_Transition(inner_channels, out_channels))
            inner_channels =  out_channels
        
        self.features.add_module('dense_block_{}'.format(len(nblocks)-1), self._make_dense_block(nblocks[len(nblocks)-1], inner_channels))
        inner_channels += growth_rate * nblocks[len(nblocks)-1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU())
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(inner_channels, out_dim)
        
        self.output_activation = nn.LogSoftmax()
        
        
        # Weight initialization
        if init_weights:
            self._initialize_weights()
        
    def _make_dense_block(self, nblock, inner_channels):
        dense_block = nn.Sequential()
        for i in range(nblock):
            dense_block.add_module('bottle_neck_layer_{}'.format(i), Dense_BottleNeck(inner_channels, self.growth_rate))
            inner_channels += self.growth_rate
            
        return dense_block
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    

###################################################################################
############################## Predefined model 4 #################################
###################################################################################

        
# Visual Transformer

class PatchEmbedding_CLSToken_PositionEmbedding(nn.Module):
    def __init__(self, in_channels: int=3, patch_size: int=16, emb_size: int=768, img_size: int=224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            
            # Kernel_size -> Patch_size, 32 x 32.
            # Output width = (Input width - Kernel size + Zero padding size) / Strides + 1
            # based on above formula, (Input width = patchsize)/patchsize + 1
            # if patchsize 32, input image 224, output width is 7 because (224-32)/32+1 = 7
            
            # Thus, batchsize x emb_size(32*32*3) x output width (7) x output height (7)
            # output emb size is batchsize x 49 x (32*32*3)
            
            
            # Using a conv layer instead of a linear one -> performance gains.
            
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        # cls_token is a torch parameter randomly initialized, in the forward the method
        # it is copied b (batch) times and prepended before the projected patches using torch.cat
        # torch.randn -> randomly get mean 0, std 1 Gaussian distribution;; size -> 1 x 1 x emb_size.
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        
        
        return x
            

# ----------- Transformer ------------ #

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int=768, num_heads: int=8, dropout: float=0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size*3)
        # self.keys = nn.Linear(emb_size, emb_size)
        # self.queries = nn.Linear(emb_size, emb_size)
        # self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # Split keys, queries and values in num_heads
        # n : number of patches.
        # h : number_heads.
        # d : dropout?
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        #queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        #keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        #values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        
        # sum up over the last axis
        # einsum -> Conduct dot product
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)   # Batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

    
# ResidualAdd: what for?

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int=4, drop_p: float =0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, 
                 emb_size: int=768,
                 drop_p: float=0.,
                 forward_expansion: int=4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
                )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
                )
                ))
        
        
# Define Transformer structure

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int=12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
    
    

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int=768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))
    

class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int=3,
                 patch_size: int=16, 
                 emb_size: int=768,
                 img_size: int=224,
                 depth: int=12,
                 n_classes: int=1000,
                 **kwargs):
        super().__init__(
            PatchEmbedding_CLSToken_PositionEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

        


        
        
        


        
        
        
    










































