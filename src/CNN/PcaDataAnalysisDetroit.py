#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 20:55:21 2025

@author: root
"""

#%% Some Remarks 21/11/22
# This Module is written by Arie Pyasik for APA.inc It is based out of unet and
# the following article :
# https://github.com/ArkaJU/U-Net-Satellite/blob/master/U-Net.ipynb
# currently based on KERAS API and Tensorflow 2
#

#%% Imports
import numpy as np 
import os
import cv2
import h5py
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, TensorBoard
# import tensorflow_addons as tfa
from keras import backend as keras
import src.CNN.DataAugmentation_Module as da
import matplotlib.pyplot as plt
import src.utils.io_utils as io_utils
import Costum_loss_module as cl
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tqdm as tq
        
#%% Load data and prepare it for analysis

print('*'*30)
print('Loading and preprocessing train data...')
print('*'*30)
# TODO: sort out the get path for the data
# Get config


# Get the directory of the script being executed
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate to the project root and construct the config path
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
config_path = os.path.join(project_root, 'configs', 'apa_config_shallow_detroit.yaml')

config = io_utils.read_yaml_config(config_path)
config=io_utils.fill_with_defaults(config['config'])


n_classes = config["cnn_model"]["num_classes"] # Need to read from config
input_size = eval(config["cnn_model"]["input_shape"])

hsp_path = os.path.join(project_root, config['data']['hsp_path'])
PCI_path = os.path.join(project_root, config['data']['PCI_path'])

# TODO : Remove after config is updated
# hsp_path = os.path.join(project_root, 'tests/runners/BoudingBoxList-83_-83_42_42.h5')
# PCI_path = os.path.join(project_root, 'tests/runners/BoudingBoxLabel-83_-83_42_42.h5')

file_hsp = h5py.File(hsp_path, 'r')
file_PCI = h5py.File(PCI_path, 'r')
img_train = file_hsp['cropped_segments'][:]
mask_train = file_PCI['cropped_segments'][:]
img_train = np.array(img_train)
mask_train = np.array(mask_train)

img_train[np.isnan(img_train)] = 0 # Why there were nones ? maybe in seg mask process ?
img_train = img_train.astype('float32')

## Normilize ##
img_train_mean = np.mean(img_train.flatten()[img_train.flatten() != 0])
img_train_std = np.std(img_train.flatten()[img_train.flatten() != 0])
non_zeros_idx = (img_train != 0)
img_train[non_zeros_idx] = (img_train[non_zeros_idx] - img_train_mean)/img_train_std # Scale

# Move between 0 and 1
img_train[non_zeros_idx] = (img_train[non_zeros_idx] - np.min(img_train[non_zeros_idx]))
img_train[non_zeros_idx] = img_train[non_zeros_idx]/np.max(img_train[non_zeros_idx])

# TODO: remove this line
# Add noise to the final result - TODO: remove this line
# img_train = da.add_random_noise(img_train) # Add noise to the final result

# Reshape for Hybrid CNN
img_train = np.reshape(img_train, list(np.shape(img_train)) + [1])
mask_train = mask_train.astype('float32')

# Get labels From Mask
num_of_examples = np.shape(mask_train)[0]
label_ls = np.zeros([num_of_examples, n_classes]).astype(int)
for i in range(num_of_examples):
    if len(np.unique(mask_train[i])) > 1 :
        label_idx = int(np.unique(mask_train[i])[1] - 1)
        label_ls[i, label_idx] = 1

#%% Split Test Train

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    img_train, label_ls, test_size=0.2, random_state=1
)

spectral_pixels_ls = np.zeros([0, 12])
pixel_label_ls = np.zeros([0,])
from tqdm import tqdm
for i in tqdm(range(np.shape(x_train)[0])):
    if np.sum(label_ls[i, :])==0:
        continue
    
    x_trainIdx = (x_train[i, :, :, 0, 0] != 0)
    spectral_pixels = x_train[i, x_trainIdx][:, :, 0]
    spectral_pixels_ls = np.append(spectral_pixels_ls, spectral_pixels, 0)
    
    if label_ls[i, 0] == 1:
        pixel_label_ls = np.append(pixel_label_ls, 0*np.ones(np.shape(spectral_pixels)[0]))
    elif label_ls[i, 1] == 1:
        pixel_label_ls = np.append(pixel_label_ls, 1*np.ones(np.shape(spectral_pixels)[0]))
    elif label_ls[i, 2] == 1:
        pixel_label_ls = np.append(pixel_label_ls, 2*np.ones(np.shape(spectral_pixels)[0]))
        
    if len(pixel_label_ls) != np.shape(spectral_pixels_ls)[0]:
        1

#%% Perform PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(spectral_pixels_ls)

# Perform PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Plot the first three principal components in a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

color_ls = ['r', 'y', 'g']
for i in range(3):
    plot_idx = (pixel_label_ls == i)
    ax.scatter(X_pca[plot_idx, 0][0:300], X_pca[plot_idx, 1][0:300], X_pca[plot_idx, 2][0:300], c=color_ls[i], marker='.')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('PCA: First Three Principal Components')

plt.show()


#%% Train an SVM classifier (multi-class)
svm = SVC(kernel='rbf', C=1, gamma='scale', decision_function_shape='ovo')  # One-vs-One (OvO)
maxIdx = int(50e3)
svm.fit(X_pca[0:maxIdx], pixel_label_ls[0:maxIdx])

# Predict using SVM
y_pred = svm.predict(X_pca[maxIdx:maxIdx+20000])
y_test = pixel_label_ls[maxIdx:maxIdx+20000]
#%% Plot Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Normalize confusion matrix to show percentages
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True) * 100

# Plot confusion matrix with percentages
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_percent, annot=True, fmt=".1f", cmap='Reds', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Percentage) - Multi-Class SVM')
plt.show()


# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

