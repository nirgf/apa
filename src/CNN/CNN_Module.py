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

import tensorflow as tf
import keras.backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow_addons as tfa
from keras import backend as keras

import matplotlib.pyplot as plt

#%% Get some functions
# def dice_coef(y_true, y_pred, smooth=1e-6):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# def dice_coef_loss(y_true, y_pred):
#     return 1-dice_coef(y_true, y_pred)

# def iou_coef(y_true, y_pred, smooth=1):
#   intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
#   union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
#   iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
#   return iou

#%% Define Net

def unet_categorical(input_size = (32, 32, 12),  n_classes = 4 , use_focal = False):
    inputs = Input(input_size)
    # norm_inputs = BatchNormalization()(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                                               kernel_initializer = 'he_normal'
                                               )(inputs)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                                               kernel_initializer = 'he_normal'
                                               )(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                                                kernel_initializer = 'he_normal'
                                                )(pool1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                                                kernel_initializer = 'he_normal'
                                                )(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                                                kernel_initializer = 'he_normal'
                                                )(pool2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                                                kernel_initializer = 'he_normal'
                                                )(conv3)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', 
                                                kernel_initializer = 'he_normal'
                                                )(pool3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', 
                                                kernel_initializer = 'he_normal'
                                                )(conv4)
    
    drop4 = Dropout(0.5)(conv4)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', 
                                                 kernel_initializer = 'he_normal'
                                                 )(pool4)
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', 
                                                 kernel_initializer = 'he_normal'
                                                 )(conv5)
    
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', 
                                              kernel_initializer = 'he_normal'
                                              )(UpSampling2D(size = (2,2))(drop5))
    
    merge6 = concatenate([drop4,up6])
    
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same',
                                                kernel_initializer = 'he_normal'
                                                )(merge6)
    
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same',
                                                kernel_initializer = 'he_normal'
                                                )(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same',
                                              kernel_initializer = 'he_normal'
                                              )(UpSampling2D(size = (2,2))(conv6))
    
    merge7 = concatenate([conv3,up7])
    
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                                                kernel_initializer = 'he_normal'
                                                )(merge7)
    
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'
                                                )(conv7)

    
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same',
                                              kernel_initializer = 'he_normal'
                                              )(UpSampling2D(size = (2,2))(conv7))
    
    merge8 = concatenate([conv2,up8])
    
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same',
                                                kernel_initializer = 'he_normal'
                                                )(merge8)
    
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same',
                                                kernel_initializer = 'he_normal'
                                                )(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same',
                                             kernel_initializer = 'he_normal'
                                             )(UpSampling2D(size = (2,2))(conv8))
    
    merge9 = concatenate([conv1,up9])
    
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                                               kernel_initializer = 'he_normal'
                                               )(merge9)
    
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                                               kernel_initializer = 'he_normal'
                                               )(conv9)
    
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same',
                                              kernel_initializer = 'he_normal'
                                              )(conv9)
    
   
    conv10 = Conv2D(n_classes, 1, activation='softmax')(conv9)
    # conv10 = Conv2D(1, 1)(conv9)

    model = Model(inputs = inputs, outputs = conv10)

#    model.compile(optimizer = Adam(learning_rate = 1e-3), loss = 'mse', \
#                  metrics=['mean_absolute_error'])
    if use_focal:
        # Compile with the built-in Focal Loss
        model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0),
                  metrics=['accuracy'])
    else:
        model.compile(Adam(learning_rate = 1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    return model
#%% For Continues data
def unet_smooth(input_size = (32, 32, 12)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                                               kernel_initializer = 'he_normal'
                                               )(inputs)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                                               kernel_initializer = 'he_normal'
                                               )(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                                                kernel_initializer = 'he_normal'
                                                )(pool1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                                                kernel_initializer = 'he_normal'
                                                )(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                                                kernel_initializer = 'he_normal'
                                                )(pool2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                                                kernel_initializer = 'he_normal'
                                                )(conv3)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', 
                                                kernel_initializer = 'he_normal'
                                                )(pool3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', 
                                                kernel_initializer = 'he_normal'
                                                )(conv4)
    
    drop4 = Dropout(0.5)(conv4)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', 
                                                 kernel_initializer = 'he_normal'
                                                 )(pool4)
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', 
                                                 kernel_initializer = 'he_normal'
                                                 )(conv5)
    
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', 
                                              kernel_initializer = 'he_normal'
                                              )(UpSampling2D(size = (2,2))(drop5))
    
    merge6 = concatenate([drop4,up6])
    
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same',
                                                kernel_initializer = 'he_normal'
                                                )(merge6)
    
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same',
                                                kernel_initializer = 'he_normal'
                                                )(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same',
                                              kernel_initializer = 'he_normal'
                                              )(UpSampling2D(size = (2,2))(conv6))
    
    merge7 = concatenate([conv3,up7])
    
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                                                kernel_initializer = 'he_normal'
                                                )(merge7)
    
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'
                                                )(conv7)

    
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same',
                                              kernel_initializer = 'he_normal'
                                              )(UpSampling2D(size = (2,2))(conv7))
    
    merge8 = concatenate([conv2,up8])
    
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same',
                                                kernel_initializer = 'he_normal'
                                                )(merge8)
    
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same',
                                                kernel_initializer = 'he_normal'
                                                )(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same',
                                             kernel_initializer = 'he_normal'
                                             )(UpSampling2D(size = (2,2))(conv8))
    
    merge9 = concatenate([conv1,up9])
    
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                                               kernel_initializer = 'he_normal'
                                               )(merge9)
    
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                                               kernel_initializer = 'he_normal'
                                               )(conv9)
    
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same',
                                              kernel_initializer = 'he_normal'
                                              )(conv9)
    
    conv10 = Conv2D(1, 1)(conv9)


    model.compile(optimizer = Adam(learning_rate = 1e-3), loss = 'mse', \
                 metrics=['mean_absolute_error'])

    return model
#%% For classification
def unet_smooth(input_size = (32, 32, 12)):
    inputs = Input(input_size)
    norm_inputs = BatchNormalization()(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                                               kernel_initializer = 'he_normal'
                                               )(norm_inputs)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                                               kernel_initializer = 'he_normal'
                                               )(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                                                kernel_initializer = 'he_normal'
                                                )(pool1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
                                                kernel_initializer = 'he_normal'
                                                )(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
                                                kernel_initializer = 'he_normal'
                                                )(pool2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                                                kernel_initializer = 'he_normal'
                                                )(conv3)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', 
                                                kernel_initializer = 'he_normal'
                                                )(pool3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', 
                                                kernel_initializer = 'he_normal'
                                                )(conv4)
    
    drop4 = Dropout(0.5)(conv4)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', 
                                                 kernel_initializer = 'he_normal'
                                                 )(pool4)
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', 
                                                 kernel_initializer = 'he_normal'
                                                 )(conv5)
    
    model.compile(optimizer = Adam(learning_rate = 1e-3), loss = 'mse', \
                 metrics=['mean_absolute_error'])

#%% Seq 3D-CNN
# This is a sequantial model based on the course 'Analyzanig satellite hyperspectral imegery' on Udemy
# https://www.udemy.com/course/hyperspectral-satellite-image-classification-using-deep-cnns/learn/lecture/36102952?start=0#overview
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.optimizers import Adam

def functional_cnn(n_categories=4, input_size=(5, 5, 12)):
    inputs = layers.Input(shape=list(input_size) + [1], name='Input_Layer')

    # First Conv3D block
    x = layers.Conv3D(16, kernel_size=(2, 2, 1), activation='linear', 
                      padding='same')(inputs)
    x = layers.LeakyReLU(negative_slope=0.1)(x)

    # Second Conv3D block
    x = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='linear', 
                      padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)

    # Third Conv3D block
    x = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='linear', 
                      padding='same', 
                      kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)

    # Fourth Conv3D block
    x = layers.Conv3D(128, kernel_size=(3, 3, 3), activation='linear', 
                      padding='same', 
                      kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)

    # Fifth Conv3D block
    x = layers.Conv3D(256, kernel_size=(3, 3, 3), activation='linear', 
                      padding='same', 
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)

    # Sixth Conv3D block
    x = layers.Conv3D(12, kernel_size=(3, 3, 3), activation='linear', 
                      padding='same', 
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)

    # MaxPooling3D
    x = layers.MaxPooling3D(pool_size=(4, 4, 2), padding='same')(x)

    # Flattening
    x = layers.Flatten()(x)

    # Fully connected layer
    x = layers.Dense(20, activation='linear', 
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(0.5)(x)

    # Output layer
    outputs = layers.Dense(n_categories, activation='softmax')(x)

    # Model definition
    model = Model(inputs=inputs, outputs=outputs, name='Road_3DCNN')

    # Compile the model
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=Adam(learning_rate=1e-2), 
        metrics=['accuracy']
    )

    return model

#%% Hybrid CNN
def HybridCNN():
 
    input_shape =  3, 3, 12, 1
    n_outputs = 4

     
    imIn = Input(shape=input_shape)
    
    conv_layer1 = Conv3D(filters=16, kernel_size=(1, 1, 7), activation='relu', padding='same')(imIn)
    conv_layer2 = Conv3D(filters=32, kernel_size=(3, 3, 5), activation='relu',padding='same')(conv_layer1)
    conv_layer3 = Conv3D(filters=32, kernel_size=(5, 5, 7), activation='relu',padding='same')(conv_layer2)

    conv3d_shape = conv_layer3.shape
    conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)

    conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu',padding='same')(conv_layer3)
    conv_layer5 = Conv2D(filters=64, kernel_size=(3,3), activation='relu',padding='same')(conv_layer4)

    conv_layer5 = GlobalAveragePooling2D()(conv_layer5)
    
    flatten_layer = Flatten()(conv_layer5)

    dense_layer1 = Dense(units=50, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)

    dense_layer2 = Dense(units=20, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
  
    output_layer = Dense(units=n_outputs, activation='softmax')(dense_layer2)
 
    model = Model(inputs=[imIn], outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate = 1e-3),\
                  metrics=['accuracy'])



    return model
#%% Run Preprocess

print('*'*30)
print('Loading and preprocessing train data...')
print('*'*30)
# TODO: sort out the get path for the data
# TODO: rename to CNN_Module without V2

n_classes = 4 # Need to read from config
file_hsp = h5py.File('Labeld_RoadsVenus.h5', 'r')
file_PCI = h5py.File('PCI_labels.h5', 'r')
img_train = file_hsp['cropped_segments'][:]
mask_train = file_PCI['cropped_segments'][:]
img_train = np.array(img_train)
mask_train = np.array(mask_train)

categorical_mask_train = np.zeros(list(np.shape(mask_train)[:-1]) + [n_classes])
mask_train_int = mask_train.astype('int')
create_matrix_labels = False
for i in range(n_classes):
    class_idx = np.where(mask_train_int == i)
    if create_matrix_labels :
        categorical_mask_train[class_idx[0], class_idx[1], class_idx[2], class_idx[3]+i] = 1    
    else:
        categorical_mask_train[class_idx[0], class_idx[1]+i] = 1

img_train = img_train.astype('float32')

# img_train /= 255

mask_train = mask_train.astype('float32')
# mask_train /= 255  # scale masks to [0, 1]

zero_idicies = np.where(mask_train_int == 0)
img_train = np.delete(img_train, zero_idicies[0][int(0.1*len(zero_idicies[0])):], axis=0)
categorical_mask_train = np.delete(categorical_mask_train, zero_idicies[0][int(0.1*len(zero_idicies[0])):], axis = 0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    img_train, categorical_mask_train, test_size=0.2, random_state=1
)

print('*'*30)
print('Creating and compiling model...')
print('*'*30)
# model = unet_categorical(use_focal = False)
model = functional_cnn()
#model = HybridCNN()
#%% Show CNN properties
model.summary()

#%% Fit Net
print('*'*30)
print('Fitting model...')
print('*'*30)

epochs = 1000
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)
model_checkpoint = ModelCheckpoint(f'weights_bceloss_{epochs}epochs.keras', monitor='val_loss', save_best_only=True)

# Define the learning rate scheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',  # Metric to monitor
    factor=0.8,          # Reduce the learning rate by this factor
    patience=20,          # Number of epochs with no improvement before reducing
    min_lr=1e-8          # Minimum learning rate
)
# model.load_weights('weights_bceloss_301epochs.keras')
history =  model.fit(X_train, y_train, batch_size=256, epochs=epochs, verbose=1, shuffle=True, validation_data=(X_test, y_test),
          callbacks=[model_checkpoint, early_stopping, reduce_lr])



#%% Test CNN - if GT is a matrix (Unet)
# plt.figure();
# plt.imshow(categorical_mask_train[104, :, :, 1:4])


# prediction = model.predict(img_train[104:104+1, :, :, :])
# plt.figure();
# plt.imshow(prediction[0, :, :, 1:4])



#%% replace loss
model = unet_categorical(use_focal = True)
model.load_weights('weights_bceloss_300epochs.keras')
history =  model.fit(X_train, y_train, batch_size=512, epochs=epochs, \
                     verbose=1, shuffle=True, validation_data=(X_test, y_test),
                     callbacks=[model_checkpoint, early_stopping, reduce_lr])

#%% Plot Training Loss
plt.figure()
plt.plot(history.history['loss'], linewidth=1, color='r')                   
plt.plot(history.history['val_loss'], linewidth=1, color='b')
plt.title('Model train vs Validation Loss', fontweight="bold")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.xticks()
plt.yticks()
plt.show()

#%% Plot confution matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd
import seaborn as sns
pred = np.argmax(model.predict(X_test), axis=1)

plt.figure(figsize = (10,7))
mat = confusion_matrix(np.add(pred, 1), np.add(np.argmax(y_test, 1), 1))
df_cm = pd.DataFrame(mat)
sns.heatmap(df_cm, annot=True, fmt='d')
plt.show()

# Classification Report
print(classification_report(pred, np.argmax(y_test, 1)))
