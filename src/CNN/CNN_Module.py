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
import tensorflow_addons as tfa
from keras import backend as keras
import src.CNN.DataAugmentation_Module as da
import matplotlib.pyplot as plt
import src.utils.io_utils as io_utils


#%% Define Net

def unet_categorical(input_size,  n_classes , use_focal = False, learning_rate = 1e-3):
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
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0),
                  metrics=['accuracy'])
    else:
        model.compile(Adam(learning_rate = learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    return model

#%% Run Preprocess

print('*'*30)
print('Loading and preprocessing train data...')
print('*'*30)
# TODO: sort out the get path for the data
# Get config
config = io_utils.read_yaml_config('configs/apa_config.yaml')
config=io_utils.fill_with_defaults(config['config'])


n_classes = config["cnn_model"]["num_classes"] # Need to read from config
input_size = eval(config["cnn_model"]["input_shape"])

file_hsp = h5py.File('Labeld_RoadsVenus.h5', 'r')
file_PCI = h5py.File('PCI_labels.h5', 'r')
img_train = file_hsp['cropped_segments'][:]
mask_train = file_PCI['cropped_segments'][:]
img_train = np.array(img_train)
mask_train = np.array(mask_train)

img_train = img_train.astype('float32')

## Normilize
img_train_mean = np.mean(img_train.flatten()[img_train.flatten() != 0])
img_train_std = np.std(img_train.flatten()[img_train.flatten() != 0])
non_zeros_idx = (img_train != 0)
img_train[non_zeros_idx] = (img_train[non_zeros_idx] - img_train_mean)/img_train_std

# img_train /= 255

mask_train = mask_train.astype('float32')
# mask_train /= 255  # scale masks to [0, 1]


from sklearn.model_selection import train_test_split
X_trainAug, X_test, y_trainAug, y_test = train_test_split(
    img_train, mask_train, test_size=0.2, random_state=1
)

categorical_mask_test = np.zeros(list(np.shape(y_test)[:-1]) + [n_classes])
mask_train_int = y_test.astype('int')
create_matrix_labels = True
for i in range(n_classes):
    class_idx = np.where(mask_train_int == i)
    if create_matrix_labels :
        categorical_mask_test[class_idx[0], class_idx[1], class_idx[2], class_idx[3]+i] = 1    
    else:
        categorical_mask_test[class_idx[0], class_idx[1]+i] = 1


#%% Augment the training
aug_database, aug_labels = da.augment_dataset(X_trainAug, \
                                              y_trainAug.astype(int), \
                                                  num_augmented_samples=int(2.5e3),\
                                                      aug_depth=2)
aug_database = np.asarray(aug_database)

aug_labels_categorical = np.zeros(list(np.shape(aug_labels)[:-1]) + [n_classes])
aug_labels_int = np.asarray(aug_labels).astype('int')
create_matrix_labels = True
for i in range(n_classes):
    class_idx = np.where(aug_labels_int == i)
    if create_matrix_labels :
        aug_labels_categorical[class_idx[0], class_idx[1], class_idx[2], class_idx[3]+i] = 1    
    else:
        aug_labels_categorical[class_idx[0], class_idx[1]+i] = 1

from sklearn.model_selection import train_test_split
X_trainFin, X_testAug, y_trainFin, y_testAug = train_test_split(
    aug_database, aug_labels_categorical, test_size=0.2, random_state=1
)

print('*'*30)
print('Creating and compiling model...')
print('*'*30)

#%% Define the model
model = unet_categorical(input_size = input_size, n_classes = n_classes)

#%% Show CNN properties
model.summary()

#%% Fit Net
print('*'*30)
print('Fitting model...')
print('*'*30)

epochs = config["training"]["epochs"]
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
model_checkpoint = ModelCheckpoint(f'weights_bceloss_{epochs}epochs.keras', monitor='val_loss', save_best_only=True)

# Define the learning rate scheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',  # Metric to monitor
    factor=0.1,          # Reduce the learning rate by this factor
    patience=10,          # Number of epochs with no improvement before reducing
    min_lr=1e-6          # Minimum learning rate
)


#%% Refine with lower learining rate
# Redifine the net with lower learning rate and load weights
model = unet_categorical(learning_rate=5e-4) # 
model.load_weights('weights_bceloss_100epochs.keras')

# set checkpoints
model_checkpoint = ModelCheckpoint(f'weights_bceloss_{epochs}_refined.keras', monitor='val_accuracy', save_best_only=True)

# Fit net
history =  model.fit(X_trainFin, y_trainFin, batch_size=8, epochs=epochs, \
                     verbose=1, shuffle=True, validation_data=(X_test, categorical_mask_test),
                     callbacks=[model_checkpoint, early_stopping, reduce_lr])

#%% Replace loss function
# Redifine the net with new loss to better treat outliers and load pretrained weights
model = unet_categorical(use_focal=True, learning_rate=1e-3)
model.load_weights(f'weights_bceloss_{epochs}_refined.keras')

# set checkpoints
model_checkpoint = ModelCheckpoint(f'weights_bceloss_{epochs}_refined2.keras', monitor='val_accuracy', save_best_only=True)

# Fit net
history =  model.fit(X_trainFin, y_trainFin, batch_size=8, epochs=epochs, \
                     verbose=1, shuffle=True, validation_data=(X_test, categorical_mask_test),
                     callbacks=[model_checkpoint, early_stopping, reduce_lr])

#%% Plot Training Loss
plt.figure()
plt.semilogy(history.history['loss'], linewidth=1, color='r')                   
plt.semilogy(history.history['val_loss'], linewidth=1, color='b')
plt.title('Model train vs Validation Loss', fontweight="bold")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.xticks()
plt.yticks()
plt.show()


#%% Plot some examples
plt.figure()
plt.imshow(categorical_mask_test[12, :, :, 1:4])
plt.figure()
plt.imshow(pred[12, :, :, 1:4])
plt.figure()
plt.imshow(X_test[12, :, :, 5])


#%% Plot confution matrix
pred = model.predict(X_test)
pred_labels_numeric = np.argmax(pred, axis=-1)
true_labels_numeric = np.argmax(categorical_mask_test, axis=-1)

# Flatten the masks
flat_true_labels = true_labels_numeric.ravel()
flat_pred_labels = pred_labels_numeric.ravel()

# Remove zeros
# flat_pred_labels = flat_pred_labels[flat_true_labels != 0]
# flat_true_labels = flat_true_labels[flat_true_labels != 0]
# Compute confusion matrix
cm = confusion_matrix(flat_true_labels, flat_pred_labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(flat_true_labels))
disp.plot(cmap='viridis')

# Classification report
print(classification_report(flat_true_labels, flat_pred_labels))
