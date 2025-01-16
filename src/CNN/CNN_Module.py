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


#%% Define Net
def create_entry_heads(input_tensor, input_size):
    """
    Creates multiple entry heads that process specific input channels,
    apply trainable weights, and merge the results to reconstruct the original shape.
    
    Parameters:
        input_tensor (Tensor): Input tensor of shape (n, 128, 128, 12).
    
    Returns:
        Tensor: Processed tensor of shape (n, 128, 128, 12).
    """
    channel_splits = input_size[-1]  # Total number of channels
    heads = []
    nor_outs = []

    for i in range(channel_splits//2):
        # Extract the i-th channel
        single_channel = Lambda(lambda x: x[..., 2*i:2*i+1])(input_tensor)

        # Shallow CNN layer to process the channel
        # processed_channel = Conv2D(
        #     8, 3, activation='relu', padding='same', kernel_initializer='he_normal'
        # )(single_channel)

        # Apply a trainable weight (scalar) to the channel
        weight = tf.Variable(1.0, trainable=True, name=f"weight_{i}")
        weighted_channel = Lambda(lambda x: x * weight)(single_channel)

        # Append to the heads list
        heads.append(weighted_channel)

        # Merge all processed channels and concatenate with the original input
        merged_output = Add()(heads)
        norm_inputs = BatchNormalization()(merged_output)
        nor_outs.append(norm_inputs)
    
    concat_output = Concatenate()(nor_outs)
    # Ensure the output shape matches the input
    # merged_output = Conv2D(
    #     6, 1, activation=None, padding='same', kernel_initializer='he_normal'
    # )(merged_output)

    return concat_output

def unet_categorical(input_size,  n_classes , use_focal = False, \
                     learning_rate = 1e-3, add_extra_dropout = True,\
                         use_weighted=False, class_weights=None):
    
    # TODO : Add net architecture in the config. will be done in the fitire

    inputs = Input(input_size)
    
    # Apply the entry heads
    processed_inputs = create_entry_heads(inputs, input_size)
    
    # norm_inputs = BatchNormalization()(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
                                               kernel_initializer = 'he_normal'
                                               )(processed_inputs)
    
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
    
    
    if add_extra_dropout:
        drop9 = Dropout(0.9)(conv9)

        conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                                                   kernel_initializer = 'he_normal'
                                                   )(drop9) # (conv9) #(drop9)
        drop10 = Dropout(0.9)(conv10)
        
        conv11 = Conv2D(2, 3, activation = 'relu', padding = 'same',
                                                  kernel_initializer = 'he_normal'
                                                  )(drop10) #(conv10) #(drop10)

    else:
        conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                                                   kernel_initializer = 'he_normal'
                                                   )(conv9) # (conv9) #(drop9)

        conv11 = Conv2D(2, 3, activation = 'relu', padding = 'same',
                                                  kernel_initializer = 'he_normal'
                                                  )(conv10) #(conv10) #(drop10)
        
   
    conv12 = Conv2D(n_classes, 1, activation='softmax')(conv11)
    # conv10 = Conv2D(1, 1)(conv9)

    model = Model(inputs = inputs, outputs = conv12)

#    model.compile(optimizer = Adam(learning_rate = 1e-3), loss = 'mse', \
#                  metrics=['mean_absolute_error'])
    if use_focal:

        # Compile with the custom Focal Loss
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.CategoricalFocalCrossentropy(
                          alpha=0.25,
                          gamma=2.0,
                          from_logits=False,
                          label_smoothing=0.0,
                          axis=-1,
                          reduction='sum_over_batch_size',
                          name='categorical_focal_crossentropy'),
                      metrics=['accuracy'])

        
        # Compile with tfa loss function (this block will be removed in the future),
        # saved for debugging
        
        # model.compile(optimizer=Adam(learning_rate=learning_rate),
        #           loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0),
        #           metrics=['accuracy'])
        
    elif use_weighted:
        
        # Custom weighted loss
        def loss_fn(labels, predictions):
            return cl.weighted_categorical_crossentropy(labels, predictions, tf.constant(class_weights))
        
        # Custom weighted loss        
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss=loss_fn,
                      metrics=['accuracy'])
    else:
        # Default categorical cross-entropy
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    
    
    return model

#%% Run Preprocess

print('*'*30)
print('Loading and preprocessing train data...')
print('*'*30)
# TODO: sort out the get path for the data
# Get config


# Get the directory of the script being executed
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate to the project root and construct the config path
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
config_path = os.path.join(project_root, 'configs', 'apa_config_detroit.yaml')

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

# Add noise to the final result
# X_test = da.add_random_noise(X_test) # Add noise to the final result

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
def preprocess_data(X_trainAug, y_trainAug, aug_depth, num_augmented_samples, add_noise = False):
    
    aug_database_dense, aug_labels_dense = da.augment_dataset(X_trainAug, \
                                                  y_trainAug.astype(int), \
                                                      num_augmented_samples=int(num_augmented_samples),\
                                                          aug_depth=aug_depth, # was 4
                                                          add_noise = add_noise)
        
    aug_database = aug_database_dense #+ aug_database_sparse
    aug_labels = aug_labels_dense #+ aug_labels_sparse
    
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
    
    return X_trainFin, X_testAug, y_trainFin, y_testAug


#%% Data Preprocess - Phase 1

aug_depth_1 = config['net_preprocess']['phase_1']['aug_depth']
num_augmented_samples_1 = config['net_preprocess']['phase_1']['num_augmented_samples']
add_noise = config['cnn_model']['add_noise']


X_trainFin_dense, X_testAug_dense, y_trainFin_dense, y_testAug_dense = \
    preprocess_data(X_trainAug, y_trainAug, aug_depth=aug_depth_1,\
                    num_augmented_samples=num_augmented_samples_1, add_noise=add_noise)
        
print('*'*30)
print('Creating and compiling model...')
print('*'*30)

#%% Fit Net
print('*'*30)
print('Fitting model...')
print('*'*30)

epochs = config["training"]["epochs"]
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
model_checkpointIni = ModelCheckpoint(f'weights_bceloss_ini.keras', monitor='val_loss', save_best_only=True)
model_checkpoint = ModelCheckpoint(f'weights_bceloss_{epochs}epochs.keras', monitor='val_loss', save_best_only=True)

# Read ReduceLROnPlateau parameters
reduce_lr_config = config['training']['reduce_lr']
monitor = reduce_lr_config['monitor']
factor = reduce_lr_config['factor']
patience = reduce_lr_config['patience']
min_lr = reduce_lr_config['min_lr']

# Create the ReduceLROnPlateau callback
reduce_lr = ReduceLROnPlateau(
    monitor=monitor,
    factor=factor,
    patience=patience,
    min_lr=eval(min_lr)
)
#%% Define the model
# Access each phase explicitly
phase_1 = config['training_phases']['phase_1']
phase_2 = config['training_phases']['phase_2']
phase_3 = config['training_phases']['phase_3']
phase_4 = config['training_phases']['phase_4']

model1 = unet_categorical(input_size = input_size, n_classes = n_classes,\
                         use_focal=phase_1['use_focal'], learning_rate = eval(phase_1['learning_rate']))

model2 = unet_categorical(input_size = input_size, n_classes = n_classes,\
                         use_focal=phase_2['use_focal'], learning_rate = eval(phase_2['learning_rate']), \
                             use_weighted = phase_2['use_weighted'], class_weights = phase_2['class_weights'])

#%% Show CNN properties
model2.summary()

#%% Fit the net
history =  model1.fit(X_trainFin_dense, y_trainFin_dense, batch_size=phase_1['batch_size'], epochs=phase_1['epochs'], \
                     verbose=1, shuffle=True, validation_data=(X_testAug_dense, y_testAug_dense),
                     callbacks=[model_checkpointIni, early_stopping, reduce_lr])

# Load Weights from previous stage
model2.load_weights('weights_bceloss_ini.keras')

history =  model2.fit(X_trainFin_dense, y_trainFin_dense, batch_size=phase_2['batch_size'], epochs=phase_2['epochs'], \
                     verbose=1, shuffle=True, validation_data=(X_testAug_dense, y_testAug_dense),
                     callbacks=[model_checkpoint, early_stopping, reduce_lr])   

#%% Refine with lower learining rate
# Redifine the net with lower learning rate and load weights

aug_depth_2 = config['net_preprocess']['phase_2']['aug_depth']
num_augmented_samples_2 = config['net_preprocess']['phase_2']['num_augmented_samples']


X_trainFin, X_testAug, y_trainFin, y_testAug = \
    preprocess_data(X_trainAug, y_trainAug, aug_depth=aug_depth_2, num_augmented_samples=num_augmented_samples_2)


model = unet_categorical(input_size = input_size, n_classes = n_classes, \
                         learning_rate=eval(phase_3['learning_rate']), \
                             add_extra_dropout = phase_3['add_extra_dropout'])

model.load_weights('weights_bceloss_100epochs.keras')

# set checkpoints
model_checkpoint = ModelCheckpoint(f'weights_bceloss_{epochs}_refined.keras', monitor='val_accuracy', save_best_only=True)

# Fit net
history =  model1.fit(X_trainFin, y_trainFin, batch_size=phase_3['batch_size'],\
                      epochs=phase_3['epochs'], \
                     verbose=1, shuffle=True, validation_data=(X_test, categorical_mask_test),
                     callbacks=[model_checkpoint, early_stopping, reduce_lr])

#%% Replace loss function
# Redifine the net with new loss to better treat outliers and load pretrained weights
model = unet_categorical(input_size = input_size, n_classes = n_classes, \
                         use_focal=phase_4['use_focal'], learning_rate=eval(phase_4['learning_rate']), \
                         add_extra_dropout = phase_4['add_extra_dropout'])
model.load_weights(f'weights_bceloss_{epochs}_refined.keras')

# set checkpoints
model_checkpoint = ModelCheckpoint(f'weights_bceloss_{epochs}_refined2.keras', monitor='val_accuracy', save_best_only=True)

# Fit net
history =  model.fit(X_trainFin, y_trainFin, batch_size=phase_4['batch_size'], epochs=phase_4['epochs'], \
                     verbose=1, shuffle=True, validation_data=(X_test, categorical_mask_test),
                     callbacks=[model_checkpoint, early_stopping, reduce_lr])

    
#%% Try Another one

# model = unet_categorical(input_size = input_size, n_classes = n_classes, \
#                          use_focal=True, learning_rate=5e-4, \
#                          add_extra_dropout = False)
# model.load_weights(f'weights_bceloss_{epochs}_refined2.keras')

# # set checkpoints
# model_checkpoint = ModelCheckpoint(f'weights_bceloss_{epochs}_refined3.keras', monitor='val_accuracy', save_best_only=True)

# # Fit net
# history =  model.fit(X_trainFin, y_trainFin, batch_size=10, epochs=epochs, \
#                      verbose=1, shuffle=True, validation_data=(X_test, categorical_mask_test),
#                      callbacks=[model_checkpoint, early_stopping, reduce_lr])


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
pred = model.predict(X_test)

plt.figure()
plt.imshow(categorical_mask_test[130, :, :, 1:4])
plt.figure()
plt.imshow(pred[130, :, :, 1:4])
plt.figure()
plt.imshow(X_test[130, :, :, 5])


#%% Plot confution matrix
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
