
import numpy as np
import os
import glob
from os.path import isfile
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Bidirectional, LSTM, Dropout, Activation, GRU
from keras.layers import Conv2D, concatenate, MaxPooling2D, Flatten, Embedding, Lambda

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop
from keras import regularizers

import librosa
import librosa.display
import matplotlib.pyplot as plt

dict_class ={'Not Parkinsons':0, 'Parkinsons':1}

reverse_map = {v: k for k, v in dict_class.items()}

nb_filters1=16 
nb_filters2=32 
nb_filters3=64
nb_filters4=64
nb_filters5=64
ksize = (4,4)
pool_size_1= (2,2) 
pool_size_2= (2,2)
pool_size_3 = (2,2)

dropout_prob = 0.20
dense_size1 = 216
lstm_count = 64
num_units = 128

BATCH_SIZE = 64
EPOCH_COUNT = 100
L2_regularization = 0.001

def conv_recurrent_model_build(model_input):
    print('Building model...')
    layer = model_input
    
    ### Convolutional blocks
    conv_1 = Conv2D(filters = nb_filters1, kernel_size = ksize, strides=1,
                      padding= 'valid', activation='relu', name='conv_1')(layer)
    pool_1 = MaxPooling2D(pool_size_1)(conv_1)

    conv_2 = Conv2D(filters = nb_filters2, kernel_size = ksize, strides=1,
                      padding= 'valid', activation='relu', name='conv_2')(pool_1)
    pool_2 = MaxPooling2D(pool_size_1)(conv_2)

    conv_3 = Conv2D(filters = nb_filters3, kernel_size = ksize, strides=1,
                      padding= 'valid', activation='relu', name='conv_3')(pool_2)
    pool_3 = MaxPooling2D(pool_size_1)(conv_3)
    
    
    conv_4 = Conv2D(filters = nb_filters4, kernel_size = ksize, strides=1,
                      padding= 'valid', activation='relu', name='conv_4')(pool_3)
    pool_4 = MaxPooling2D(pool_size_2)(conv_4)
    
    
    conv_5 = Conv2D(filters = nb_filters5, kernel_size = ksize, strides=1,
                      padding= 'valid', activation='relu', name='conv_5',input_shape=(288,432,4))(pool_4)
    pool_5 = MaxPooling2D(pool_size_2)(conv_5)

    flatten1 = Flatten()(pool_5)
    ### Recurrent Block
    
    # Pooling layer
    pool_lstm1 = MaxPooling2D(pool_size_3, name = 'pool_lstm')(layer)
    print(pool_lstm1)
    # Embedding layer

    squeezed = Lambda(lambda x: K.squeeze(x, axis= -1))(pool_lstm1)
    #flatten2 = K.squeeze(pool_lstm1, axis = -1)
#     dense1 = Dense(dense_size1)(flatten)
    
    # Bidirectional GRU
    lstm = Bidirectional(GRU(lstm_count))(squeezed)  #default merge mode is concat
    
    # Concat Output
    concat = concatenate([flatten1, lstm], axis=-1, name ='concat')
    
    ## Softmax Output
    output = Dense(num_classes, activation = 'softmax', name='preds')(concat)
    
    model_output = output
    model = Model(model_input, model_output)
    
#     opt = Adam(lr=0.001)
    opt = RMSprop(lr=0.0005)  # Optimizer
    model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )
    return model
npzfile = np.load('all_datavalidate.npz')

Xval = npzfile['arr_0']
Yval = npzfile['arr_1']

import random

test_indices = random.sample(range(len(Xval)), 10)
train_indices = []
for i in range(len(Xval)):
    if i not in test_indices:
        train_indices.append(i)

x_new_train = []
y_new_train = []
YNEWTRAIN = []
x_test = []
y_test = []
YTEST = []

for i in test_indices:
    x_test.append(Xval[i])
    y_test.append(Yval[i])
    

for i in train_indices:
    x_new_train.append(Xval[i])
    y_new_train.append(Yval[i])
    
    
Xval = np.asarray(x_new_train)
Yval = np.asarray(y_new_train)
Xtest = np.asarray(x_test)
Ytest = np.asarray(y_test)

Yval = keras.utils.np_utils.to_categorical(Yval)
Ytest = keras.utils.np_utils.to_categorical(Ytest)
batch_size = 32
num_classes = 2
n_features = Xval.shape[2] 
n_time = Xval.shape[1]

n_frequency = 216
n_frames = 128
    #reshape and expand dims for conv2d
#     x_train = x_train.reshape(-1, n_frequency, n_frames)

    
#     x_val = x_val.reshape(-1, n_frequency, n_frames)
Xval = np.expand_dims(Xval, axis = -1)
    
    
#input_shape = (n_frames, n_frequency,4)
input_shape = (128, 216,1)
model_input = Input(input_shape, name='input')
    
model = conv_recurrent_model_build(model_input)
model.load_weights('weights.bestv4_1.h5')
from sklearn.metrics import classification_report
y_true = np.argmax(Yval, axis = 1)
y_pred = model.predict(Xval)

y_pred = np.argmax(y_pred, axis=1)
labels = [0,1]
target_names = dict_class.keys()

