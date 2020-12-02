import numpy as np
from tensorflow import keras
import mne
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier
from mne_features.feature_extraction import extract_features

from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K


def extract_features_data(data, selected_features, sfreq=256, funcs_params = None):
    data_T = np.transpose(data, axes=[0,2,1])
    #data_T = np.nan_to_num(data_T)
    extracted_features = extract_features(X = data_T, selected_funcs = {selected_features}, funcs_params = funcs_params,
                                         sfreq = sfreq, return_as_df=True, n_jobs=-1)
    
    return extracted_features


def filter_data(data, sfreq=256, l_freq=0.5, h_freq=35, verbose=False):
    """do something""" 
    data_t = np.transpose(data, axes=[0,2,1])
    filtered_t = mne.filter.filter_data(data = data_t, l_freq=l_freq, h_freq = h_freq, sfreq=sfreq, verbose=False)
    
    return np.transpose(filtered_t, axes=[0,2,1])


def onehotencode(all_labels):
    y = all_labels.copy()
    
    enc = LabelBinarizer()
    enc.fit(np.hstack(y.copy()))
    return enc


def simple_mlp(learning_rate = 0.10, input_shape=(65000,9)):
    
    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape = input_shape))
    
    model.add(keras.layers.Dense(50, activation = "relu"))
    
    model.add(keras.layers.Dense(30, activation = "relu"))

    model.add(keras.layers.Dense(10, activation = "relu"))
       
    model.add(keras.layers.Dense(1, activation="sigmoid"))
  
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics=["accuracy"])
    
    return model

def model1(learning_rate = 1., input_shape = (65000,9)):

    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape = input_shape))
    
    model.add(keras.layers.LSTM(10, return_sequences= False, input_shape = input_shape, batch_size = 50))
 
    model.add(keras.layers.Dense(20, activation="relu"))
    
    model.add(keras.layers.Dense(10, activation="relu"))

    model.add(keras.layers.Dense(1, activation="sigmoid"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics=["accuracy"])

    return model



def model2(learning_rate = 1., input_shape = (65000,9)):

    model = keras.models.Sequential()
    
    model.add(keras.layers.InputLayer(input_shape = input_shape))
    
    model.add(keras.layers.LSTM(40, return_sequences = True))
    
    model.add(keras.layers.Dropout(0.2))
    
    model.add(keras.layers.LSTM(30, return_sequences = False))
    
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=0.5)
    
    model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics=["accuracy"])
    
           
    return model
              
             

        
        
def model3(learning_rate = 0.00001, input_shape = (15360,9), activation = "relu", neurons = 20):

    model = keras.models.Sequential()
        
    model.add(keras.layers.Conv1D(filters = 30, kernel_size = 3, input_shape = input_shape))  

    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.MaxPooling1D())
    
    model.add(keras.layers.Conv1D(filters = 50, kernel_size = 3))
    
    model.add(keras.layers.MaxPooling1D())
    
    model.add(keras.layers.Dropout(0.1))
        
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(neurons, activation=activation))
      
    model.add(keras.layers.Dropout(0.1))
    
    model.add(keras.layers.Dense(2, activation="softmax"))
    
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])
           
    return model


def conv(learning_rate = 0.001, input_shape = (15360,9), activation = "relu", neurons = 50):

    model = keras.models.Sequential()
    
    model.add(keras.layers.Conv1D(filters = 20, kernel_size = 1, activation = activation, input_shape = input_shape))  
    
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.MaxPooling1D())
    
    model.add(keras.layers.Conv1D(filters = 40 , kernel_size = 3, activation = activation))
    
    model.add(keras.layers.MaxPooling1D())
    
    model.add(keras.layers.Dropout(0.2))
        
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(neurons, activation=activation))
    
    model.add(keras.layers.Dropout(0.2))
   
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics=["accuracy"])
           
    return model

def model4(learning_rate = 0.001, input_shape = (1,65000,9)):

    model = keras.models.Sequential()
    
    model.add(keras.layers.InputLayer(input_shape = input_shape))

    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
                                  
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics=["accuracy"])
           
    return model



