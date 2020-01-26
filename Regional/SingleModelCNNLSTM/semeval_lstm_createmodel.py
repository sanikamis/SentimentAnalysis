import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import argparse
import time
from keras import backend as K
from keras.layers import Embedding
from keras.layers import MaxPooling1D, TimeDistributed, Conv1D, Flatten, Bidirectional
from keras.layers.core import Dense, Activation, Dropout 
from keras.layers.recurrent import LSTM,GRU
from keras.models import Sequential
from keras.models import model_from_json
from numpy import arange, sin, pi, random
from numpy import * 
from sklearn import svm
from sklearn.externals import joblib
#import cv2

np.random.seed(1234)

# Global hyper-parameters
sequence_length = 2000  #  40 words * 25 dimensional vectors + sentiment
cnn_epochs = 1
lstm_epochs = 100
batch_size = 1000
row_count = 31007#19000 #2093 #
word_count = 40
dimensions = 25

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ft", "--trainFile", type=str, default="D:/data/SemEvalDev2017_rep_pt.csv", help="Csv file for Training")
ap.add_argument("-fm", "--modelFile", type=str, default="modelCNN.json", help="Json file for CNN Models")
ap.add_argument("-fml", "--lstm_modelFile", type=str, default="modelLSTM.json", help="Json file for LSTM Model")
ap.add_argument("-fw", "--cnn_weightFile", type=str, default="weightCNN.h5", help="h5 file for Model Weights")
ap.add_argument("-fwl", "--lstm_weightFile", type=str, default="weightLSTM.h5", help="h5 file for Model Weights")
ap.add_argument("-col", "--column", type=int, default=0, help="column od the csv file")
args = vars(ap.parse_args())

column = args["column"]
trainFile = args["trainFile"]
modelFile = args["modelFile"]
lstm_modelFile = args["lstm_modelFile"]
cnn_weightFile = args["cnn_weightFile"]
lstm_weightFile = args["lstm_weightFile"]

def data_from_csv(path):
    """get data from csv. use instead of gen_wave
    :return: csv column
    """
    dataframe = read_csv(path, usecols=[column], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    return dataset

# get the generic features of the frames calculated before in 'first_step_convert_video_to_csv_features'
# of train and test data from the corresponding files train.csv and test.csv
def get_split_prep_data():
    
    data = pd.read_csv(trainFile)
    print("data.shape:")
    print(data.shape)     
    print(data[ data['sentiment'] == 0].size/sequence_length)
    print(data[ data['sentiment'] == 2].size/sequence_length)
    print(data[ data['sentiment'] == 1].size/sequence_length)

    X = data.iloc[:,:sequence_length].values
    #X = reshape(X,(row_count,word_count,dimensions)) 
    X = reshape(X,(row_count,sequence_length,1)) 
    print("X.shape: ")
    print(X.shape)
    Y = pd.get_dummies(data['sentiment']).values
    print("Y.shape: ")
    print(Y.shape)
    return X, Y

def build_LSTM_model():
    model = Sequential()
    model.add(Conv1D(filters=12, kernel_size=50, border_mode='valid', input_shape=(sequence_length,1))) 
    model.add(MaxPooling1D(pool_size=3))
    model.add(Conv1D(filters=12, kernel_size=50, border_mode='valid')) 
    model.add(MaxPooling1D(pool_size=3))
    model.add(Conv1D(filters=12, kernel_size=50, border_mode='valid')) 
    model.add(MaxPooling1D(pool_size=3))
        
    
    #model.add(Bidirectional(LSTM(3, return_sequences=True),
    #                        input_shape=(sequence_length, 1)))
    
    model.add(LSTM(
                #input_length=sequence_length,
                input_dim=1,
                output_dim=3,
                return_sequences=False))
    #keras.layers.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False)
	#keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)

    model.add(Dropout(0.2))
    #model.add(Flatten())
    model.add(Dense(3,activation='sigmoid'))
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    print ("Compilation Time : ", time.time() - start)
    return model


def run_network(model=None, data=None):    
    
    global_start_time = time.time()

    if data is None:
        print ('Loading data... ')
        X_train, y_train = get_split_prep_data()
    else:
        X_train, y_train = data

    print ('\nData Loaded. Compiling...\n')

       
    lstm_model = build_LSTM_model();
    #X_train = reshape(X_train,(row_count,sequence_length,1))
    try:
        print("Training  ...")
        lstm_model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=lstm_epochs, validation_split=0.05)
        
        # serialize model to JSON
        model_json = lstm_model.to_json()
        with open(lstm_modelFile, "w") as json_file: json_file.write(model_json)
        # serialize weights to HDF5
        lstm_model.save_weights(lstm_weightFile)
        print("Saved model to disk")
    except KeyboardInterrupt:
        print("prediction exception")
        print ('Training duration (s) : ', time.time() - global_start_time)
        return model

         
    print ('Training duration (s) : ', time.time() - global_start_time)    
    
    
    
    return model


run_network()
