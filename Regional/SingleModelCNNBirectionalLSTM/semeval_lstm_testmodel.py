import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import read_csv
import csv
import argparse
import time
from keras import backend as K
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import model_from_json
from numpy import arange, sin, pi, random

from keras.layers import Embedding
from keras.layers import MaxPooling1D, TimeDistributed, Conv1D, Flatten
from numpy import * 

from sklearn import svm
from sklearn.externals import joblib

np.random.seed(1234)

# Global hyper-parameters
sequence_length = 2000  # 40 words * 25 dimensional vectors + sentiment
batch_size = 100
word_count = 40
dimensions = 25
row_count = 12284 #2093 #12284

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ft", "--testFile", type=str, default="D:/data/SemEvalTest2017_rep_pt.csv", help="Csv file for Testing")
ap.add_argument("-fm", "--modelFile", type=str, default="modelCNN.json", help="Json file for CNN Models")
ap.add_argument("-fml", "--lstm_modelFile", type=str, default="modelLSTM.json", help="Json file for LSTM Model")
ap.add_argument("-fw", "--cnn_weightFile", type=str, default="weightCNN.h5", help="h5 file for Model Weights")
ap.add_argument("-fwl", "--lstm_weightFile", type=str, default="weightLSTM.h5", help="h5 file for Model Weights")
ap.add_argument("-col", "--column", type=int, default=0, help="column od the csv file")

args = vars(ap.parse_args())

column = args["column"]
testFile = args["testFile"]
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
    data = pd.read_csv(testFile)
    print("data.shape:")
    print(data.shape)     
    print(data[ data['sentiment'] == 0].size/sequence_length)
    print(data[ data['sentiment'] == 2].size/sequence_length)
    print(data[ data['sentiment'] == 1].size/sequence_length)
    X = data.iloc[:,:sequence_length].values
    X = reshape(X,(row_count,sequence_length,1)) 
    print("X.shape: ")
    print(X.shape)
    y_svmTest = data['sentiment'].values
    Y = pd.get_dummies(data['sentiment']).values
    print("Y.shape: ")
    print(Y.shape)
    return X, Y, y_svmTest

def load_LSTM_model():
  
    # load json and create model
    json_file = open(lstm_modelFile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(lstm_weightFile)
    print("Loaded GRU model from disk")
    return model


def run_network(model=None, data=None):
    global_start_time = time.time()

    if data is None:
        print ('Loading data... ')
        X_test, y_test, y_svmTest = get_split_prep_data()
    else:
        X_test, y_test, y_svmTest = data

    print ('\nData Loaded. Compiling...\n')
    
    trueCount = 0
    totalCount = 0
    print ("(Predicted, Actual):")

    PP,PN,PU,NN,NP,NU,UU,UP,UN = 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    cnnTrueCount = 0
    lstmTrueCount = 0
    svmTrueCount = 0
    unionTrueCount = 0
    
    
    lstm_model = load_LSTM_model()
    
    print('len(X_test)',len(X_test))
    for x in range(len(X_test)):
        actual = np.argmax(y_test[x])
        lstm_result = lstm_model.predict(X_test[x].reshape(1,X_test.shape[1],X_test.shape[2]),batch_size=batch_size,verbose = 2)[0]
        print ('actual', actual , 'predicted', lstm_result)
        lstm_result = np.argmax(lstm_result)
        if actual == lstm_result:
            lstmTrueCount+=1
        
    print ("(PP,PN,PU,NN,NP,NU,UU,UP,UN) PredictedActual")
    print (PP,PN,PU,NN,NP,NU,UU,UP,UN)
    print ("acc: ", lstmTrueCount / row_count)
    
    file = open("TestResults.txt","w") 
         
    file.write("acc: "+ str(lstmTrueCount / row_count) + "\n") 
         
    file.close() 
          
    
    return #models, y_test


run_network()
