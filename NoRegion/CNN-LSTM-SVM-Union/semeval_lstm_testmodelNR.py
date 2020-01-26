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
sequence_length = 1000  # 40 words * 25 dimensional vectors + sentiment
batch_size = 100
word_count = 40
dimensions = 25
row_count = 12284 #2093 #12284

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ft", "--testFile", type=str, default="D:/data/SemEvalTest2017_rep_NR_pt.csv", help="Csv file for Testing")
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

def load_CNN_model():   
    # load json and create model
    json_file = open(modelFile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(cnn_weightFile)
    print("Loaded model and weight from disk")
    return model

def load_LSTM_model():
  
    # load json and create model
    json_file = open(lstm_modelFile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(lstm_weightFile)
    print("Loaded LSTM model from disk")
    return model


def run_network(model=None, data=None):
    global_start_time = time.time()

    if data is None:
        print ('Loading data... ')
        X_test, y_test, y_svmTest = get_split_prep_data()
    else:
        X_test, y_test, y_svmTest = data

    print ('\nData Loaded. Compiling...\n')

    model = load_CNN_model()
    
    print(model.summary())
    trueCount = 0
    totalCount = 0
    print ("(Predicted, Actual):")

    PP,PN,PU,NN,NP,NU,UU,UP,UN = 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    cnnTrueCount = 0
    lstmTrueCount = 0
    svmTrueCount = 0
    unionTrueCount = 0
    
    cnn_model = load_CNN_model()
    lstm_model = load_LSTM_model()
    svm_model = joblib.load('modelSVM.sav')
    print('len(X_test)',len(X_test))
    for x in range(len(X_test)):
        actual = np.argmax(y_test[x])
        cnn_result = cnn_model.predict(X_test[x].reshape(1,X_test.shape[1],X_test.shape[2]),batch_size=batch_size,verbose = 2)[0]
        #cnn_result = np.argmax(cnn_result)
        lstm_result = lstm_model.predict(X_test[x].reshape(1,X_test.shape[1],X_test.shape[2]),batch_size=batch_size,verbose = 2)[0]
        #lstm_result = np.argmax(lstm_result)
        
        X_svm = np.concatenate((cnn_result,lstm_result), axis=0)
        X_svm = reshape(X_svm,(-1,6)) 
        svm_result = svm_model.predict(X_svm)[0]
        union_result = cnn_result + lstm_result + svm_result
        print ('actual', actual , 'cnn', cnn_result,'lstm', lstm_result, 'svm', svm_result, 'union', union_result)
        cnn_result = np.argmax(cnn_result)
        lstm_result = np.argmax(lstm_result)
        union_result = np.argmax(union_result)
        
        if actual == cnn_result:
            cnnTrueCount+=1
        if actual == lstm_result:
            lstmTrueCount+=1
        if actual == svm_result:
            svmTrueCount+=1
        if actual == union_result:
            unionTrueCount+=1
        
    print ("(PP,PN,PU,NN,NP,NU,UU,UP,UN) PredictedActual")
    print (PP,PN,PU,NN,NP,NU,UU,UP,UN)
    print ("CNN acc: ", cnnTrueCount / row_count)
    print ("LSTM acc: ", lstmTrueCount / row_count)
    print ("SVM acc: ", svmTrueCount / row_count)
    print ("UNION acc: ", unionTrueCount / row_count)

    file = open("TestResults.txt","w") 
         
    file.write("CNN acc: "+ str(cnnTrueCount / row_count) + "\n") 
    file.write("LSTM acc: "+ str(lstmTrueCount / row_count) + "\n") 
    file.write("SVM acc: "+ str(svmTrueCount / row_count) + "\n") 
    file.write("UNION acc: "+ str(unionTrueCount / row_count) + "\n") 
         
    file.close() 
    
        
    #model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    #score,acc = model.evaluate(X_test, y_test, verbose = 2, batch_size = batch_size)
    #print("score: %.2f" % (score))
    #print("acc: %.2f" % (acc))
    
    return #models, y_test


run_network()
