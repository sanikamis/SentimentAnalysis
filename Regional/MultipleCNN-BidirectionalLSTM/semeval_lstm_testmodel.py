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

np.random.seed(1234)

# Global hyper-parameters
sequence_length = 2001  # 8 regions * 10 words * 25 dimensional vectors + sentiment
batch_size = 1000
region_count = 8
word_count = 10
dimensions = 25
row_count = 12284

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ft", "--testFile", type=str, default="D:/data/SemEvalTest2017_rep_pt.csv", help="Csv file for Testing")
ap.add_argument("-fo", "--outputFile", type=str, default="semResults.csv", help="Csv file for Output")
ap.add_argument("-fm", "--modelFile", type=str, default="modelConv.json", help="Json file for CNN Models")
ap.add_argument("-fml", "--lstm_modelFile", type=str, default="modelLSTM.json", help="Json file for LSTM Model")
ap.add_argument("-fw", "--weightFile", type=str, default="model.h5", help="h5 file for Model Weights")
ap.add_argument("-fwl", "--lstm_weightFile", type=str, default="weightLSTM.h5", help="h5 file for Model Weights")
ap.add_argument("-col", "--column", type=int, default=0, help="column od the csv file")
args = vars(ap.parse_args())

column = args["column"]
testFile = args["testFile"]
outputFile = args["outputFile"]
modelFile = args["modelFile"]
lstm_modelFile = args["lstm_modelFile"]
weightFile = args["weightFile"]
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

    listX = list();
    for i in range(0,region_count):
        X = data.iloc[:,(i*word_count*dimensions):((i+1)*word_count*dimensions)].values
        X = reshape(X,(row_count,word_count,dimensions)) 
        
        print("X"+str(i)+".shape: ")
        print(X.shape)
        listX.append(X)

    Y = pd.get_dummies(data['sentiment']).values
    print("Y.shape: ")
    print(Y.shape)
    return listX, Y

def load_models():
    listModels = list();
    for i in range(0,region_count):    
        # load json and create model
        json_file = open(modelFile, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("weightConv"+str(i)+".h5")
        listModels.append(model)
    print("Loaded model and weights from disk")
    return listModels

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
        X_testList, y_test = get_split_prep_data()
    else:
        X_testList, y_test = data

    print ('\nData Loaded. Compiling...\n')

    models = load_models()
    
    print(models[0].summary())
    trueCount = 0
    totalCount = 0
    print ("(Actual, Predicted):")

    PP,PN,PU,NN,NP,NU,UU,UP,UN = 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    layer_output_list = list();
    for i in range(0,region_count):
        result = models[i].predict(X_testList[i],batch_size=batch_size,verbose = 2)[0]
        get_layer_output = K.function([models[i].layers[0].input, K.learning_phase()],
                                                      [models[i].layers[3].output])
        layer_output = get_layer_output([X_testList[i], 0])[0]
        print(layer_output.shape)
        print(layer_output[0])
        layer_output_list.append(layer_output)
       
    X_test = []
    #Create row_count training sequence from each regions ith item
    for i in range(0,row_count):
        region_X_list = list();
        for j in range(0,region_count):
            region_X_list.append(layer_output_list[j][i])
        X_test.append(region_X_list)
    X_test = np.array(X_test)  # shape (samples, sequence_length)   

    lstm_model = load_LSTM_model()
  
    for x in range(len(X_test)):
    
        result = lstm_model.predict(X_test[x].reshape(1,X_test.shape[1],X_test.shape[2]),batch_size=batch_size,verbose = 2)[0]
        #print(np.argmax(y_test[x]), np.argmax(result))
        print (x,np.argmax(y_test[x]),np.argmax(result),result)

        if np.argmax(result) == np.argmax(y_test[x]):
            if np.argmax(y_test[x]) == 0:
                UU += 1
            else: 
                if np.argmax(y_test[x]) == 1:
                    PP += 1
                else:
                    NN += 1 
        else:
            if np.argmax(y_test[x]) == 1:
                if np.argmax(result) == 0:
                    UP += 1
                else:
                    NP += 1
            else:
                if np.argmax(y_test[x]) == 2:
                    if np.argmax(result) == 0:
                        UN += 1
                    else:
                        PN += 1 
                else:
                    if np.argmax(result) == 1:
                        PU += 1
                    else:
                        NU += 1     
    print ("(PP,PN,PU,NN,NP,NU,UU,UP,UN)")
    print (PP,PN,PU,NN,NP,NU,UU,UP,UN)
    print ("acc: ", str(float(PP+NN+UU) / len(X_test)))
    
    #model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    #score,acc = model.evaluate(X_test, y_test, verbose = 2, batch_size = batch_size)
    #print("score: %.2f" % (score))
    #print("acc: %.2f" % (acc))
    
    return models, y_test


run_network()
