import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras import backend as K
from keras.layers import MaxPooling1D, Conv1D, Flatten, Bidirectional
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from numpy import *
from pandas import read_csv

np.random.seed(1234)

# Global hyper-parameters
sequence_length = 1001  # 8 regions * 10 words * 25 dimensional vectors + sentiment
epochs_CNN = 300
epochs_LSTM = 300
batch_size = 100
region_count = 40
word_count = 25
dimensions = 1
row_count = 31007

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ft", "--trainFile", type=str, default="D:/data/SemEvalDev2017_rep_NR_pt.csv", help="Csv file for Training")
ap.add_argument("-fm", "--modelFile", type=str, default="modelConv.json", help="Json file for CNN Models")
ap.add_argument("-fml", "--lstm_modelFile", type=str, default="modelLSTM.json", help="Json file for LSTM Model")
ap.add_argument("-fw", "--weightFile", type=str, default="weight.h5", help="h5 file for Model Weights")
ap.add_argument("-fwl", "--lstm_weightFile", type=str, default="weightLSTM.h5", help="h5 file for Model Weights")
ap.add_argument("-col", "--column", type=int, default=0, help="column od the csv file")
args = vars(ap.parse_args())

column = args["column"]
trainFile = args["trainFile"]
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
    
    data = pd.read_csv(trainFile)
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

def build_CNN_models():
    listModels = list();
    for i in range(0,region_count):
        model = Sequential()
        model.add(Conv1D(filters=12, kernel_size=3, border_mode='valid', input_shape=(word_count,dimensions))) 
        """
        model.add(Conv1D(filters=10, kernel_size=3, border_mode='valid')) 
        model.add(Conv1D(filters=6, kernel_size=3, border_mode='valid')) 
        """
        model.add(MaxPooling1D(pool_size=3)) 
        model.add(Flatten())
        model.add(Dense(3,activation='sigmoid'))
        #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        start = time.time()
        model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
        print ("Compilation Time : ", time.time() - start)
        listModels.append(model)
    return listModels

def get_LSTM_data(models, X_trainList):
    
    # train data
    print ("Creating LSTM train data...")
	
    X_train = []
    layer_output_list = list();
    for i in range(0,region_count):
        print("Generating LSTM input from Convolution " + str(i) + "...")
        get_layer_output = K.function([models[i].layers[0].input, K.learning_phase()],
                                                  [models[i].layers[3].output])
        layer_output = get_layer_output([X_trainList[i], 1])[0]
        layer_output_list.append(layer_output)
    
    #Create row_count training sequence from each regions ith item
    for i in range(0,row_count):
        region_X_list = list();
        for j in range(0,region_count):
            region_X_list.append(layer_output_list[j][i])
        X_train.append(region_X_list)
    X_train = np.array(X_train)  # shape (samples, sequence_length)
            
    print ("LSTM Train data shape  : ", X_train.shape)
    #print (X_train[0])

    return X_train

def build_LSTM_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(3, return_sequences=True),
                                input_shape=(region_count, 3)))        
    model.add(Dropout(0.3))
    model.add(Flatten())
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
        X_trainList, y_train = get_split_prep_data()
    else:
        X_trainList, y_train = data

    print ('\nData Loaded. Compiling...\n')

    models = build_CNN_models()

    try:
        for i in range(0,region_count):
            print("Training Convolution " + str(i) + "...")
            models[i].fit(X_trainList[i], y_train, batch_size=batch_size, nb_epoch=epochs_CNN, validation_split=0.05)
            # serialize model to JSON
            model_json = models[i].to_json()
            with open(modelFile, "w") as json_file: json_file.write(model_json)
            # serialize weights to HDF5
            models[i].save_weights("weightConv"+str(i)+".h5")
            print("Saved model to disk")
    
    except KeyboardInterrupt:
        print("prediction exception")
        print ('Training duration (s) : ', time.time() - global_start_time)
        return models
    
    X_train = get_LSTM_data(models, X_trainList)
    lstm_model = build_LSTM_model();
    try:
        print("Training LSTM ...")
        lstm_model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs_LSTM, validation_split=0.05)
        
        # serialize model to JSON
        model_json = lstm_model.to_json()
        with open(lstm_modelFile, "w") as json_file: json_file.write(model_json)
        # serialize weights to HDF5
        lstm_model.save_weights(lstm_weightFile)
        print("Saved LSTM model to disk")
    except KeyboardInterrupt:
        print("prediction exception")
        print ('Training duration (s) : ', time.time() - global_start_time)
        return models
    
    print ('Training duration (s) : ', time.time() - global_start_time)    
    return models


run_network()
