""" Import libraries """
import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
import getopt
from decimal import Decimal
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import os
import random
import sys

#Configure some warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
pd.options.mode.chained_assignment = None 

def check_input(argv, dataset, n_predict, threshold):

    try:
        opts, args = getopt.getopt(argv, "d:n:m:")
    except getopt.GetoptError:
        print("Error: python predict -d <dataset> -n <number of time series selected> -m <error value as double>")
        sys.exit(1)

    print(args)
    if len(args) != 0:
        print("Error: python predict -d <dataset> -n <number of time series selected> -m <error value as double>")
        sys.exit(1)

    for opt, arg in opts:
        print(opt)
        if opt in ("-d"):
            dataset[0] = arg
        elif opt in ("-n"):
            if arg.isdigit() == True:
                n_predict[0] = int(arg)
        elif opt in ("-m"):
            if arg.replace('.','',1).isdigit() == True:
                threshold[0] = Decimal(arg) #double variable type
        else:
            print("Error: python predict -d <dataset> -n <number of time series selected> -m <error value as double>")
            sys.exit(1)

    if(dataset == '' or n_predict is None):
        print("Error: python predict -d <dataset> -n <number of time series selected> -m <error value as double>")
        sys.exit(1)

    print(f"Dataset is {dataset[0]}")
    print(F"Number for predict is {n_predict[0]}")
    print(F"Threshold is {threshold[0]}")



def main(argv):

    dataset = [0]
    n_predict = [0]
    threshold = [0]

    check_input(argv, dataset, n_predict, threshold)
   
    if os.path.isfile(dataset[0]) == False:
        print("Error: Enter a valid CSV file")
        sys.exit(1)

    """Parsing the input dataset"""
    df=pd.read_csv(dataset[0], delimiter='\t', header=None)
    df = df.T
    rows_df = df.shape[1]

    """**Make new column with date**"""

    max_column = len(df.index)
    new_col = [x for x in range(0, max_column)]
    df.insert(loc=0, column='Date', value=new_col)


    """**Choose n random timeseries for forecast**"""

    anomalies_list =  []
    anomalies_list = random.sample(range(0, rows_df), n_predict[0])

    print(anomalies_list)

    """**Number of timeseries for training**"""

    n_training = rows_df

    """**Preprocessing dataset**"""

    len_dataframe = len(df.index)
    first_80_percent =  math.floor(len_dataframe * 80/100)
    last_20_percent = len_dataframe - first_80_percent

    training_set_list = []
    testing_set_list = []
    for i in range (1, n_training+1):
        temp_training_set = []
        temp_testing_set = []
        temp_training_set = df.iloc[1:first_80_percent, i].values #to idio me katw alla ana grammi
        temp_testing_set = df.iloc[first_80_percent:, i].values
        training_set_list.append(list(temp_training_set)) #etsi exoume kathe metoxi ksexwrista!
        testing_set_list.append(list(temp_testing_set))

    """**Concatenate training set and scaling**"""

    TIME_STEPS=30
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    scaler = []

    for k in range(0, n_training):

        scaler.append(StandardScaler())

        temp_train=np.array(training_set_list[k])
        temp_train=temp_train.reshape(-1,1)
        scaler[k] = scaler[k].fit(temp_train)
        training_set_list[k] = scaler[k].transform(temp_train)

        def create_sequences(X, y, time_steps=TIME_STEPS):
            Xs, ys = [], []
            for i in range(0, len(X)-time_steps):
                Xs.append(X[i:(i+time_steps)])
                ys.append(y[i+time_steps])
            
            return np.array(Xs), np.array(ys)

        tmp_x1, tmp_y1 = create_sequences(training_set_list[k], training_set_list[k])

        X_train.append(tmp_x1)
        y_train.append(tmp_y1)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = X_train.reshape(n_training*2889, TIME_STEPS, 1)
    y_train = y_train.reshape(n_training*2889)

    """**Construct the model**"""

    flag_saved_model = True;

    if flag_saved_model == False:
        model = Sequential()
        model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(rate=0.2))
        model.add(RepeatVector(X_train.shape[1]))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(rate=0.2))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(rate=0.2))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(rate=0.2))

        model.add(TimeDistributed(Dense(X_train.shape[2])))
        model.compile(optimizer='adam', loss='mae')
        model.summary()

    """**Fit model or load saved model**"""

    if flag_saved_model == False:
        
        keras.backend.clear_session()
        history = model.fit(X_train, y_train, epochs=40, batch_size=64, validation_split=0.2, shuffle=False)
        name = 'my_model_B.h5'
        print(name)
        model.save(name)
        
    else:
        # Recreate the exact same model, including its weights and the optimizer
        name = 'pre_trained_models/model_B.h5'
        model = tf.keras.models.load_model(name)

    """**Plot training loss and validation loss**"""

    if flag_saved_model == False:
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.legend();

    """**Processing test set and detecting anomalies**"""

    for k in range(0, n_predict[0]):
        
        #Scale and calculate Test set
        test_set = df.iloc[first_80_percent:, k+1].values
        test_set=np.array(test_set)
        test_set=test_set.reshape(-1,1)
        test_set = scaler[k].transform(test_set)
        X_test, y_test = create_sequences(test_set, test_set)

        
        #Evaluate the model
        print(model.evaluate(X_test, y_test))

        #Predict and calculate test loss
        X_test_pred = model.predict(X_test, verbose=0)
        test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)

        #Create a new dataframe with the required information
        test_set_date = df.loc[first_80_percent:, 'Date']
        test_score_df = pd.DataFrame(test_set_date[TIME_STEPS:])
        test_score_df['loss'] = test_mae_loss
        test_score_df['threshold'] = threshold[0]
        test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
        test_score_df['Close'] = test_set[TIME_STEPS:]

        #Plot test loss
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_score_df['Date'], y=test_score_df['loss'], name='Test loss'))
        fig.add_trace(go.Scatter(x=test_score_df['Date'], y=test_score_df['threshold'], name='Threshold'))
        fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
        fig.show()

        #Assign anomalies
        anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
        print(anomalies.shape)


        #Inverse transform test dataframe
        [tmp_test_score] = scaler[k].inverse_transform([np.array(test_score_df['Close'])])
        test_score_df['Close'] = tmp_test_score
        
        if len(anomalies.Close.value_counts()) > 0:
            [tmp_anomalies] = scaler[k].inverse_transform([np.array(anomalies['Close'])])
            anomalies.loc[:, 'Close'] = tmp_anomalies
        
        #Plot detected anomalies
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_score_df['Date'], y=test_score_df['Close'] , name='Close price'))
        fig.add_trace(go.Scatter(x=anomalies['Date'], y=anomalies['Close'], mode='markers', name='Anomaly'))
        fig.update_layout(showlegend=True, title='Detected anomalies')
        fig.show()





if __name__ == "__main__":
    main(sys.argv[1:])