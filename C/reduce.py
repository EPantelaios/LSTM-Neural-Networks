""" Import libraries """
import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
import getopt
from decimal import Decimal
from keras.models import Model
from keras.layers import LSTM
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
import os
import random
import tensorflow.random

os.environ['PYTHONHASHSEED'] = str(1)
random.seed(1)
tensorflow.random.set_seed(1)
np.random.seed(1)

import sys
pd.options.display.max_colwidth = None



def plot_examples(stock_input, stock_decoded):
    n = 10  
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(list(np.arange(0, 1, 200))):
        # display original
        ax = plt.subplot(2, n, i + 1)
        if i == 0:
            ax.set_ylabel("Input", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_input[idx])
        ax.get_xaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        if i == 0:
            ax.set_ylabel("Output", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_decoded[idx])
        ax.get_xaxis().set_visible(False)



def check_input(argv, dataset):

    try:
        opts, args = getopt.getopt(argv, "d:")
    except getopt.GetoptError:
        print("Error: python reduce -d <dataset>")
        sys.exit(1)

    if len(args) != 0:
        print("Error: python reduce -d <dataset>")
        sys.exit(1)

    for opt, arg in opts:
        if opt in ("-d"):
            dataset[0] = arg
        else:
            print("Error: python reduce -d <dataset>")
            sys.exit(1)

    if(dataset == ''):
        print("Error: python reduce -d <dataset>")
        sys.exit(1)



def main(argv):

    dataset = [0]
    check_input(argv, dataset)
    print(dataset[0])

    if os.path.isfile(dataset[0]) == False:
        print("Error: Enter a valid CSV file")
        sys.exit(1)
    
    df=pd.read_csv(dataset[0], delimiter='\t', header=None)
    df = df.T
    print(f'Number of rows and columns: {df.shape}')
    df.head(5)

    #conv
    window_length=10
    epochs= 50

    x_train_cnn_list = []
    x_test_cnn_list = []
    s=1

    #successfully scaled after dayz xD
    sc_cnn = []
    training_set_scaled_cnn = []

    for i in range (0,340):
        sc_cnn.append(MinMaxScaler(feature_range = (0, 1)))
        temp=df[i][1:]
        temp=np.array(temp)
        temp=temp.reshape(-1,1)
        temp = sc_cnn[i].fit_transform(temp)
        training_set_scaled_cnn.append(temp)  
        s=0
        while(s<len(df[0])-1):
            tempo_x_train = temp[s:s+10]
            tempo_x_train=np.asarray(tempo_x_train).astype(np.float32)
            x_train_cnn_list.append(tempo_x_train)
            s=s+10

    x_train_final = np.asarray(x_train_cnn_list).astype(np.float32)

    #test
    s=1
    s=0
    x_test_final_cnn_list = []
    x_test_final_cnn_list_temp = []
    for i in range (340,359):

        sc_cnn.append(MinMaxScaler(feature_range = (0, 1)))
        temp=df[i][1:]
        temp=np.array(temp)
        temp=temp.reshape(-1,1)
        temp = sc_cnn[i].fit_transform(temp)
        training_set_scaled_cnn.append(temp) 
        s=0 
        while(s<len(df[i])-1):


            tempo_x_test = temp[s:s+10]
            tempo_x_test=np.asarray(tempo_x_test).astype(np.float32)
            x_test_cnn_list.append(tempo_x_test)
            x_test_final_cnn_list_temp.append(tempo_x_test)
            s=s+10
        x_test_final_cnn_list.append(list(x_test_final_cnn_list_temp))
        x_test_final_cnn_list_temp = []

    x_test_final = np.asarray(x_test_cnn_list).astype(np.float32)

    for i in range(0,len(x_test_final_cnn_list)):
        x_test_final_cnn_list[i]=np.asarray(x_test_final_cnn_list[i]).astype(np.float32)

    input_window = Input(shape=(window_length,1))
    x = Conv1D(16, 3, activation="relu", padding="same")(input_window) # 10 dims
    x = MaxPooling1D(2, padding="same")(x) # 5 dims
    x = Conv1D(1, 3, activation="relu", padding="same")(x) # 5 dims
    encoded = MaxPooling1D(2, padding="same")(x) # 3 dims

    encoder = Model(input_window, encoded)

    # 3 dimensions in the encoded layer
    x = Conv1D(1, 3, activation="relu", padding="same")(encoded) # 3 dims
    x = UpSampling1D(2)(x) # 6 dims
    x = Conv1D(16, 2, activation='relu')(x) # 5 dims
    x = UpSampling1D(2)(x) # 10 dims
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x) # 10 dims
    autoencoder = Model(input_window, decoded)
    autoencoder.summary()

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    history = autoencoder.fit(x_train_final, x_train_final,
                    epochs=epochs, batch_size=256, shuffle=True)

    encoder.save("encoder_C_question.h5")

    for i in range(0,len(x_test_final_cnn_list)):
        decoded_stocks = autoencoder.predict(x_test_final_cnn_list[i])
        plot_examples(x_test_final_cnn_list[i],decoded_stocks)



if __name__ == "__main__":
    main(sys.argv[1:])