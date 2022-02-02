""" Import libraries """
import os
import keras
import pandas as pd
import numpy as np
import getopt
from decimal import Decimal
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def check_input(argv, dataset_input, dataset_query):

    try:
        opts, args = getopt.getopt(argv, "d:q:")
    except getopt.GetoptError:
        print("Error: python create_csv_file -d <dataset> -q <queryset>")
        sys.exit(1)

    if len(args) != 0:
        print("Error: python create_csv_file -d <dataset> -q <queryset>")
        sys.exit(1)

    for opt, arg in opts:
        if opt in ("-d"):
            dataset_input[0] = arg
        elif opt in ("-q"):
            dataset_query[0] = arg
        else:
            print("Error: python create_csv_file -d <dataset> -q <queryset>")
            sys.exit(1)

    if(dataset_input == '' or dataset_query is ''):
        print("Error: python create_csv_file -d <dataset> -q <queryset>")
        sys.exit(1)




def main(argv):

    filename_i = [0]
    filename_q = [0]
    
    check_input(argv, filename_i, filename_q)

    if os.path.isfile(filename_i[0]) == False or os.path.isfile(filename_q[0]) == False:
        print("Error: Enter a valid CSV file")
        sys.exit(1)

    make_enc(filename_i[0])
    make_enc(filename_q[0])



#same for query
def make_enc(filename):

    encoder = 'pre_trained_models/encoder_C.h5'

    df_q =pd.read_csv(filename, delimiter='\t', header=None)
    df_q = df_q.T

    to_be_encoded = []
    s=0
    # scaled
    encoded_set_scaled= []
    to_be_encoded_final = []
    training_set_scaled_cnn= []
    for i in range(0, df_q.shape[1]):
        encoded_set_scaled.append(MinMaxScaler(feature_range = (0, 1)))
        temp=df_q[i][1:]
        temp=np.array(temp)
        temp=temp.reshape(-1,1)
        temp = encoded_set_scaled[i].fit_transform(temp)
        training_set_scaled_cnn.append(temp)  
        s=0
        while(s<len(df_q[0])-1):
            tempo_x_train = temp[s:s+10]

            tempo_x_train=np.asarray(tempo_x_train).astype(np.float32)
            to_be_encoded.append(tempo_x_train)
            s=s+10
        to_be_encoded_final.append(np.asarray(to_be_encoded).astype(np.float32))
        to_be_encoded=[]

    encoder = keras.models.load_model(encoder)
    the_encoded = []
    for i in range(0,df_q.shape[1]):
        the_encoded.append(encoder.predict(to_be_encoded_final[i]))


    new_list_q=[]
    for i in range(0,df_q.shape[1]):
        the_temp_encoded = the_encoded[i].reshape(-1,1)
        the_temp_encoded = encoded_set_scaled[i].inverse_transform(the_temp_encoded)
        k = the_temp_encoded.reshape(the_encoded[0].shape[0]*the_encoded[0].shape[1])
        new_list_q.append(list(k))

    new_df_q = pd.DataFrame()

    for i in range(0,df_q.shape[1]):
        new_list_q[i].insert(0,df_q[i][0])
        new_df_q[i]=new_list_q[i]

    new_df_q = new_df_q.T

    filename=filename[:-4]
    filename= filename + "_enc.csv"
    k = new_df_q.to_csv(filename, sep='\t', encoding='utf-8',header=False,index=False)



if __name__ == "__main__":
    main(sys.argv[1:])