#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:51:02 2018

@author: sameepshah
"""

import glob as glob
import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
#from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

#from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.core import Activation
#from keras.layers import Conv1D
#from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.layers import GlobalAveragePooling1D
from keras import optimizers
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from hyperopt import Trials, STATUS_OK, tpe
#from keras.datasets import imdb

from hyperas import optim
from hyperas.distributions import choice, uniform






def textprocessing():
    
    positive_Test = glob.glob("/home/sshah33/Opioid_Data/Test/Yes/*.txt")
    negative_Test = glob.glob("/home/sshah33/Opioid_Data/Test/No/*.txt")
    positive_Train = glob.glob("/home/sshah33/Opioid_Data/Train/Yes/*.txt")
    negative_Train = glob.glob("/home/sshah33/Opioid_Data/Train/No/*.txt")

    
    
    def read_files_in_one_dataframe_column(file_name_list):
        result_df_list = []
        for file_name in file_name_list:
            result_df = pd.read_csv(file_name, names=["Cuis"])
            result_df_list.append(result_df)
            
        sum_result_df = pd.concat(result_df_list)
        return 	sum_result_df
    
    
    df_Test_P = read_files_in_one_dataframe_column(positive_Test)
    #print(df_Test_P)
    df_Test_N = read_files_in_one_dataframe_column(negative_Test)
    df_Train_P = read_files_in_one_dataframe_column(positive_Train)
    df_Train_N = read_files_in_one_dataframe_column(negative_Train)


    positive_Test_D = pd.DataFrame(df_Test_P)
    positive_Test_D["label"] = 0

    negative_Test_D = pd.DataFrame(df_Test_N)
    negative_Test_D["label"] = 1

    Test = pd.concat(objs = [positive_Test_D, negative_Test_D],axis = 0,join = "outer")

    positive_Train_D = pd.DataFrame(df_Train_P)
    positive_Train_D["label"] = 0

    negative_Train_D = pd.DataFrame(df_Train_N)
    negative_Train_D["label"] = 1


    Train = pd.concat(objs = [positive_Train_D, negative_Train_D],axis = 0,join = "outer")
    #print(Test)

    Test.columns = ['Cuis','labels']
    Train.columns = ['Cuis','labels']
    Test_file = (Test['Cuis'])
    Test_label = Test['labels']
    Train_file = (Train['Cuis'])
    Train_label = Train['labels']
    
    #return Test_file,Test_label,Train_file,Train_label

    def maxlength(Train_file):
        CUIsLength = []
        for x in Train_file:
            y = x.split()
            z = len(y)
            #print(z)
            CUIsLength.append(z)
        highest = max(CUIsLength)
        #print(highest)
        return highest
    
    
    
    
    MAXLEN = maxlength(Train_file)

    X_file, X_dev, y_label, y_dev = train_test_split(Train_file, Train_label, test_size = 0.20, random_state=2018)
    tokenizer = Tokenizer(lower = False, oov_token='UNK')
    tokenizer.fit_on_texts(X_file)
    vocab_size = len(tokenizer.word_index) + 1
    #print(encoded)# determine the vocabulary size
    print('Vocabulary Size: %d' % vocab_size)
    encoded_train = tokenizer.texts_to_sequences(X_file)
    X_train = pad_sequences(encoded_train, maxlen = MAXLEN, padding='post')
    
    encoded_dev = tokenizer.texts_to_sequences(X_dev)
    X_dev = pad_sequences(encoded_dev, maxlen = MAXLEN, padding='post')
    print(X_train.shape)
    
    encoded_test = tokenizer.texts_to_sequences(Test_file)
    X_test = pad_sequences(encoded_test, maxlen = MAXLEN, padding='post')
    print(X_test.shape)
    
    return X_train, X_dev, X_test, y_label, y_dev, Test_label, vocab_size, MAXLEN
    


def create_model(X_train, X_dev, X_test, y_label, y_dev, Test_label, vocab_size,MAXLEN):
    
    #Maxlen = maxlength(Train_file)
    output_dim = {{choice([100,150,200])}}
    e = Embedding(vocab_size,output_dim,input_length=MAXLEN)
    
    model = Sequential()
    model.add(e)
    model.add(GlobalMaxPooling1D())
    
    model.add(Dropout({{uniform(0, 1)}}))
    
    model.add(Dense({{choice([100, 150, 50])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    
    # If we choose 'four', add an additional fourth layer
    
    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense(16))

        # We can also choose between complete sets of layers

        model.add({{choice([Dropout(0.5), Activation('relu')])}})
        #model.add(Activation('relu'))
    
    
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    
    Adam = optimizers.Adam(lr={{choice([1e-4, 1e-1, 1e-2])}}, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy',optimizer=Adam ,metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir="logs/run_a")
    result = model.fit(np.array(X_train), np.array(y_label), epochs=15, batch_size = {{choice([5,10,15])}}, validation_data = (np.array(X_dev), np.array(y_dev)), callbacks=[tensorboard])
    
    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc']) 
    print('Best validation acc of epoch:', validation_acc)
    
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model }

if __name__=="__main__":
    

    
    
    X_train, X_dev, X_test, y_label, y_dev, Test_label, vocab_size, MAXLEN = textprocessing()
   
    best_run, best_model = optim.minimize(model=create_model,
                                          data=textprocessing,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    
    
   
    
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Test_label))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

 
    
    