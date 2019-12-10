import json
import random
import pandas as pd
import copy
import keras
from keras.models import load_model
import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import glob
import sys
import concurrent.futures
import yaml
import csv
# reload(sys)
# sys.setdefaultencoding('utf8')


with open('/home/mohan/Face_pred.yaml', 'r') as fp:
    data = yaml.load(fp)
def predi(item):

    encoder_json_path = data['prediction']['encoder_json']
    weights_path = data['prediction']['weights_path']
    test = pd.read_csv(item)
    print(test)
    intial_shape = test.shape[0]
    print(item + '   initial_csv_shape',test.shape)
    # X_test = []
    # _ = [X_test.append(list(map(float,f.encode('ascii', 'ignore').decode('ascii').strip()[1:-1].split(',')))) for f in test['FEATURE_VECTOR']]
    #test = test.drop(['character'],axis = 1)
    """
    Should pass the 128 dimension vectors to X_test
    """
    X_test = test.values
    with open(encoder_json_path) as json_data:
        hypes = json.load(json_data)
    num_classes = len(hypes)
    model = load_model(weights_path)
    model.summary()
    model = Sequential()
    model.add(Dense(256, input_dim = 128, activation = 'tanh' ))
    model.add(Dense(512, activation = 'tanh'))
    model.add(Dense(128, activation = 'tanh'))
    model.add(Dropout(0.6))
    model.add(Dense(num_classes, activation='softmax'))
    #model = load_model(weights_path)
    model.summary()
    pred_classes = model.predict_classes(np.array(X_test))
    prob = model.predict(np.array(X_test))
    max_prob = np.max(prob, axis = 1)
    pred_class = []
    for p in pred_classes:
        for k,v in hypes.items():
            if p==v:
                pred_class.append(k)
    test['pred_class_label'] = pred_class
    test['probability'] = max_prob
    test = test[['pred_class_label','probability']]
    print(test)
    test.to_csv(data['prediction']['final_csv_path_to_save']+item.split('/')[-1].split('.csv')[0]+'_pred.csv',index = False)
    final_shape = test.shape[0]
    print(item +'   final_csv_shape',test.shape)
    if(intial_shape != final_shape):
        print('anamoly detected ...............................................................................', item )
    return None 


df = pd.read_csv(data['prediction']['embeddings_csv'])
df = df.drop(['character'], axis = 1)
df.to_csv('/home/mohan/dff.csv',index = False)
predi('/home/mohan/dff.csv')