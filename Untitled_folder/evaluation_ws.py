import tensorflow as tf
import pandas as pd
import numpy as np
import json
import time

def Evaluation(input):
    start_time = time.time()
    df1 = input.loc[(input['CONFIDENCE']>= threshold[1]),['CONFIDENCE','CHARACTER','tagged_label']]
    df2 = df1.loc[(df1['CHARACTER'] == df1['tagged_label']),['CONFIDENCE','CHARACTER','tagged_label']]
    precision = df2['CHARACTER'].count()/df1['CHARACTER'].count()
    recall = df1['CONFIDENCE'].count()/input['CONFIDENCE'].count()
    print("time_1: ", time.time() - start_time )
    return precision,recall
    
threshold = [0.9,0.8,0.6,0.7]
input = pd.read_csv('/home/mohan/Downloads/results-20190208-120617.csv')
precision,recall = Evaluation(input)
def pre_rec(data,precision,recall):
    for i in range(len(threshold)):
        start_time = time.time()
        d = {}
        d['PRECISION'] = precision
        d['RECALL'] = recall
        d['THRESHOLD'] = threshold[i]
        print("------------------>",d)
        with open('/home/mohan/precision_recall.json', 'w') as fp:
                json.dump(d, fp)
        print("time_2: ", time.time() - start_time )

pre_rec(input,precision,recall)

