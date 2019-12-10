import pandas as pd
import numpy as np
import json
import os
import scipy
from scipy import spatial

df = pd.read_csv('/home/mohan/Documents/pyannotefacetracking/20181025_1_183003.000_190008.845_cluster_mapper.csv')

df1 = pd.read_json('/home/mohan/Documents/pyannotefacetracking/20181025_1_183003.000_190008.845_embeddings.json')

# df1.drop(['BQTIME','CHARACTER','CONFIDENCE','DATE','ROI','TIMESTAMP'],axis = 1,inplace = True)

df3 = df1.join(pd.DataFrame(df1['FEATURE_VECTOR'].values.tolist()))

df3.to_csv('df.csv', index = False)

mean = df3.groupby(['FACE_ID','VIDEO_ID']).mean()

df5 = pd.merge(mean,df, on = 'FACE_ID', how = 'left')

# df5 = pd.concat([df4,df],axis=1,ignore_index= True,sort= False)

# df1.drop(['FEATURE_VECTOR'],axis= 1,inplace = True)
# df1.to_csv('gh.csv', index = False)

centroid = df5.groupby(['VIDEO_ID','FACE_ID','LEVEL_1_CLUSTERID']).mean()

arr = spatial.distance.cdist(mean.iloc[:,3:], centroid.iloc[:,3:], metric='euclidean')
arr = np.max(arr,axis=0)
centroid['eucl'] = arr

centroid.sort_values(by=['LEVEL_1_CLUSTERID','eucl'],ascending = False)

for i in range(centroid['LEVEL_1_CLUSTERID']):
    arr = 

# grouped['FACE_ID','VIDEO_ID'].agg(np.mean)
''' grouped.aggregate(np.mean(axis=1))
 '''


for i in range(123):
    print('except_{0}'.format(i))