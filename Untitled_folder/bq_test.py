import csv
import pandas as pd
import numpy as np
import json

df = pd.read_csv("/home/mohan/Downloads/results-20190409-173807.csv")
df1 = pd.read_csv("/home/mohan/Downloads/results-20190409-173742.csv")

df['VIDEO_ID'].equals(df1['VIDEO_ID'])

df1['VIDEO_ID'].where((df['VIDEO_ID'].values!=df1['VIDEO_ID'].values),other=np.nan)

