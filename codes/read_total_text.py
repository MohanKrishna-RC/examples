from os.path import dirname, join as pjoin
import scipy.io as sio
import glob
import numpy as np 
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures 

for _file in glob.glob("/home/mohan/Train_total_text/*"):
    data = loadmat(_file)
    print(list(data.keys()))
    print(data)
    break

