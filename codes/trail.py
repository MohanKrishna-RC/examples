import pandas as pd
import os
import numpy as np
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]},index=['a', 'b', 'c'])
df.to_hdf('data.h5', key='df', mode='w')
df1 = pd.read_hdf('data.h5','df')
os.remove('data.h5')

ages_out = np.arange(0, 101).reshape(101, 1)