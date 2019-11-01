# from sklearn import datasets
# from sklearn import svm
# from sklearn.externals import joblib
# # load iris dataset
# iris = datasets.load_iris()
# X, y = iris.data, iris.target
# # train model
# clf = svm.LinearSVC()
# clf.fit(X, y)
# # persistent model
# joblib.dump(clf, 'iris_model.pickle')


import os
import random
import glob2
import glob
from pathlib import Path
src = Path("/home/mohan/Documents/age_gender_embed_2/kali/data/")

data_dir = '/home/mohan/Documents/age_gender_embed_2/kali/data/*'
a = data_dir.split(',')

for image_path in src.glob("*.jpg"):
    print(image_path)
    # file = image_path.name
    file = random.choice(image_path)
    print(file)

a = input('AGE')