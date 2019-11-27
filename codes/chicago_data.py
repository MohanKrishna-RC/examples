from __future__ import print_function
import sys, os
import tempfile, urllib, zipfile
# Confirm that we're using Python 2
assert sys.version_info.major is 2, 'Oops, not running Python 2'

# Set up some globals for our file paths
BASE_DIR = tempfile.mkdtemp()
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'chicago_taxi_output')
TRAIN_DATA = os.path.join(DATA_DIR, 'train', 'data.csv')
EVAL_DATA = os.path.join(DATA_DIR, 'eval', 'data.csv')
SERVING_DATA = os.path.join(DATA_DIR, 'serving', 'data.csv')

# Download the zip file from GCP and unzip it
zip, headers = urllib.urlretrieve('https://storage.googleapis.com/tfx-colab-datasets/chicago_data.zip')
zipfile.ZipFile(zip).extractall(BASE_DIR)
zipfile.ZipFile(zip).close()

print("Here's what we downloaded:")
#ls -lR {os.path.join(BASE_DIR, 'data')}
# !pip install -q tensorflow_data_validation

import tensorflow_data_validation as tfdv
print('TFDV version: {}'.format(tfdv.version.__version__))
train_stats = tfdv.generate_statistics_from_csv(data_location=TRAIN_DATA) #(ignore the snappy warnings)
tfdv.visualize_statistics(train_stats)

""" Schema """
schema = tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema=schema)

""" Check evaluation data for errors """
# Compute stats for evaluation data
eval_stats = tfdv.generate_statistics_from_csv(data_location=EVAL_DATA)

# Compare evaluation data with training data
tfdv.visualize_statistics(lhs_statistics=eval_stats, rhs_statistics=train_stats,
                          lhs_name='EVAL_DATASET', rhs_name='TRAIN_DATASET')

""" Check for evaluation anomalies """
anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=schema)
tfdv.display_anomalies(anomalies)                          

""" Fix evaluation anomalies in the schema """


import tempfile

MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))
if os.path.isdir(export_path):
    print('\nAlready saved a model, cleaning up\n')
    !rm -r {export_path}


