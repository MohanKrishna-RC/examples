import apache_beam as beam  
import os
import preprocess
import shutil
import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_model_analysis as tfma
from google.protobuf import text_format 
from tensorflow.python.lib.io import file_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import schema_utils
from trainer import task
from trainer import taxi


BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, 'data')

OUTPUT_DIR = os.path.join(BASE_DIR, 'chicago_taxi_output')

# Base dir containing train and eval data
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
EVAL_DATA_DIR = os.path.join(DATA_DIR, 'eval')

# Base dir where TFT writes training data
TFT_TRAIN_OUTPUT_BASE_DIR = os.path.join(OUTPUT_DIR, 'tft_train')
TFT_TRAIN_FILE_PREFIX = 'train_transformed'

# Base dir where TFT writes eval data
TFT_EVAL_OUTPUT_BASE_DIR = os.path.join(OUTPUT_DIR, 'tft_eval')
TFT_EVAL_FILE_PREFIX = 'eval_transformed'

TF_OUTPUT_BASE_DIR = os.path.join(OUTPUT_DIR, 'tf')

# Base dir where TFMA writes eval data
TFMA_OUTPUT_BASE_DIR = os.path.join(OUTPUT_DIR, 'tfma')

SERVING_MODEL_DIR = 'serving_model_dir'
EVAL_MODEL_DIR = 'eval_model_dir'


def get_tft_train_output_dir(run_id):
    return _get_output_dir(TFT_TRAIN_OUTPUT_BASE_DIR, run_id)


def get_tft_eval_output_dir(run_id):
    return _get_output_dir(TFT_EVAL_OUTPUT_BASE_DIR, run_id)


def get_tf_output_dir(run_id):
    return _get_output_dir(TF_OUTPUT_BASE_DIR, run_id)

BASE_DIR = os.getcwd() #present directory

DATA_DIR = os.path.join(BASE_DIR, 'data')

OUTPUT_DIR = os.path.join(BASE_DIR, 'chicago_taxi_output')