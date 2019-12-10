import json
import warnings
import pre_final_file
import heirarchy
import collections
import yaml
import argparse
import shutil
import copy

parser = argparse.ArgumentParser(
    description='Acoustic Training and Prediction Service')

# #=================Arguments used to run the Training Jobs ======================#
# parser.add_argument('--bucket_train_data_path', type=str)
# parser.add_argument('--run_name', type=str, default='age gender')
# parser.add_argument('--mlflow_tracking_uri', type=str,
#                     default='http://35.231.153.159:5000')

parser.add_argument('--FeatureExtraction', type=bool, default=True)
parser.add_argument('--level_1', type=bool, default=False)
parser.add_argument('--level_2', type=bool, default=True)
parser.add_argument('--level_3_Music', type=bool, default=False)
parser.add_argument('--level_3_nonMusic', type=bool, default=False)
parser.add_argument('--level_3_combined', type=bool, default=False)

parser.add_argument('--training', type=bool, default=True)
parser.add_argument('--prediction', type=bool, default=False)

# FeatureExtraction Parser
parser.add_argument('--input_path_to_audio_files', type=str)
parser.add_argument('--output_path_to_audio_features', type=str)

# Csv Related data Parser
parser.add_argument('--combined_csv_for_all_classes', type=str,default="meta_data/combined_granular.csv")

# level_1 Training Parser
parser.add_argument('--input_extra_csv_path', type=str)
parser.add_argument('--output_h5_file', type=str, default=None)
parser.add_argument('--output_training_scalar', type=str, default=None)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--steps_per_epoch', type=str, default=100)

parser.add_argument('--Root_path',type=str)
parser.add_argument('--model1',type=str)
# parser.add_argument('--model2',type=str)
parser.add_argument('--model3',type=str)
parser.add_argument('--model4',type=str)

parser.add_argument('--prediction_csv_level_1',type=str)
parser.add_argument('--prediction_csv_level_2',type=str)
parser.add_argument('--prediction_csv_level_3_music',type=str)
parser.add_argument('--prediction_csv_level_3_nonmusic',type=str)
parser.add_argument('--prediction_csv_level_3_combined',type=str)

# parser.add_argument('--classes', type=list, default=['speech,nonspeech'])

# level_2 Training Parser
# parser.add_argument('--input_extra_csv_path', type=str)
# parser.add_argument('--output_h5_file', type=str, default='')
# parser.add_argument('--output_training_scalar', type=str, default='')
# parser.add_argument('--epochs', type=int, default=50)
# parser.add_argument('--steps_per_epoch', type=str, default=100)
# parser.add_argument('--classes', type=str, default=['Music', 'nonMusic'])

# level_3_Music Training Parser
# parser.add_argument('--input_extra_csv_path', type=str)
# parser.add_argument('--output_h5_file', type=str, default='')
# parser.add_argument('--output_training_scalar', type=str, default='')
# parser.add_argument('--epochs', type=int, default=50)
# parser.add_argument('--steps_per_epoch', type=str, default='Adam')
# parser.add_argument('--classes', type=str,
#                     default=['Joyful_sounds', 'Sad', 'Angry', 'Scary'])

# level_3_nonMusic Training Parser
# parser.add_argument('--input_extra_csv_path', type=str)
# parser.add_argument('--output_h5_file', type=str, default='')
# parser.add_argument('--output_training_scalar', type=str, default='')
# parser.add_argument('--epochs', type=int, default=50)
# parser.add_argument('--steps_per_epoch', type=str, default='Adam')
# parser.add_argument('--classes', type=str,
#                     default=['Vocal_sounds', 'clapping', 'Natural_sounds', 'Foley_sounds'])

# level_3_combined Training Parser
# parser.add_argument('--input_extra_csv_path', type=str,default='')
# parser.add_argument('--output_h5_file', type=str, default='')
# parser.add_argument('--output_training_scalar', type=str, default='')
# parser.add_argument('--epochs', type=int, default=50)
# parser.add_argument('--steps_per_epoch', type=str, default='Adam')
# parser.add_argument('--classes', type=str, default=['Joyful_sounds', 'Sad', 'Angry',
#                                                     'Scary', 'Vocal_sounds', 'clapping', 'Natural_sounds', 'Foley_sounds'])

args = vars(parser.parse_args())
print(args)

mode__training = True
mode__prediction = False

FeatureExtraction__input_path_to_audio_files = args['input_path_to_audio_files']
FeatureExtraction__output_path_to_audio_features = args['output_path_to_audio_features']

csv_related_data__combined_csv_for_all_classes = args['combined_csv_for_all_classes']

level_1__input_extra_csv_path = args['input_extra_csv_path']
level_1__output_h5_file = args['output_h5_file']
level_1__output_training_scalar = args['output_training_scalar']
level_1__training__epochs = args['epochs']
level_1__training__steps_per_epoch = args['steps_per_epoch']
# level_1__training__classes = args['classes']

# level_2__input_extra_csv_path = args['input_extra_csv_path']
# level_2__output_h5_file = args['output_h5_file']
# level_2__output_training_scalar = args['output_training_scalar']
level_2__training__epochs = args['epochs']
level_2__training__steps_per_epoch = args['steps_per_epoch']
# level_2__training__classes = args['classes']

level_3_Music__input_extra_csv_path = args['input_extra_csv_path']
level_3_Music__output_h5_file = args['output_h5_file']
level_3_Music__output_training_scalar = args['output_training_scalar']
level_3_Music__training__epochs = args['epochs']
level_3_Music__training__steps_per_epoch = args['steps_per_epoch']
# level_3_Music__training__classes = args['classes']

level_3_nonMusic__input_extra_csv_path = args['input_extra_csv_path']
level_3_nonMusic__output_h5_file = args['output_h5_file']
level_3_nonMusic__output_training_scalar = args['output_training_scalar']
level_3_nonMusic__training__epochs = args['epochs']
level_3_nonMusic__training__steps_per_epoch = args['steps_per_epoch']
# level_3_nonMusic__training__classes = args['classes']

level_3_combined__input_extra_csv_path = args['input_extra_csv_path']
level_3_combined__output_h5_file = args['output_h5_file']
level_3_combined__output_training_scalar = args['output_training_scalar']
level_3_combined__training__epochs = args['epochs']
level_3_combined__training__steps_per_epoch = args['steps_per_epoch']
# level_3_combined__training__classes = args['classes']


models__Root_path = args['Root_path']
models__model1 = args['model1']
# models__model2 = args['model2']
models__model3 = args['model3']
models__model4 = args['model4']
# models__model5 = args['model5']

# config_prediction__input_extra_csv_path = args['input_extra_csv_path']
# config_prediction__output_h5_file = args = ['output_h5_file']
# config_prediction__assets__prediction_csv_level_1 = args['prediction_csv_level_1']
# config_prediction__assets__prediction_csv_level_2 = args['prediction_csv_level_2']
# config_prediction__assets__prediction_csv_level_3_music = args['prediction_csv_level_3_music']
# config_prediction__assets__prediction_csv_level_3_nonmusic = args['prediction_csv_level_3_nonmusic']
# config_prediction__assets__prediction_csv_level_3_combined = args['prediction_csv_level_3_combined']

pre_final_file.start_training(locals())

print('Training/Prediction Job complete')

