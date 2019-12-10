import pandas as pd
import sys
import os
import shutil
import glob
import json
import yaml
from prepare_data_speech_music import pack_features_to_hdf5
import main_crnn_sed

h_naming = {
     "Higher_level" : [ 
        {
            "speech" : "",
            "nonspeech" : 
            {
                "Music": ["Happy","Sad","Angry","Scary"],
                "nonMusic":["laughter","clapping", "wind", "footsteps"]
            }
                    
        }
    ]
}

models = {
    "Root_path" : "/home/aodev/python-environments/dcase2017_task4_cvssp/models/",
    "model1" : "",
    "model2" : "",
    "model3" : "",
    "model4" : "",
}

config_acoustic = {
                "input_path": "/home/aodev/python-environments/datafiles/eval_exc_p/*" ,
                "model_path": "/home/aodev/python-environments/dcase2017_task4_cvssp/models/all_classes_p_files/*",
                "embedding_path" : "/home/aodev/python-environments/dcase2017_task4_cvssp/combined_classes/",
                "csv_file_path" : " "
                }

class Model_hierarchy:

    def __init__(self, h_naming):
        self.h_naming = h_naming
    
    # def depth(self):
    #     h_naming.keys()
    #     return h_naming

    def level_mapping(self,keyss):
        level_keys = []
        level_val = []
        for key,vals in keyss.items():
            level_keys.append(key)
            level_val.append(vals)
        return level_keys, level_val

    def granular_mapping(self,val):
        return val

def get_h5_file(config_acoustic):

    dir_path = config_acoustic["embedding_file_p"]

    df = pd.read_csv(config_acoustic["csv_path"])
    os.system("")
    

    return "get the embeddings, save, and return the path"

def train_a_model(config_acoustic):

    


    return "prints model_training is done, and returns path_of_model"

model_h_instance = Model_hierarchy(h_naming)

level1_mapping,_ = model_h_instance.level_mapping(h_naming['Higher_level'][0])

level2_mapping,_ = model_h_instance.level_mapping(h_naming['Higher_level'][0]['nonspeech'])

level3_1_mapping = model_h_instance.granular_mapping(h_naming['Higher_level'][0]['nonspeech']['Music'])

level3_2_mapping = model_h_instance.granular_mapping(h_naming['Higher_level'][0]['nonspeech']['nonMusic'])


# def train_hierachial_model(model_h_instance, train=["1","2","3"]):

#     embedding = get_base_embeddings(frames, config_acoustic["embedding_path"])
#     model = {}
#     for depth in train:

#         model[depth] = train_a_model(y,embedding)

#     return model

from collections import namedtuple

Model_path = namedtuple("Model_path",'Root_path model1 model2 model3 model4')
model = Model_path(*list(models.values()))

def predict_from_embedding(depth = 4):
    base_embedding = ""
    Root_path = model.Root_path

    for i in range((depth+1)):

        if i >1 :
            model_key = "model" + str(i-1)
            model_path = Root_path + model.model_key
        
    return "Path of csv files"
    

def list_all_audiofiles():

    return "Text file containing path of all audiofiles"

def generate_h5_audiofiles(file_list):

    return "Path of the h5 file"

def predict_all():

    file_list = list_all_audiofiles()
    h5_path = generate_h5_audiofiles(file_list)

    final_prediction_path = predict_from_embedding(models,depth=depth)

    return h5_path, final_prediction_path













# def hierarchy_depth:

#     return hierarch_depth

# def filter_by_depth(depth):
#     return

