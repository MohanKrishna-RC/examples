workspace = "/home/aodev/python-environments/dcase2017_task4_cvssp"

import pandas as pd
import json
import yaml
with open('training.yaml') as f:
    data = yaml.load(f)

step_time_in_sec = 1.0

if data['train']['Higher_level']:
    lbs = data['classes']['Higher_level']
elif data['train']['Higher_level']:
    lbs = data['classes']['Middle_level']
else:
    if data['train']['Granular_level']['Music']:
        lbs = data['classes']['Music']
    elif data['train']['Granular_level']['Music']:
        lbs = data['classes']['nonMusic']


idx_to_lb = {index: lb for index, lb in enumerate(lbs)}
lb_to_idx = {lb: index for index, lb in enumerate(lbs)}
num_classes = len(lbs)

