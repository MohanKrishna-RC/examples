import pandas as pd
import numpy as np
import glob
import os
import json


class QualityParams(object):

    def __init__(self,path_to_json):
        self.path = path_to_json
        self.data = self._read_json()
        self.df = self.imp_data_from_csv()

    def _read_json(self):
        with open(self.path) as f:
            _data = json.load(f)
        print(_data)
        return _data

    def imp_data_from_csv(self):
        df = pd.read_csv(self.data['Screen_share']['input_data']['Input_csv_path'])
        print(df.head())
        return df

    def dist_betw_coordinates(self):
        """ you could remove the sqrt for computation benefits, its a symetric func
        that does not change the relative ordering of distances """
        roi_coordinates_distance = list(zip(self.df['xmin'],self.df['ymin'],self.df['xmax'],self.df['ymax']))
        frame_coordinate_distance = list(zip(self.df['xmin'],self.df['ymin'],self.df['xmax'],self.df['ymax']))
        return roi_coordinates_distance, frame_coordinate_distance
        
    def area_of_the_frame(self):
        _,frame_coordinate_distance = self.dist_betw_coordinates()
        area_of_frame = [np.sqrt(((frame_coordinate_distance[i][2]-frame_coordinate_distance[i][0])** 2 + (frame_coordinate_distance[i][3] - frame_coordinate_distance[i][1])**2)) for i in range(len(frame_coordinate_distance))]
        return area_of_frame

    def area_of_roi(self):
        roi_coordinates_distance,_ = self.dist_betw_coordinates()
        area_of_roi = [np.sqrt(((roi_coordinates_distance[i][2]-roi_coordinates_distance[i][0])** 2 + (roi_coordinates_distance[i][3] - roi_coordinates_distance[i][1])**2)) for i in range(len(roi_coordinates_distance))]
        return area_of_roi

    def screen_share(self):
        area_of_the_frame = self.area_of_the_frame()
        area_of_roi = self.area_of_roi()
        self.df['screen_share'] =  [area_of_roi[i]//area_of_the_frame[i] for i in range(len(area_of_roi))]
        print(self.df)
        self.df.to_csv(self.data['Screen_share']['output_data']['Output_csv_path'])

dd = QualityParams("/home/mohan/Documents/Python-Necessities/data/raw/config_schema_for_Quality_params.json")
dd.dist_betw_coordinates()
dd.screen_share()

