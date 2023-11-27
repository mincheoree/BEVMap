import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
import os 
from nuscenes.utils.data_classes import Box
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
# from nuscenes.map_expansion import arcline_path_utils
# from nuscenes.map_expansion.bitmap import BitMap
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
import argparse

def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage", 
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="arg parser")
    args = parser.parse_args()
    dataroot = 'data/mini/'
    nusc_maps = get_nusc_maps(dataroot)
    nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
    scene2map = {}
    
    for rec in nusc.scene: 
        log = nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']
    
    layer_names = ['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
    
