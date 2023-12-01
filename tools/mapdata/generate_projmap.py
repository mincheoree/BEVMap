import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Polygon
import descartes
import numpy as np
import cv2
from nuscenes.utils.geometry_utils import view_points
import os 
from nuscenes.utils.data_classes import Box
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer 
# from nuscenes.map_expansion import arcline_path_utils
# from nuscenes.map_expansion.bitmap import BitMap
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
import argparse
import copy

color_map = dict(drivable_area='#a6cee3',
                             road_segment='#1f78b4',
                             road_block='#b2df8a',
                             lane='#33a02c',
                             ped_crossing='#fb9a99',
                             walkway='#e31a1c',
                             stop_line='#fdbf6f',
                             carpark_area='#ff7f00',
                             road_divider='#cab2d6',
                             lane_divider='#6a3d9a',
                             traffic_light='#7e772e')

def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage", 
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps

def render_map_in_image(nusc,
                        nusc_map,
                         sample_token, 
                            camera_channel,
                            near_plane = 1e-8, 
                            render_behind_cam = True, 
                            render_outside_im = True,
                            min_polygon_area = 1000,
                            alpha = 1.0,
                            patch_radius = 50.0, 
                            im_size = (1600, 900), 
                            out_path = None
                            ):

        layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
        # Check layers whether we can render them.

        sample_record = nusc.get('sample', sample_token)
        cam_token = sample_record['data'][camera_channel]
        cam_record = nusc.get('sample_data', cam_token)
        cs_record = nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])

        # Retrieve the current map
        poserecord =  nusc.get('ego_pose', cam_record['ego_pose_token'])
        ego_pose = poserecord['translation']
        box_coords = (
        ego_pose[0] - patch_radius,
        ego_pose[1] - patch_radius,
        ego_pose[0] + patch_radius,
        ego_pose[1] + patch_radius,
    )
        records_in_patch = nusc_map.get_records_in_patch(box_coords, layer_names, 'intersect')

        # Init axes.
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, im_size[0])
        ax.set_ylim(0, im_size[1])
        # ax.imshow(im)

        depth_points = []

        # Retrieve and render each record.
        for layer_name in layer_names:
            for token in records_in_patch[layer_name]:
                record = nusc_maps[map_name].get(layer_name, token)
                if layer_name == 'drivable_area':
                    polygon_tokens = record['polygon_tokens']
                else:
                    polygon_tokens = [record['polygon_token']]

                for polygon_token in polygon_tokens:
                    polygon = nusc_maps[map_name].extract_polygon(polygon_token)

                    # Convert polygon nodes to pointcloud with 0 height.
                    points = np.array(polygon.exterior.xy)
                    points = np.vstack((points, np.zeros((1, points.shape[1]))))
                    
                    
                    
                    # Transform into the ego vehicle frame for the timestamp of the image.
                    points = points - np.array(poserecord['translation']).reshape((-1, 1))
                    points = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, points)

                    # Transform into the camera.
                    points = points - np.array(cs_record['translation']).reshape((-1, 1))
                    points = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, points)

                    # Remove points that are partially behind the camera.
                    depths = points[2, :]
        #             print(points.shape)
        #             print(depths)
                    behind = depths < near_plane
                    if np.all(behind):
                        continue

                    if render_behind_cam:
                        # Perform clipping on polygons that are partially behind the camera.
                        points = NuScenesMapExplorer._clip_points_behind_camera(points, near_plane)
                     
                    elif np.any(behind):
                        # Otherwise ignore any polygon that is partially behind the camera.
                        continue

                    # Ignore polygons with less than 3 points after clipping.
                    if len(points) == 0 or points.shape[1] < 3:
                        continue
        
                    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
                    points = view_points(points, cam_intrinsic, normalize=True)
                        
                    # Skip polygons where all points are outside the image.
                    # Leave a margin of 1 pixel for aesthetic reasons.
                    inside = np.ones(points.shape[1], dtype=bool)
                    inside = np.logical_and(inside, points[0, :] > 1)
                    inside = np.logical_and(inside, points[0, :] < im_size[0] - 1)
                    inside = np.logical_and(inside, points[1, :] > 1)
                    inside = np.logical_and(inside, points[1, :] < im_size[1] - 1)

                    if render_outside_im:
                        if np.all(np.logical_not(inside)):
                            continue
                    else:
                        if np.any(np.logical_not(inside)):
                            continue
                    
                    points = points[:2, :]
                    
                
                    ### get polygon
                    points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
                 
                
                    polygon_proj = Polygon(points)

                    # Filter small polygons
                    if polygon_proj.area < min_polygon_area:
                        continue
                    label = layer_name
                    ax.add_patch(descartes.PolygonPatch(polygon_proj, fc=color_map[layer_name], alpha=0.3,
                                                        label=label))
                     
                    # depths = depths[inside]
                    # final = np.concatenate([points.T, depths[:, None]], axis =1).astype(np.float32)
                    # depth_points.append(final)

        # Display the image.
        plt.axis('off')
        ax.invert_yaxis()

        if out_path is not None:
            plt.tight_layout()
            plt.savefig(f'{out_path}.png', bbox_inches='tight', pad_inches=0)
            plt.close()

    
    
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
    
    layer_names = ['road_segment', 'road_block', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
    cams = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    os.makedirs(os.path.join(dataroot, 'projmap'))
    for cam in cams: 
        os.makedirs(os.path.join(dataroot, 'projmap', cam))
    for sample in tqdm(nusc.sample): 
        sample_token = sample['token']
        map_name = scene2map[nusc.get('scene', sample['scene_token'])['name']]
        nusc_map = nusc_maps[map_name]
        for cam in cams:
            cam_token = sample['data'][cam]
            out_path = os.path.join(dataroot, 'projmap', cam, cam_token)
            render_map_in_image(nusc, nusc_map, sample_token = sample_token, camera_channel=cam, out_path = out_path)
