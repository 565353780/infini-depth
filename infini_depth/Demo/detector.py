import sys
sys.path.append('../camera-control')
sys.path.append('../../../camera-control')

import os

from camera_control.Module.camera_convertor import CameraConvertor

from infini_depth.Module.detector import Detector


def demo():
    home = os.environ['HOME']
    model_file_path = f'{home}/chLi/Model/InfiniDepth/infinidepth.ckpt'

    colmap_data_folder_path = f'{home}/chLi/Dataset/GS/haizei_zihan/colmap_normalized/'

    camera_list = CameraConvertor.loadColmapDataFolder(colmap_data_folder_path)

    detector = Detector(model_file_path)
    depth = detector.detectCameras(camera_list)
    return True
