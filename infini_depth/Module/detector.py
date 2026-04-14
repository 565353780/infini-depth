import os
import torch

from typing import List, Optional

from camera_control.Module.camera import Camera


class Detector(object):
    def __init__(
        self,
        model_file_path: Optional[str]=None,
    ) -> None:
        if model_file_path is not None:
            self.loadModel(model_file_path)
        pass

    def loadModel(
        self,
        model_file_path: str,
    ) -> bool:
        if not os.path.exists(model_file_path):
            print('[ERROR][Detector::loadModel]')
            print('\t model file not exist!')
        return True

    def detectCameras(
        self,
        camera_list: List[Camera],
    ) -> List[Camera]:
        return []
