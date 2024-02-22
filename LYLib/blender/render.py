import sys
import os
import random
import pickle
import bpy
import yaml
import re
import numpy as np
import cv2
import Imath
import array
import OpenEXR
import cv2
import subprocess
import shutil
import json
from mathutils import Euler, Matrix, Vector

from argparse import ArgumentParser

abs_path = os.path.abspath(__file__)

from .bpy_helper import * 

def render_rgbd_sequence(c2ws, intrinsic, save_path ):
    np.random.seed(2024)
    os.system("rm -r {}".format(save_path))
    os.makedirs(os.path.join(save_path, "color"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "depth"), exist_ok=True)
    
    with open(os.path.join(abs_path, "default_config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    fx, fy, cx, cy = intrinsic
    w, h = int(cx * 2), int(cy * 2)
    g_cam_fov = np.rad2deg(2 * np.arctan(cx/fx))
    
    config["g_resolution_x"] = w
    config["g_resolution_y"] = h
    config["g_cam_fov"] = g_cam_fov
    
    init_all(config)
    
    render_obj_by_cam_poses(config, c2ws)
    
if __name__ == "__main__":
    pass
    
    
