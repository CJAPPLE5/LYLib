import os
import sys
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation
os.chdir("/home/liuyong/projects/Reconstruction/gaussian-splatting")
sys.path.append("/home/liuyong/projects/Reconstruction/gaussian-splatting")  
from scripts.colmap.database import *
from scene.colmap_loader import read_points3D_binary
import open3d as o3d
import random
import torch


feature_extract_command = "colmap feature_extractor \
     --database_path {} \
     --image_path {} \
     --SiftExtraction.gpu_index=0,1,2,3"
     
feature_matcher_command = "colmap exhaustive_matcher \
     --database_path {} \
     --SiftMatching.num_threads=24 \
     --SiftMatching.gpu_index=0"

# input_path: sparse_model path
# output_path: triangulate path
triangulate_command = "colmap point_triangulator \
     --database_path {} \
     --image_path {} \
     --input_path {} \
     --output_path {}"

def getColmapPose(c2w):
     w2c = np.linalg.inv(c2w)
     w2c_t = w2c[:3,3]
     w2c_r = w2c[:3,:3]
     w2c_quat = Rotation.from_matrix(w2c_r).as_quat()
     w2c_quat = w2c_quat[[3,0,1,2]]
     return w2c_quat, w2c_t

def generate_sparse_model_with_pose(project_base, cameras, imagename2pose, imagename2cameraid):
     os.makedirs(project_base, exist_ok=True)
     sparse_model_folder = os.path.join(project_base, "sparse")
     triangulate_folder = os.path.join(project_base, "triangulate")
     database_folder = os.path.join(project_base, "database")
     image_folder = os.path.join(project_base, "images")
     database_path = os.path.join(database_folder, "database.db")

     os.system("rm -r {}".format(sparse_model_folder))
     os.system("rm -r {}".format(triangulate_folder))
     os.system("rm -r {}".format(database_folder))
     
     os.makedirs(sparse_model_folder, exist_ok=True)
     os.makedirs(triangulate_folder, exist_ok=True)
     os.makedirs(database_folder, exist_ok=True)

     # extract feature
     os.system(feature_extract_command.format(database_path, image_folder))
     # match feature
     os.system(feature_matcher_command.format(database_path))

     # create sparse model txt
     # write points3D.txt
     f = open(os.path.join(sparse_model_folder, "points3D.txt"), "w")
     f.close()

     # write cameras.txt
     f = open(os.path.join(sparse_model_folder, "cameras.txt"), "w")
     for camera in cameras:
          camera_id, W, H, fx, fy, cx, cy = camera
          f.write("{} PINHOLE {} {} {} {} {} {}\n".format(camera_id, W, H, fx, fy, cx, cy))
     f.close()
    
     # write images.txt
     f = open(os.path.join(sparse_model_folder, "images.txt"), "w")
     db = COLMAPDatabase.connect(database_path)
     images = dict(
          (image_id, image_name)
          for image_id, image_name in db.execute("SELECT image_id, name FROM images")
     )
     for image_id, image_name in tqdm(images.items()):
          camera_id = imagename2cameraid[image_name]
          pose = imagename2pose[image_name]
          w2c_quat, w2c_t = pose
          f.write("{} {} {} {} {} {} {} {} {} {}\n\n".format(
               image_id, 
               w2c_quat[0], w2c_quat[1], w2c_quat[2], w2c_quat[3],
               w2c_t[0], w2c_t[1], w2c_t[2], 
               camera_id, image_name
          ))
     f.close()
     
     # triangulate sparse model
     os.system(triangulate_command.format(database_path,
                                          image_folder,
                                          sparse_model_folder,
                                          triangulate_folder))
     
     # save ply model
     points_binary_path = os.path.join(triangulate_folder, "points3D.bin")
     
     xyz, color, _ = read_points3D_binary(points_binary_path)
     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
     pcd.colors = o3d.utility.Vector3dVector(color/255.0)
     o3d.io.write_point_cloud(os.path.join(triangulate_folder, "points3d.ply"), pcd)
     
# replica test

# replica_base = "/home/liuyong/projects/Reconstruction/gaussian-splatting/data/Replica"
# test_scene = "office0"
# test_images = list(range(0, 2000,10))

# origin_images_folder = os.path.join(replica_base, test_scene, "results")
# origin_pose_path = os.path.join(replica_base, test_scene, "traj.txt")
# poses = np.loadtxt(origin_pose_path).reshape(-1,4,4)

# replica_colmap_base = "/home/liuyong/projects/Reconstruction/gaussian-splatting/data/Replica/test"
# # camera_id, W, H, fx, fy, cx, cy
# cameras = [[1, 1200, 680, 600, 600, 599.5, 339.5]]
# imagename2pose = {}
# imagename2cameraid = {}
# camera_id = 1

# for image_idx in test_images:
#      image_name = "frame%06d.jpg"%(image_idx)
#      w2c_quat, w2c_t = getColmapPose(poses[image_idx])
#      imagename2pose[image_name] = [w2c_quat, w2c_t]
#      imagename2cameraid[image_name] = camera_id

# generate_sparse_model_with_pose(replica_colmap_base, cameras,
#                                 imagename2pose, imagename2cameraid)

image_num = 2000

waymo_base = "/home/liuyong/projects/Reconstruction/gaussian-splatting/data/pytorch_waymo/train"
image_base = os.path.join(waymo_base, "rgbs")
metadata_base = os.path.join(waymo_base, "metadata")

image_names = sorted(os.listdir(image_base), key=lambda x : int(x.split(".")[0]))
image_paths = [os.path.join(image_base, i) for i in image_names]
metadata_paths = [os.path.join(metadata_base, i.replace(".png", ".pt")) for i in image_names]

if image_num == -1:
     image_num = len(image_paths)

random_indices = random.sample(range(len(image_paths)), image_num)  # 选择要采样的索引位置 

image_paths = [image_paths[i] for i in random_indices]
metadata_paths = [metadata_paths[i] for i in random_indices]

cam_info = {}
cameras = []
cam_waymoidx2colmapidx = {}
for i in tqdm(range(len(metadata_paths))):
     camera = torch.load(metadata_paths[i])
     cam_idx = camera["cam_idx"]
     H, W = camera["H"], camera["W"]
     fx, fy = camera["intrinsics"][0], camera["intrinsics"][1]
     cx, cy = W / 2.0, H / 2.0
     if cam_idx not in cam_info:
          cam_info[cam_idx] = {"H": camera["H"],
                                        "W": camera["W"],
                                        "intrinsic": camera["intrinsics"],
                                        "camera_idx": len(cam_info) + 1}
          # camera_id, W, H, fx, fy, cx, cy
          cameras.append([len(cam_info), W, H, fx, fy, cx, cy])
          cam_waymoidx2colmapidx[cam_idx] = len(cam_info)
     if len(cam_info) == 12:
          break

# create colmap workspace
project_base = os.path.join(waymo_base, "colmap")
os.system("rm -r {}".format(project_base))
os.makedirs(project_base, exist_ok=True)

project_image_folder = os.path.join(project_base, "images")
os.makedirs(project_image_folder)
for image_path in image_paths:
     image_name = os.path.basename(image_path)
     os.system("cp {} {}".format(image_path,
                                 os.path.join(project_image_folder, image_name)))

# create imagename2pose
imagename2pose = {}
imagename2camidx = {}
for image_idx, image_path in  enumerate(image_paths):
     image_name = os.path.basename(image_path)
     metadata = torch.load(metadata_paths[image_idx])
     c2w_ = metadata["c2w"].cpu().numpy()
     c2w_[:3,1:3] *= -1
     c2w = np.eye(4)
     c2w[:3,:] = c2w_
     w2c_quat, w2c_t = getColmapPose(c2w)
     imagename2pose[image_name] = [w2c_quat, w2c_t]
     cam_waymoidx = metadata["cam_idx"]
     imagename2camidx[image_name] = cam_waymoidx2colmapidx[cam_waymoidx]
     
# print(imagename2pose)
# print(cameras)
generate_sparse_model_with_pose(project_base, cameras,
                                imagename2pose,
                                imagename2camidx)

