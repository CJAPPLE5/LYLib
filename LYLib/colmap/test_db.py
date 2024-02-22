import os
import torch
from tqdm import tqdm
from database import *

# read cameras
def sort_images(image_names):
    return sorted(image_names, key=lambda x : int(x.split(".")[0]))

waymo_base = "../../data/pytorch_waymo/train"

image_base = os.path.join(waymo_base, "rgbs")
metadata_base = os.path.join(waymo_base, "metadata")

image_names = sort_images(os.listdir(image_base))
image_paths = [os.path.join(image_base, i) for i in image_names]
metadata_paths = [os.path.join(metadata_base, i.replace(".png", ".pt")) for i in image_names]

cam_info = {}
cam_set = set()
for i in tqdm(range(len(metadata_paths))):
    camera = torch.load(metadata_paths[i])
    cam_idx = camera["cam_idx"]
    if cam_idx not in cam_info:
        cam_info[cam_idx] = {"H": camera["H"],
                                    "W": camera["W"],
                                    "intrinsic": camera["intrinsics"],
                                    "camera_idx": len(cam_info) + 1}


data_base_path = "../../data/pytorch_waymo/colmap/database/database.db"

db = COLMAPDatabase.connect(data_base_path)

PINHOLE=1

for cam_id in cam_info:
    camera_info = cam_info[cam_id]
    camera_id = camera_info['camera_idx']
    H, W = camera_info["H"], camera_info["W"]
    intrinsic = camera_info["intrinsic"].cpu().numpy()
    params = np.array([intrinsic[0], intrinsic[1], W / 2.0, H / 2.0], dtype=np.float64)
    db.update_camera(PINHOLE, W, H, params, camera_id)

db.commit()


images = dict(
    (image_id, image_name)
    for image_id, image_name in db.execute("SELECT image_id, name FROM images")
)

for image_id, image_name in tqdm(images.items()):
    metadata_path = os.path.join("/home/liuyong/projects/Reconstruction/gaussian-splatting/data/pytorch_waymo/train/meta_rgbs/69", image_name.replace(".png", ".pt"))
    metadata = torch.load(metadata_path)
    cam_idx = metadata["cam_idx"]
    cam_idx_db = cam_info[cam_idx]["camera_idx"]
    db.update_image_camera(image_id, cam_idx_db)

images = dict(
    (image_id, [image_name, camera_id])
    for image_id, image_name, camera_id in db.execute("SELECT image_id, name, camera_id FROM images")
)

# print(images)
db.close()