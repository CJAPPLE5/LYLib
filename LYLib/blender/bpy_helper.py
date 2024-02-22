import bpy
from mathutils import Euler, Matrix, Vector
import numpy as np
import os
import OpenEXR
import cv2
import array
import Imath
import re

def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x


def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


def get_calibration_matrix_K_from_blender(camd):
    # print(camd.type)
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0  # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
         (0, s_v, v_0),
         (0, 0, 1)))
    return K

# 初始化场景设置
def scene_setting_init(config):
    bpy.context.scene.tool_settings.use_keyframe_insert_auto = False
    sce = bpy.context.scene.name
    # 'BLENDER_EEVEE'
    bpy.data.scenes[sce].render.engine = config["g_engine_type"]

    #output
    bpy.data.scenes[sce].render.image_settings.color_mode = config["g_depth_color_mode"]
    bpy.data.scenes[sce].render.image_settings.color_depth = config["g_depth_color_depth"]
    bpy.data.scenes[sce].render.image_settings.file_format = config["g_depth_file_format"]
    bpy.data.scenes[sce].render.use_overwrite = config["g_depth_use_overwrite"]
    bpy.data.scenes[sce].render.use_file_extension = config["g_depth_use_file_extension"] 
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    bpy.context.scene.render.film_transparent = True


    #dimensions
    bpy.data.scenes[sce].render.resolution_x = config["g_resolution_x"]
    bpy.data.scenes[sce].render.resolution_y = config["g_resolution_y"]
    bpy.data.scenes[sce].render.resolution_percentage = config["g_resolution_percentage"]

    if config["use_gpu"]:
            # only cycles engine can use gpu
        bpy.data.scenes[sce].render.engine = 'CYCLES'
        bpy.data.scenes[sce].render.tile_x = config["g_hilbert_spiral"]
        bpy.data.scenes[sce].render.tile_x = config["g_hilbert_spiral"]
        bpy.context.user_preferences.addons['cycles'].preferences.devices[0].use = True
        bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.types.CyclesRenderSettings.device = 'GPU'
        bpy.data.scenes[sce].cycles.device = 'GPU'
        
# 连接节点保存深度图
def node_setting_init(config):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for node in tree.nodes:
        tree.nodes.remove(node)
    
    render_layer_node = tree.nodes.new('CompositorNodeRLayers')
    file_output_node = tree.nodes.new('CompositorNodeOutputFile')


    file_output_node.format.color_mode = config["g_depth_color_mode"]
    file_output_node.format.color_depth = config["g_depth_color_depth"]
    file_output_node.format.file_format = config["g_depth_file_format"] 
    file_output_node.base_path = config["g_save_path_color"]


    color_output_node = tree.nodes.new('CompositorNodeOutputFile')
    color_output_node.format.color_mode = config["g_rgb_color_mode"]
    color_output_node.format.color_depth = config["g_rgb_color_depth"]
    color_output_node.format.file_format = config["g_rgb_file_format"] 
    color_output_node.base_path = config["g_save_path_depth"]
    
    links.new(render_layer_node.outputs[2], file_output_node.inputs[0])
    links.new(render_layer_node.outputs[0], color_output_node.inputs[0])
    
# 设置相机fov等
def camera_setting_init(config):
    bpy.data.cameras['Camera'].type = 'PERSP'
    bpy.data.cameras['Camera'].clip_start = config["g_depth_clip_start"]
    bpy.data.cameras['Camera'].clip_end = config["g_depth_clip_end"]
    bpy.data.objects['Camera'].rotation_mode = config["g_rotation_mode"]
    cam = bpy.context.scene.camera.data
        
    cam.lens_unit = 'FOV'
    cam.angle = np.radians(config["g_cam_fov"])
    
    
def render( cam_pose=None):
    cam_ob = bpy.context.scene.camera
    trans, rot, scale =  Matrix(cam_pose).decompose()
    cam_ob.location = trans
    cam_ob.rotation_euler = rot.to_euler()


    file_output_node = bpy.context.scene.node_tree.nodes[1]
    file_output_node.file_slots[0].path = 'frame-######.depth' # blender placeholder #

    file_output_node = bpy.context.scene.node_tree.nodes[2]
    file_output_node.file_slots[0].path = 'frame-######.color.png' # blender placeholder #
    
    bpy.ops.render.render(animation=False, write_still=True)

# 用相机位姿序列渲染图片
def render_obj_by_cam_poses(config, cam_poses):
    # slam_poses = []
    save_path_depth = config["g_save_path_depth"]
    bpy.context.scene.frame_set(0)
    for i,cam_pose in enumerate(cam_poses):
        # 渲染
        render(cam_pose)
        # exr转png
        # exr2png(os.path.join(save_path_depth, "frame-%06d.depth.exr"%i), config["g_resolution_y"], 
                # config["g_resolution_x"], config["g_max_depth"])
    # 保存slam位姿
    # for i in range(len(slam_poses)):
    #     np.savetxt(os.path.join(obj_path, "frame-%06d.pose.txt"%i), slam_poses[i],fmt='%.3f')

def render_obj_by_animation(config, obj_path, cam_poses):
    animation_start = config["animation_start"]
    animation_end = config["animation_end"]
    ori_pose_inv = np.linalg.inv(cam_poses[0])
    slam_poses = []
    bpy.context.scene.frame_set(animation_start)
    bpy.context.scene.frame_start = animation_start
    bpy.context.scene.frame_end = animation_end
    render(config, obj_path)
    for frame in range(animation_start, animation_end+1):
        slam_poses.append(blender2slam(cam_poses[frame-animation_start], ori_pose_inv))
        
    for curr_frame in range(animation_start, animation_end+1):
        real_frame = curr_frame - animation_start
        ori_depth_name = "frame-%06d.depth.exr" % curr_frame
        real_depth_name = "frame-%06d.depth.exr" % real_frame
        ori_color_name = "frame-%06d.color.png" % curr_frame
        real_color_name = "frame-%06d.color.png" % real_frame

        os.rename(os.path.join(obj_path, ori_depth_name), os.path.join(obj_path, real_depth_name))
        os.rename(os.path.join(obj_path, ori_color_name), os.path.join(obj_path, real_color_name))
        
        exr2png(os.path.join(obj_path, "frame-%06d.depth.exr"%(curr_frame-animation_start)), config["g_resolution_y"], 
                config["g_resolution_x"], config["g_max_depth"], config["depth_noise"], config["noise_scale"])
    
    for i in range(len(slam_poses)):
        np.savetxt(os.path.join(obj_path, "frame-%06d.pose.txt"%i), slam_poses[i],fmt='%.3f')
        
def init_all(config):
    scene_setting_init(config)
    node_setting_init(config)
    camera_setting_init(config)
    bpy.context.scene.frame_set(0)

def set_depth_path(new_path):
    file_output_node = bpy.context.scene.node_tree.nodes[1]
    file_output_node.base_path = new_path
    
    color_output_node = bpy.context.scene.node_tree.nodes[2]
    color_output_node.base_path = new_path
    

# 从blender_pose路径下读取cam_pose(blender坐标系下)
def getCamPoses(pose_base):
    total_poses =  os.listdir(pose_base)
    pose_format = "frame-(\d{6}).pose.txt"
    cam_poses = []
    total_poses = sorted(total_poses, key=lambda x: re.findall(pose_format, x)[0]) 
    for pose_name in total_poses:
        pose_path = os.path.join(pose_base, pose_name)
        cam_poses.append(np.loadtxt(pose_path))
        
    return cam_poses

# 缩放物体, 并返回初始位置，旋转等信息，用于后续恢复
def scale_obj(obj):
    # print(obj.location)
    obj.select_set(True)
    bpy.ops.object.transform_apply( rotation = True, scale = True )
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    ori_location = obj.location.copy()
    ori_rotation = obj.rotation_euler.copy()
    ori_scale = obj.scale.copy()
    maxDim = max(obj.dimensions)
    obj.scale = obj.scale / maxDim
    obj.location = (0.0,0.0,0.0)
    obj.rotation_euler = (np.radians(0),0,0)
    obj.select_set(False)
    return [ori_location, ori_rotation, ori_scale]


def exr2png(exr_path):
    depth_file = OpenEXR.InputFile(exr_path)
    depth_arr = array.array('f', depth_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)))
    dw = depth_file.header()['dataWindow']  
    width = dw.max.x - dw.min.x + 1  
    height = dw.max.y - dw.min.y + 1 
    depth = np.array(depth_arr).reshape(height,width)
    depth *= 1000.0
    depth = depth.astype(np.uint16)
    cv2.imwrite(exr_path.replace(".exr",".png"), depth)

def blender2slam(b_pose, ori_pose_inv):
    s_pose = b_pose.copy()
    s_pose = ori_pose_inv @ s_pose
    
    s_pose[1,3] *= -1
    s_pose[2,3] *= -1 
    rot_mat = s_pose[:3,:3]

    rvec, _ = cv2.Rodrigues(rot_mat)
    rvec[1][0] *= -1
    rvec[2][0] *= -1
    
    rot_mat, _ = cv2.Rodrigues(rvec)
    s_pose[:3,:3] = rot_mat
    
    # np.savetxt(save_path, s_pose)
    return s_pose


def get_animation_pose(frame_start, frame_end):
    cam_poses = []
    for i in range(frame_start, frame_end+1):
        bpy.context.scene.frame_set(i)
        cam_poses.append(np.array(bpy.context.scene.camera.matrix_world))
    return cam_poses
