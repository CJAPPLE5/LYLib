PROJECT_PATH="/home/liuyong/projects/Reconstruction/gaussian-splatting/data/pytorch_waymo/train/colmap"

# colmap feature_extractor \
#     --database_path $PROJECT_PATH/database/database.db \
#     --image_path $PROJECT_PATH/../train/cam_rgbs/69 \
#     --SiftExtraction.gpu_index=0,1,2,3

# colmap exhaustive_matcher --database_path $PROJECT_PATH/database/database.db \
#     --SiftMatching.num_threads=-1 \
#     --SiftMatching.gpu_index=0

# colmap point_triangulator \
#     --database_path $PROJECT_PATH/database/database.db \
#     --image_path $PROJECT_PATH/images \
#     --input_path $PROJECT_PATH/sparse \
#     --output_path $PROJECT_PATH/triangulate

colmap image_undistorter \
    --image_path $PROJECT_PATH/images \
    --input_path $PROJECT_PATH/triangulate \
    --output_path $PROJECT_PATH/dense

colmap patch_match_stereo \
    --workspace_path $PROJECT_PATH/dense

colmap stereo_fusion \
    --workspace_path $PROJECT_PATH/dense \
    --output_path $PROJECT_PATH/dense/fused.ply