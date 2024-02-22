import os

SFM_COMMAND="colmap automatic_reconstructor \
    --workspace_path {} \
    --image_path {}"

def sfm(project_path):
    os.system(SFM_COMMAND.format(SFM_COMMAND, project_path))
