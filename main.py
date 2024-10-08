import numpy as np
#! train model
# from models import model_igibson_online_continual_newlog as md
# from models import model_igibson_online_continual_newlog_12_4 as md
# from models import model_igibson_online_continual_newlog_12_4_symop as md
from models import model_igibson as md


modelPath = './Experiments'

EXPLORATION = 1 
READ_FROM_DEPTH = 2 
READ_FROM_TURTLEBOT = 3
READ_FROM_COOKED_DATA = 4
TURTLEBOT_EXPLORATION = 5
# TURTLEBOT_POINTCLOUD = 6

# mode = READ_FROM_TURTLEBOT
# scale_factor = 2.5 #7.5
mode = READ_FROM_COOKED_DATA
mode = EXPLORATION 
scale_factor = 10
# mode = READ_FROM_DEPTH

renderer = None
if mode in [EXPLORATION, READ_FROM_DEPTH]:
    from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
    meshpath = "data/mesh.obj"
    
    # meshpath = "gibson/Angiola/mesh_z_up_scaled.obj"
    # meshpath = "gibson/Artois/mesh_z_up_scaled.obj"
    # meshpath = "gibson/Avonia/mesh_z_up_scaled.obj"
    # meshpath = "gibson/Barboursville/mesh_z_up_scaled.obj"
    # meshpath = "gibson/Beach/mesh_z_up_scaled.obj"
    # meshpath = "gibson/Branford/mesh_z_up_scaled.obj"
    # meshpath = "gibson/Bolton/mesh_z_up_scaled.obj"
    renderer = MeshRenderer(width=1200, height=680)
    renderer.load_object(meshpath, scale=np.array([1, 1, 1])*scale_factor)
    renderer.add_instance_group([0])
    camera_pose = np.array([0, 0, 1])    
    view_direction = np.array([1, 0, 0])
    renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 0])
    renderer.set_fov(90)


model    = md.Model(modelPath, 3, scale_factor, mode, renderer, device='cuda:0')

model.train()