import numpy as np
import argparse
from models import model_igibson as md

# Define modes
EXPLORATION = 1 
READ_FROM_COOKED_DATA = 2


# Define the model path
scale_factor = 10
modelPath = './Experiments'

def main():
    parser = argparse.ArgumentParser(description='Train the model with or without exploration.')
    parser.add_argument('--no_explore', action='store_true', help='Disable exploration mode.')
    
    args = parser.parse_args()

    # Set the mode based on the --no_explore flag
    if args.no_explore:
        mode = READ_FROM_COOKED_DATA
    else:
        mode = EXPLORATION  # Default to exploration mode

    renderer = None
    if mode in [EXPLORATION]:
        from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
        meshpath = "data/mesh.obj"
        
        renderer = MeshRenderer(width=1200, height=680)
        renderer.load_object(meshpath, scale=np.array([1, 1, 1]) * scale_factor)
        renderer.add_instance_group([0])
        camera_pose = np.array([0, 0, 1])    
        view_direction = np.array([1, 0, 0])
        renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 0])
        renderer.set_fov(90)

    # Initialize and train the model
    model = md.Model(modelPath, 3, scale_factor, mode, renderer, device='cuda:0')
    model.train()

if __name__ == '__main__':
    main()
