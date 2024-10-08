import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import igl
import torch
from torch.autograd import Variable
from torch import Tensor
from dataprocessing import isdf_sample
from dataprocessing import transform

def interpolate_np_arrays(start_array, end_array, num_steps):
    lin_int = np.linspace(0, 1, num=num_steps)[:, None]
    return start_array[None, :] * (1 - lin_int) + end_array[None, :] * lin_int


def viz(points, speeds, lidar_points=None, meshpath=None, camera_positions=None):
    #! visualize the mesh
    import trimesh
    if True:
        if not meshpath:
            meshpath = "datasets/igib-seqs/Beechwood_0_int_scene_mesh.obj"


        mesh = trimesh.load(meshpath)
        mesh.visual.face_colors = [200, 192, 207, 255]
        scene = trimesh.Scene(mesh)

        # Define line segments for X (red), Y (green), and Z (blue) axes
        axis_length = 1.0
        x_axis = trimesh.load_path(np.array([[0, 0, 0], [axis_length, 0, 0]]))
        y_axis = trimesh.load_path(np.array([[0, 0, 0], [0, axis_length, 0]]))
        z_axis = trimesh.load_path(np.array([[0, 0, 0], [0, 0, axis_length]]))
        x_axis.colors = [[255, 0, 0, 255]]
        y_axis.colors = [[0, 255, 0, 255]]
        z_axis.colors = [[0, 0, 255, 255]]
        scene.add_geometry([ x_axis, y_axis, z_axis])

        # Define camera positions
        if camera_positions is not None:
            cm_pc = trimesh.PointCloud(np.array(camera_positions), colors=[[255, 0, 0, 255]])
            scene.add_geometry([cm_pc])

        # Define a plane
        height = 0.3
        size = 7
        center = np.array([-5, 0, height])
        plane_vertices =[
            center + np.array([-size, -size, 0]),
            center + np.array([size, -size, 0]),
            center + np.array([size, size, 0]),
            center + np.array([-size, size, 0])
        ]
        plane_faces = [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 2],
            [0, 2, 1]
        ]
        plane_mesh = trimesh.Trimesh(plane_vertices, plane_faces, process=False)
        plane_mesh.visual.face_colors = [100, 100, 255, 100]

        # add points cloud
        if True:
            if points is not None and (speeds is not None):
                start_points = points[:,:3]
                start_speeds = speeds[:, 0]
                # start_colors = (np.outer(1.0 - start_speeds, [255, 0, 0, 50]) + 
                #     np.outer(start_speeds, [255, 255, 255, 50])).astype(np.uint8)
                colormap = plt.get_cmap('viridis')
                start_colors = colormap(start_speeds)

                end_points = points[:,3:6]
                end_speeds = speeds[:, 1]
                colormap = plt.get_cmap('viridis')
                end_colors = colormap(end_speeds)
                # end_colors = (np.outer(1.0 - end_speeds, [255, 0, 0, 80]) + 
                #     np.outer(end_speeds, [255, 255, 255, 80])).astype(np.uint8)

                point_cloud = trimesh.PointCloud(start_points, start_colors)
                point_cloud = trimesh.PointCloud(end_points, end_colors)
                scene.add_geometry([point_cloud])
        

        if lidar_points is not None:
            points = lidar_points
            point_cloud = trimesh.PointCloud(points, colors=[255, 0, 0, 255])

            scene.add_geometry([ x_axis, y_axis, z_axis, point_cloud])


        scene.show()

def unsigned_distance_without_bvh(triangles, query_points):
    # Assuming your tensors are called triangles and query_points
    triangles_np = triangles
    query_points_np = query_points

    # Flatten and get unique vertices
    vertices, inverse_indices = np.unique(triangles_np, axis=0, return_inverse=True)

    # Convert back the inverse indices to get faces
    faces = inverse_indices.reshape(-1, 3)


    # Compute the squared distance (Note: to get the actual distance, take the sqrt of the results)
    squared_d, closest_faces, closest_points = igl.point_mesh_squared_distance(query_points_np, vertices, faces)

    # distances would be the sqrt of squared_d
    unsigned_distance = np.sqrt(squared_d)

    return unsigned_distance

if __name__ == "__main__":
    # #! load dataset
    # dataset = isdf_sample.IGibsonDataset(root_dir='datasets/igib-seqs/results', traj_file='datasets/igib-seqs/traj.txt', config='datasets/igib-seqs/igibson_info.json', col_ext='.jpg')
    # dataset.get_speeds([0])
    # #! load lidar dataset 
    # # lidar_file = 'datasets/lidar-seqs/results/depth000000.npy'
    # # lidar_data = np.load(lidar_file)
    # matrix = np.loadtxt('datasets/lidar-seqs/traj.txt')
    # matrix = matrix.reshape(-1, 4, 4)
    # matrix0 = matrix[0]
    # matrix0pos = matrix0[:3, 3]
    
    if False:

        dataset = isdf_sample.IGibsonDataset(root_dir='datasets/lidar-seqs/results', traj_file='datasets/lidar-seqs/traj.txt', config='datasets/igib-seqs/igibson_info.json', use_lidar=True, col_ext='.jpg')
        frame_idx = 3
        points, speeds = dataset.get_speeds([frame_idx], minimum=0.05, maximum=0.3, num=50000)
        points = points.cpu().numpy()
        speeds = speeds.cpu().numpy()

        frame0 = dataset[frame_idx]
        depth = frame0['depth']
        depth_dir = frame0['depth_dirs']
        T = frame0['T']
        lidar_points = depth_dir * depth[:, None] + T[:3, 3]
        # print("depth", depth)
        # print("depth_dir", depth_dir)



        #!################## previous code ###################
        # FRAMES = range(0, 30)
        # # FRAMES = [0]
        # maximum = 0.5
        # minimum = 0.05

        # points, speeds = dataset.get_speeds(FRAMES, minimum, maximum, 100000, False)
        # points = points.cpu().numpy()
        # speeds = speeds.cpu().numpy()

    

    # points = np.load("mypoints.npy")
    # speeds = np.load("myspeeds.npy")

    # points = np.load("logtau/sampled_points_12_01.npy")
    # speeds = np.load("logtau/speed_12_01.npy")

    points = np.load("logtau/sampled_points_11_30.npy")
    speeds = np.load("logtau/speed_11_30.npy")

    meshpath = "mesh_z_up.obj"
    meshpath = "mesh_z_up_11_28.obj"
    meshpath = "mesh_scaled_11_29_15.obj" #xiao

    surf_pc = np.load("surf_pc.npy")
    num_samples = 10000

    camera_positions = np.array([[-0.3, -0.2, 0]])
    if True:
        ptsp = True 
        if ptsp:
            all_points = np.load("logtau/sampled_points_12_01.npy")
            all_points = all_points[:num_samples*30]
            all_speeds = None
            for k in range(30):
                speeds = np.load("logtau/batch_speeds_12_01_%s.npy"%(k))
                if all_speeds is None:
                    all_speeds = speeds
                else:
                    all_speeds = np.concatenate((all_speeds, speeds), axis=0)
            np.save("logtau/batch_speeds_12_01.npy", all_speeds)
            all_points = np.load("logtau/sampled_points_11_30.npy")
            # all_speeds = np.load("logtau/speed_11_30.npy")
            # all_points = np.load("logtau/sampled_points_12_01.npy")
            # all_speeds = np.load("logtau/speed_12_01.npy")
            all_speeds = np.load("logtau/batch_speeds_12_01.npy")
            all_points = all_points[:300000,:]
            all_speeds = all_speeds[:300000,:]
            
            all_points = np.load("mypoints.npy")
            all_speeds = np.load("myspeeds.npy")
            viz(all_points, all_speeds, None, meshpath, camera_positions)
        elif True:
            viz(None, None, None, meshpath, camera_positions)
        else:
            surf_pc = Variable(torch.from_numpy(surf_pc)).cuda()
            for k in range(30):
                start_points = points[k*num_samples:(k+1)*num_samples,:3]
                start_points = Variable(torch.from_numpy(start_points)).cuda()
                diff = start_points[:,None,:] - surf_pc[None,:,:]
                dists = diff.norm(dim=-1)
                dists, closest_ixs = dists.min(axis=-1)
                dists -= 0.01
                # add a ceiling and floor
                dist_ceil = torch.abs(start_points[:, 2]-0.12)
                dist_floor = torch.abs(start_points[:, 2]+0.12)
                dists = torch.min(torch.min(dists, dist_ceil), dist_floor)

                minimum = 0.005
                maximum = 0.05
                start_speed = torch.clip(dists, minimum, maximum)/maximum

                # torch.clip((dists - minimum) / (maximum - minimum), 0.1, 1)

                end_points = points[k*num_samples:(k+1)*num_samples,3:]
                end_points = Variable(torch.from_numpy(end_points)).cuda()
                diff = end_points[:,None,:] - surf_pc[None,:,:]
                dists = diff.norm(dim=-1)
                dists, closest_ixs = dists.min(axis=-1)
                dists -= 0.01
                dist_ceil = torch.abs(end_points[:, 2]-0.12)
                dist_floor = torch.abs(end_points[:, 2]+0.12)
                dists = torch.min(torch.min(dists, dist_ceil), dist_floor)

                minimum = 0.005
                maximum = 0.05
                end_speed = torch.clip(dists, minimum, maximum)/maximum

                speeds = torch.stack([start_speed, end_speed], dim=-1)
                print(speeds.shape)
                np.save("logtau/batch_speeds_12_01_%s.npy"%(k), speeds.cpu().numpy())

    if False:
        speeds1 = np.load("logtau/speed_11_30.npy")
        speeds2 = np.load("logtau/batch_speeds_12_01.npy")

        print(speeds1[:50])
        print(speeds2[:50])

    if False:
        points = np.load("logtau/sampled_points_11_30.npy")
        speeds = np.load("logtau/speed_11_30.npy")

        where_d = np.where((points[:, 2] < 0)&(points[:, 5] < 0))
        points = points[where_d]
        speeds = speeds[where_d]
        
        np.save("logtau/sampled_points_12_01.npy", points)
        np.save("logtau/speed_12_01.npy", speeds)

    if True: # plot the distribution of distance between start and end points
        all_points = np.load("mypoints.npy")
        all_speeds = np.load("myspeeds.npy")

        start_points = all_points[:,:3]
        end_points = all_points[:,3:]
        diff = start_points - end_points
        dists = np.linalg.norm(diff, axis=-1)
        import matplotlib.pyplot as plt
        plt.hist(dists, bins=100)
        plt.show()


