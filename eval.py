import numpy as np
import torch 
import bvh_distance_queries
import igl

from models import model_igibson as md
import time

def check_collision(query_points, triangles, scale=1):
    query_points /= scale 
    query_points = torch.tensor(query_points, dtype=torch.float32, device='cuda')
    
    #! interpolate points
    new_query_points = []
    interpolation_steps = 10
    for i in range(len(query_points) - 1):
        start_point = query_points[i]
        end_point = query_points[i+1]
        for t in range(interpolation_steps):
            alpha = t/interpolation_steps
            interpolated_point = (1-alpha)*start_point + alpha*end_point
            new_query_points.append(interpolated_point)
    new_query_points = torch.vstack(new_query_points)
    query_points = new_query_points
    
    query_points = query_points.unsqueeze(dim=0)
    #print(query_points.shape)
    bvh = bvh_distance_queries.BVH()
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    distances, closest_points, closest_faces, closest_bcs= bvh(triangles, query_points)
    torch.cuda.synchronize()
    #unsigned_distance = abs()
    #print(distances.shape)
    unsigned_distance = torch.sqrt(distances).squeeze()
    unsigned_distance = unsigned_distance.detach().cpu().numpy()
    
    if np.min(unsigned_distance)<=0.001:
        return False
    else:
        return True
    
def calculate_trajectory_lengths(trajectories):
    # This function calculates the length of each trajectory.
    # trajectories: a numpy array of shape (N, 64, 3)
    nt = trajectories.shape[0]
    lengths = np.zeros(nt)

    for i in range(nt):
        for j in range(1, trajectories.shape[1]):
            lengths[i] += np.linalg.norm(trajectories[i, j, :] - trajectories[i, j-1, :])
    
    return lengths


def our_method_eval(meshpath, modelPath, datapath):
    v, f = igl.read_triangle_mesh(meshpath)
    vertices=v
    faces=f

    vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')
    faces = torch.tensor(faces, dtype=torch.long, device='cuda')
    triangles = vertices[faces].unsqueeze(dim=0)


    # Load the data
    scale_factor = 10 
    mode = 2
    renderer = None
    model    = md.Model('./Experiments', 3, scale_factor, mode, renderer, device='cuda:0')
    model.load(modelPath)

    #TODO: write start and goal random samples
    explored_data = np.load(datapath, allow_pickle=True).reshape(-1, 8)
    potential_points = explored_data[:, 3:6]
    valid_mask = explored_data[:, 7] > 0.95
    valid_points = potential_points[valid_mask]
    N = 100
    rand_idx = np.random.choice(len(valid_points), 2*N)
    start_points = valid_points[rand_idx][:N]
    end_points = valid_points[rand_idx][N:]
    

    evalant_times = []
    fail_trajs = []
    succ_trajs = []
    succ_lens = []
    for i in range(N):
        src = start_points[i]
        tar = end_points[i]
        start_time = time.time()
        cur_trajectory = model.predict_trajectory(src, tar, step_size=0.05, tol=0.05)
        end_time = time.time()
        
        if check_collision(cur_trajectory, triangles):
            evalant_times.append(end_time - start_time)
            lengths = calculate_trajectory_lengths(cur_trajectory[None,])
            succ_lens.append(lengths[0])
            succ_trajs.append(cur_trajectory)
        else:
            fail_trajs.append(cur_trajectory)

    success_lens = np.array(succ_lens)
    success_rate = len(success_lens) / N
    evalant_times = np.array(evalant_times)

    print('Success rate: ', success_rate)
    print('Average time: ', np.mean(evalant_times))
    print('Average trajectory length: ', np.mean(success_lens))

    #! vis
    if True:
        import trimesh 
        mesh = trimesh.load_mesh(meshpath)
        M = 5
        trajs = fail_trajs
        r = np.random.choice(len(trajs), M)
        for i in range(M):
            scene = trimesh.Scene(mesh)
            pc = trimesh.PointCloud(trajs[r[i]])
            scene.add_geometry([pc])
            scene.show()

    # return trajectories, fail_trajs

if __name__ == '__main__':
    meshpath = './data/mesh.obj'
    modelpath = './Experiments/10_09_14_50/Model_Epoch_04000_ValLoss_3.739335e-02.pt'
    datapath = './data/explored_data.npy'

    our_method_eval(meshpath, modelpath, datapath)
