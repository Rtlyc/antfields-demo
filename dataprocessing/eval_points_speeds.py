import trimesh
import numpy as np
import torch
import bvh_distance_queries
import igl


def point_obstacle_distance(query_points, triangles_obs):
    query_points = query_points.unsqueeze(dim=0)
    #print(query_points.shape)
    bvh = bvh_distance_queries.BVH()
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    distances, closest_points, closest_faces, closest_bcs= bvh(triangles_obs, query_points)
    torch.cuda.synchronize()
    unsigned_distance = torch.sqrt(distances).squeeze()
    #print(closest_points.shape)
    return unsigned_distance

def point_append_list(X_list,Y_list, 
                      triangles_obs, numsamples, dim, offset, margin):
    
    OutsideSize = numsamples + 2
    WholeSize = 0

    while OutsideSize > 0:
        P  = torch.rand((8*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        dP = torch.rand((8*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        rL = (torch.rand((8*numsamples,1),dtype=torch.float32, device='cuda'))*np.sqrt(dim)
        nP = P + torch.nn.functional.normalize(dP,dim=1)*rL
        #nP = torch.rand((8*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        
        PointsInside = torch.all((nP <= 0.5),dim=1) & torch.all((nP >= -0.5),dim=1)
        

        x0 = P[PointsInside,:]
        x1 = nP[PointsInside,:]

        #print(x0.shape[0])
        if(x0.shape[0]<=1):
            continue
        #print(len(PointsOutside))
        

        obs_distance0 = point_obstacle_distance(x0, triangles_obs)
        where_d          =  (obs_distance0 > offset) & (obs_distance0 < margin)
        x0 = x0[where_d]
        x1 = x1[where_d]
        y0 = obs_distance0[where_d]

        obs_distance1 = point_obstacle_distance(x1, triangles_obs)
        
        y1 = obs_distance1

        print(x0.shape)
        #print(x1.shape)
        #print(y0.shape)
        #print(y1.shape)

        x = torch.cat((x0,x1),1)
        y = torch.cat((y0.unsqueeze(1),y1.unsqueeze(1)),1)

        X_list.append(x)
        Y_list.append(y)
        
        OutsideSize = OutsideSize - x.shape[0]
        WholeSize = WholeSize + x.shape[0]
        
        if(WholeSize > numsamples):
            break
    return X_list, Y_list

def sample_points_and_speeds(meshpath, numsamples, dim, minimum, maximum):
    offset = minimum
    margin = maximum
    v_obs, f_obs = igl.read_triangle_mesh(meshpath)
    v_obs = torch.tensor(v_obs, dtype=torch.float32, device='cuda')
    f_obs = torch.tensor(f_obs, dtype=torch.long, device='cuda')
    t_obs = v_obs[f_obs].unsqueeze(dim=0)
    
    X_list = []
    Y_list = []
    #N_list = []
    
    X_list, Y_list = point_append_list(X_list,Y_list,  t_obs, numsamples, dim, offset, margin)
   
    X = torch.cat(X_list,0)[:numsamples]
    Y = torch.cat(Y_list,0)[:numsamples]

    sampled_points = X.detach().cpu().numpy()
    distance = Y.detach().cpu().numpy()
    #normal = N.detach().cpu().numpy()
    
    distance0 = distance[:,0]
    distance1 = distance[:,1]
    speed  = np.zeros((distance.shape[0],2))
    speed[:,0] = np.clip(distance0 , a_min = offset, a_max = margin)/margin
    speed[:,1] = np.clip(distance1 , a_min = offset, a_max = margin)/margin

    if False:
        # visualize(meshpath, sampled_points[:, :3], speed[:, 0])
        visualize(meshpath, sampled_points[:, 3:6], speed[:, 1])

    if True:
        np.save("sampled_points.npy", sampled_points)
        np.save("speed.npy", speed)





def visualize(meshpath, points, speeds):
    mesh = trimesh.load(meshpath)

    # point cloud
    pc = trimesh.PointCloud(points)

    # colors
    colors = trimesh.visual.interpolate(speeds, 'viridis')
    pc.colors = colors

    # scene
    scene = trimesh.Scene([mesh, pc])

    # show
    scene.show()

if __name__ == '__main__':
    meshpath = "mesh_scaled_11_29_15.obj"

    if False:
        points = np.random.rand(100, 3)
        speeds = np.random.rand(100)

        points = np.load("sampled_points.npy")[:10000, :3]
        speeds = np.load("speed.npy")[:10000, 0]

        visualize(meshpath, points, speeds)
    
    if True:
        maximum = 0.05
        minimum = 0.005
        sample_points_and_speeds(meshpath, 500000, 3, minimum, maximum)
