import numpy as np
from torch.utils.data import Dataset 
import torch 
from scipy.spatial.transform import Rotation as R 
import os, cv2, json
from torchvision import transforms
from dataprocessing import transform, data_pc_validate
import matplotlib.pyplot as plt
import igl

"""
Lidar Parameters From iGibson:
vertical_fov: -15, 15
vertical_resolution: 16
horizontal_fov: -45, 45
horizontal_resolution: 468
"""

class IGibsonDataset(Dataset):
    def __init__(
        self,
        root_dir,
        traj_file=None,
        config=None,
        rgb_transform=None,
        depth_transform=None,
        noisy_depth=False,
        use_lidar=False,
        col_ext=".png",     
    ):
        self.Ts = None
        if traj_file is not None:
            self.Ts = np.loadtxt(traj_file).reshape(-1, 4, 4)
        self.root_dir = root_dir
        self.config_file = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with open(self.config_file) as f:
            configs = json.load(f)
        self.fx = configs["dataset"]["camera"]["fx"]
        self.fy = configs["dataset"]["camera"]["fy"]
        self.cx = configs["dataset"]["camera"]["cx"]
        self.cy = configs["dataset"]["camera"]["cy"]
        self.W = configs["dataset"]["camera"]["w"]
        self.H = configs["dataset"]["camera"]["h"]
        depth_scale = configs["dataset"]["depth_scale"]
        inv_depth_scale = 1. / depth_scale
        self.min_depth = configs["sample"]["depth_range"][0]
        self.max_depth = configs["sample"]["depth_range"][1]
        self.n_rays = configs["sample"]["n_rays"]
        self.n_rays = 5000 #5000
        self.n_strat_samples = configs["sample"]["n_strat_samples"]
        self.n_surf_samples = configs["sample"]["n_surf_samples"]
        self.dist_behind_surf = configs["sample"]["dist_behind_surf"]  

        self.depth_transform = transforms.Compose([DepthScale(inv_depth_scale), DepthFilter(self.max_depth)])
        self.rgb_transform = rgb_transform
        self.col_ext = col_ext 
        self.noisy_depth = noisy_depth

        self.up_vec = np.array([0., 1., 0.])
        self.use_lidar = use_lidar
        if use_lidar:
            pass 
        else:
            self.dirs_C = transform.ray_dirs_C(1,self.H,self.W,self.fx,self.fy,self.cx,self.cy,self.device,depth_type="z")

        print("dataset length: ", len(self))

    def __len__(self):
        return self.Ts.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        s = f"{idx:06}"
        depth_file = os.path.join(self.root_dir,"depth" + s + ".npy")
        
        rgb_file = os.path.join(self.root_dir,"frame" + s + self.col_ext)

        depth = np.load(depth_file)
        image = cv2.imread(rgb_file)

        T = None 
        if self.Ts is not None:
            T = self.Ts[idx]

        if self.use_lidar:
            depth_dirs, depth = depth, np.linalg.norm(depth, axis=-1)
            rotation_matrix_90_z = np.array([
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]
            ])
            depth_dirs = depth_dirs.dot(rotation_matrix_90_z.T)/depth[:, None]
             
            
        
        sample = {"image": image, "depth": depth, "T": T}

        if self.use_lidar:
            sample["depth_dirs"] = depth_dirs

        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])

        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])


        return sample
    
    def get_speeds(self, idx_list, minimum=0.07, maximum=0.3, num=10000, is_gt_speed=False):
        all_points = []
        all_bounds = []
        device = self.device
        for idx in idx_list:
            sub_dataset = self[idx]
            depth_np = sub_dataset["depth"][None, ...]
            T_np = sub_dataset["T"][None, ...]

            depth = torch.from_numpy(depth_np).float().to(device)
            T = torch.from_numpy(T_np).float().to(device)
            # pc = transform.pointcloud_from_depth_torch(depth[0], self.fx, self.fy, self.cx, self.cy)

            if self.use_lidar:
                depth_dirs_np = sub_dataset["depth_dirs"][None, ...]
                depth_dirs = torch.from_numpy(depth_dirs_np).float().to(device)
                sample_pts = sample_lidar_points(depth, T, self.n_rays, depth_dirs, self.dist_behind_surf, self.n_strat_samples, self.n_surf_samples, self.min_depth, self.max_depth, device=device)
            else:
                sample_pts = sample_points(depth, T, self.n_rays, self.dirs_C, self.dist_behind_surf, self.n_strat_samples, self.n_surf_samples, self.min_depth, self.max_depth, None, device=device)

            bound = bounds_pc(sample_pts["pc"], sample_pts["z_vals"],
                              sample_pts["surf_pc"], sample_pts["depth_sample"])
            all_points.append(sample_pts["pc"])
            all_bounds.append(bound)
        
        pc = torch.cat(all_points, dim=0)
        bounds = torch.cat(all_bounds, dim=0)
        pc = pc.view(-1, 3)
        bounds = bounds.view(-1, 1)

        if is_gt_speed: #ground truth
            bounds = self.get_gt_bounds("datasets/igib-seqs/Beechwood_0_int_scene_mesh.obj", pc)
        
        speeds = torch.clip(bounds, minimum, maximum)/maximum

        valid_indices = torch.where((speeds <= 1) & (speeds > 0))[0] 
        if num <= len(valid_indices):
            # Select without replacement if num is less than or equal to the size of valid_indices
            start_indices = valid_indices[torch.randperm(len(valid_indices))[:num]]
            end_indices = valid_indices[torch.randperm(len(valid_indices))[:num]]
        else:
            # Select with replacement if num is greater than the size of valid_indices
            rand_indices = torch.randint(0, len(valid_indices), (num,))
            start_indices = valid_indices[rand_indices]

            rand_indices = torch.randint(0, len(valid_indices), (num,))
            end_indices = valid_indices[rand_indices]

        end_indices = torch.randint(0, pc.shape[0], (num,))
        x0 = pc[start_indices]
        x1 = pc[end_indices]
        x = torch.cat((x0, x1), dim=1)
        y = torch.cat((speeds[start_indices], speeds[end_indices]), dim=1)
        z = torch.cat((bounds[start_indices], bounds[end_indices]), dim=1)

        return x,y,z

    def get_speeds2(self, idx_list, minimum=0.07, maximum=0.3, num=10000, is_gt_speed=False):
        all_points = []
        all_bounds = []
        device = self.device
        for idx in idx_list:
            sub_dataset = self[idx]
            depth_np = sub_dataset["depth"][None, ...]
            T_np = sub_dataset["T"][None, ...]

            depth = torch.from_numpy(depth_np).float().to(device)
            T = torch.from_numpy(T_np).float().to(device)
            # pc = transform.pointcloud_from_depth_torch(depth[0], self.fx, self.fy, self.cx, self.cy)

            if self.use_lidar:
                depth_dirs_np = sub_dataset["depth_dirs"][None, ...]
                depth_dirs = torch.from_numpy(depth_dirs_np).float().to(device)
                sample_pts = sample_lidar_points(depth, T, self.n_rays, depth_dirs, self.dist_behind_surf, self.n_strat_samples, self.n_surf_samples, self.min_depth, self.max_depth, device=device)
            else:
                sample_pts = sample_points(depth, T, self.n_rays, self.dirs_C, self.dist_behind_surf, self.n_strat_samples, self.n_surf_samples, self.min_depth, self.max_depth, None, device=device)

        pc = torch.cat(all_points, dim=0)
        bounds = torch.cat(all_bounds, dim=0)
        pc = pc.view(-1, 3)
        bounds = bounds.view(-1, 1)

        if is_gt_speed: #ground truth
            bounds = self.get_gt_bounds("datasets/igib-seqs/Beechwood_0_int_scene_mesh.obj", pc)
        
        speeds = torch.clip(bounds, minimum, maximum)/maximum
        #'''

        #speeds = torch.clip(bounds, minimum, maximum)/maximum

        valid_indices = torch.where((bounds < maximum) & (bounds > minimum))[0] 

        x0 = points[valid_indices]
        y0 = torch.clip(bounds[valid_indices], minimum, maximum)/maximum

        dP = torch.rand((x0.shape[0],3),dtype=torch.float32, device='cuda')-0.5
        rL = (torch.rand((x0.shape[0],1),dtype=torch.float32, device='cuda'))*1.5
        x1 = x0 + torch.nn.functional.normalize(dP,dim=1)*rL

        position = torch.tensor(position).cuda()

        ray0 = x1 - position*scale_factor
        ray1 = sample_pts["surf_pc"] - position*scale_factor
        norm0 = ray0.norm(dim=-1)
        norm1 = ray1.norm(dim=-1)
        dot = (ray0/norm0.unsqueeze(1))@(ray1/norm1.unsqueeze(1)).T
        dot, closest_ixs1 = dot.max(axis=-1)
        # print(dot)

        valid_indices1 = torch.where( (norm0 < norm1[closest_ixs1])&(dot>0.995)&(norm0<2.5))[0]
        #print(dot)
        x0 = x0[valid_indices1]
        x1 = x1[valid_indices1]
        y0 = y0[valid_indices1]

        diff1 = x1.unsqueeze(1) - sample_pts["surf_pc"] 
        dists1 = diff1.norm(dim=-1)
        dists1, closest_ixs1 = dists1.min(axis=-1)
        # dists1 -= 0.02
        y1 = torch.clip(dists1, minimum, maximum)/maximum

        # print(closest_ixs1)

        points = torch.cat((x0, x1), dim=1)
        speeds = torch.cat((y0, y1.unsqueeze(1)), dim=1)
        bounds = torch.cat((bounds[valid_indices][valid_indices1], dists1.unsqueeze(1)), dim=1)

        points /= scale_factor
        bounds /= scale_factor
        #if False and is_gt_speed: #ground truth
        #    bounds = self.get_gt_bounds("datasets/igib-seqs/Beechwood_0_int_scene_mesh.obj", pc)
        # print(bounds)
        #points, speeds, bounds = sample_points_and_speeds_from_bounds(pc, bounds, minimum=minimum, maximum=maximum, num=num)
        
        return points[0:5000], speeds[0:5000], bounds[0:5000]

class TurtleBotDataset(Dataset):
    def __init__(
        self,
        root_dir,
        traj_file=None,
        config=None,
        rgb_transform=None,
        depth_transform=None,
        noisy_depth=False,
        use_lidar=False,
        col_ext=".png",     
    ):
        self.root_dir = root_dir
        self.config_file = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with open(self.config_file) as f:
            configs = json.load(f)
        self.fx = configs["dataset"]["camera"]["fx"]
        self.fy = configs["dataset"]["camera"]["fy"]
        self.cx = configs["dataset"]["camera"]["cx"]
        self.cy = configs["dataset"]["camera"]["cy"]
        self.W = configs["dataset"]["camera"]["w"]
        self.H = configs["dataset"]["camera"]["h"]
        depth_scale = configs["dataset"]["depth_scale"]
        inv_depth_scale = 1. / depth_scale
        self.min_depth = configs["sample"]["depth_range"][0]
        self.max_depth = configs["sample"]["depth_range"][1]
        self.n_rays = configs["sample"]["n_rays"]
        self.n_rays = 5000 #5000
        self.n_strat_samples = configs["sample"]["n_strat_samples"]
        self.n_surf_samples = configs["sample"]["n_surf_samples"]
        self.dist_behind_surf = configs["sample"]["dist_behind_surf"]  

        self.depth_transform = transforms.Compose([DepthScale(inv_depth_scale), DepthFilter(self.max_depth)])
        self.rgb_transform = rgb_transform
        self.col_ext = col_ext 
        self.noisy_depth = noisy_depth

        self.up_vec = np.array([0., 1., 0.])
        self.use_lidar = use_lidar
        self.dirs_C = transform.ray_dirs_C(1,self.H,self.W,self.fx,self.fy,self.cx,self.cy,self.device,depth_type="z")

        self.depthfiles, self.posefiles, self.rgbfiles = data_pc_validate.get_filenames_for_tbot(self.root_dir)

        self.Ts = []
        self.skip = 5
        for idx in range(0, len(self.posefiles), self.skip):
            pose = np.load(self.posefiles[idx], allow_pickle=True)
            pose = data_pc_validate.convert_transform_odom_to_cam(pose)
            temp = pose.copy()
            pose_pred = data_pc_validate.convert_cam_xyz_to_odom_xyz(temp)
            pose[0, 3] -= 3
            self.Ts.append(pose)
        self.Ts = np.array(self.Ts)

        print("dataset length: ", len(self))

    def __len__(self):
        return len(self.Ts)
    
    def __getitem__(self, idx):
        T_cur = self.Ts[idx]
        idx *= self.skip
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        s = f"{idx:06}"
        depth = np.load(self.depthfiles[idx], allow_pickle=True).astype(np.int32)
        # print("depth: ", depth.shape) # (680, 1200)
        # print("depth: ", depth)
        # load the pose npy file
        # pose = np.load(self.posefiles[idx])
        # pose = data_pc_validate.convert_transform_odom_to_cam(pose)
        # print("pose: ", pose.shape) # (4, 4)
        # print("pose: ", pose)
        # load the rgb image
        rgb = cv2.imread(self.rgbfiles[idx])
             
            
        
        sample = {"image": rgb, "depth": depth, "T": T_cur}

        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])

        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])

        return sample
    
    def get_speeds(self, idx_list, minimum=0.07, maximum=0.3, num=10000):
        # original data
        all_points = []
        all_bounds = []
        device = self.device
        #minimum = 0.1
        #maximum = 0.5
        surf_pc = []
        for idx in idx_list:
            sub_dataset = self[idx]
            depth_np = sub_dataset["depth"][None, ...]
            T_np = sub_dataset["T"][None, ...]

            depth = torch.from_numpy(depth_np).float().to(device)
            T = torch.from_numpy(T_np).float().to(device)
            # pc = transform.pointcloud_from_depth_torch(depth[0], self.fx, self.fy, self.cx, self.cy)

            sample_pts = sample_points(depth, T, self.n_rays, self.dirs_C, self.dist_behind_surf, self.n_strat_samples, self.n_surf_samples, self.min_depth, self.max_depth, None, device=device)

            bound = bounds_pc(sample_pts["pc"], sample_pts["z_vals"],
                              sample_pts["surf_pc"], sample_pts["depth_sample"])
            all_points.append(sample_pts["pc"])
            all_bounds.append(bound)
            surf_pc.append(sample_pts["surf_pc"])
        
        surf_pc = torch.cat(surf_pc, dim=0)
        pc = torch.cat(all_points, dim=0)
        bounds = torch.cat(all_bounds, dim=0)
        pc = pc.view(-1, 3)
        bounds = bounds.view(-1, 1)
        bounds -= 0.15
        addfilter = False 
        if addfilter:
            above_ground = (pc[:,2]>0.1) & (pc[:,2]<0.4)
            pc = pc[above_ground]
            bounds = bounds[above_ground]

        speeds = torch.clip(bounds, minimum, maximum)/maximum

        valid_indices = torch.where((speeds <= 1) & (speeds > 0))[0] 
        if num <= len(valid_indices):
            # Select without replacement if num is less than or equal to the size of valid_indices
            start_indices = valid_indices[torch.randperm(len(valid_indices))[:num]]
            end_indices = valid_indices[torch.randperm(len(valid_indices))[:num]]
        else:
            # Select with replacement if num is greater than the size of valid_indices
            rand_indices = torch.randint(0, len(valid_indices), (num,))
            start_indices = valid_indices[rand_indices]

            rand_indices = torch.randint(0, len(valid_indices), (num,))
            end_indices = valid_indices[rand_indices]

        end_indices = torch.randint(0, pc.shape[0], (num,))
        x0 = pc[start_indices]
        x1 = pc[end_indices]
        x = torch.cat((x0, x1), dim=1)
        y = torch.cat((speeds[start_indices], speeds[end_indices]), dim=1)
        z = torch.cat((bounds[start_indices], bounds[end_indices]), dim=1)

        return x,y,z,surf_pc 

        
# class ReplicaDataset(Dataset):
#     def __init__(
#         self,
#         root_dir,
#         traj_file=None,
#         config=None,
#         rgb_transform=None,
#         depth_transform=None,
#         noisy_depth=False,
#         col_ext=".png",
#     ):
#         self.Ts = None
#         if traj_file is not None:
#             self.Ts = np.loadtxt(traj_file).reshape(-1, 4, 4)
#         self.root_dir = root_dir
#         self.config_file = config

#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#         with open(self.config_file) as f:
#             configs = json.load(f)
#         self.fx = configs["dataset"]["camera"]["fx"]
#         self.fy = configs["dataset"]["camera"]["fy"]
#         self.cx = configs["dataset"]["camera"]["cx"]
#         self.cy = configs["dataset"]["camera"]["cy"]
#         self.W = configs["dataset"]["camera"]["w"]
#         self.H = configs["dataset"]["camera"]["h"]
#         depth_scale = configs["dataset"]["depth_scale"]
#         inv_depth_scale = 1. / depth_scale
#         self.min_depth = configs["sample"]["depth_range"][0]
#         self.max_depth = configs["sample"]["depth_range"][1]
#         self.n_rays = configs["sample"]["n_rays"]
#         self.n_rays = 5000 #5000
#         self.n_strat_samples = configs["sample"]["n_strat_samples"]
#         self.n_surf_samples = configs["sample"]["n_surf_samples"]
#         self.dist_behind_surf = configs["sample"]["dist_behind_surf"]  

#         self.depth_transform = transforms.Compose([DepthScale(inv_depth_scale), DepthFilter(self.max_depth)])
#         self.rgb_transform = rgb_transform
#         self.col_ext = col_ext 
#         self.noisy_depth = noisy_depth

#         self.up_vec = np.array([0., 1., 0.])
#         self.dirs_C = transform.ray_dirs_C(1,self.H,self.W,self.fx,self.fy,self.cx,self.cy,self.device,depth_type="z")

#         print("dataset length: ", len(self))

#     def __len__(self):
#         return self.Ts.shape[0]
    
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
        
#         s = f"{idx:06}"
#         if self.noisy_depth:
#             depth_file = os.path.join(self.root_dir,"ndepth" + s + ".png")
#         else:
#             depth_file = os.path.join(self.root_dir,"depth" + s + ".png")
        
#         rgb_file = os.path.join(self.root_dir,"frame" + s + self.col_ext)

#         depth = cv2.imread(depth_file, -1)
#         depth_vis = cv2.imread(depth_file)
#         image = cv2.imread(rgb_file)

#         T = None 
#         if self.Ts is not None:
#             T = self.Ts[idx]
        
#         sample = {"image": image, "depth": depth, "T": T, "depth_vis": depth_vis}

#         if self.rgb_transform:
#             sample["image"] = self.rgb_transform(sample["image"])

#         if self.depth_transform:
#             sample["depth"] = self.depth_transform(sample["depth"])


#         return sample

#     def get_speeds(self, idx_list, minimum=0.07, maximum=0.3, num=10000, is_gt_speed=False, scale=1.0):
#         all_points = []
#         all_bounds = []
#         device = self.device
#         for idx in idx_list:
#             sub_dataset = self[idx]
#             im_np = sub_dataset["image"][None, ...]
#             depth_np = sub_dataset["depth"][None, ...]
#             T_np = sub_dataset["T"][None, ...]
#             depth_vis_np = sub_dataset["depth_vis"][None, ...]


#             im = torch.from_numpy(im_np).float().to(device)/255.
#             depth = torch.from_numpy(depth_np).float().to(device)
#             T = torch.from_numpy(T_np).float().to(device)
#             pc = transform.pointcloud_from_depth_torch(depth[0], self.fx, self.fy, self.cx, self.cy)
#             normals = transform.estimate_pointcloud_normals(pc)
#             norm_batch = normals[None,:]
#             norm_batch = None

#             # print(dirs_C.shape)
#             sample_pts = sample_points(depth, T, self.n_rays, self.dirs_C, self.dist_behind_surf, self.n_strat_samples, self.n_surf_samples, self.min_depth, self.max_depth, norm_batch, device=device)

#             bound = bounds_pc(sample_pts["pc"], sample_pts["z_vals"], sample_pts["depth_sample"])
#             all_points.append(sample_pts["pc"])
#             all_bounds.append(bound.cpu())

#         pc = torch.cat(all_points, dim=0)
#         bounds = torch.cat(all_bounds, dim=0)
#         pc = pc.view(-1, 3)
#         bounds = bounds.view(-1, 1)

#         if is_gt_speed: #ground truth
#             bounds = self.get_gt_bounds("datasets/isdf-seqs/mesh.obj", pc)
        
#         # minimum = self.min_depth
#         speeds = torch.clip((bounds - minimum) / (maximum - minimum), 0, 1)

#         valid_indices = torch.where((speeds < 1) & (speeds > 0))[0]
#         if num <= len(valid_indices):
#             # Select without replacement if num is less than or equal to the size of valid_indices
#             start_indices = valid_indices[torch.randperm(len(valid_indices))[:num]]
#         else:
#             # Select with replacement if num is greater than the size of valid_indices
#             rand_indices = torch.randint(0, len(valid_indices), (num,))
#             start_indices = valid_indices[rand_indices]

#         end_indices = torch.randint(0, pc.shape[0], (num,))
#         x0 = pc[start_indices]
#         x1 = pc[end_indices]
#         x = torch.cat((x0, x1), dim=1)
#         y = torch.cat((speeds[start_indices], speeds[end_indices]), dim=1)

#         #! scaling
#         x = x * scale
#         y = y 

#         # import matplotlib.pyplot as plt
#         # plt.scatter(all_bounds, all_speeds)
#         # plt.xlabel("ground truth distance")
#         # plt.ylabel("speed")
#         # plt.show()
#         # temp = speeds[start_indices]
#         # plt.hist(temp.cpu().numpy(), bins=100)
#         # plt.title("speed distribution")
#         # plt.show()
#         return x, y
    
#     def get_gt_bounds(self, meshpath, x):
#         device = x.device
#         x_np = x.cpu().numpy()

#         v, f = igl.read_triangle_mesh(meshpath)
#         t_obs = v[f].reshape(-1, 3)
#         y = unsigned_distance_without_bvh(t_obs, x_np)
#         y = torch.from_numpy(y).float().to(device).unsqueeze(1)
#         return y
        

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

class DepthScale(object):
    """Scale depth image to meters"""

    def __init__(self, scale):
        self.scale = scale 

    def __call__(self, depth):
        depth = depth.astype(np.float32)
        return depth * self.scale

class DepthFilter(object):
    """scale depth to meters"""

    def __init__(self, max_depth):
        self.max_depth = max_depth 

    def __call__(self, depth):
        far_mask = depth > self.max_depth 
        depth[far_mask] = 0
        return depth 
    
def get_batch_data(depth_batch, T_WC_batch, dirs_C, indices_b, indices_h, indices_w, norm_batch=None):
    """
    Get depth, ray direction and pose for the sampled pixels.
    Only render where depth is valid
    """
    depth_sample = depth_batch[indices_b, indices_h, indices_w].view(-1)
    mask_valid_depth = depth_sample != 0

    norm_sample = None
    if norm_batch is not None:
        norm_sample = norm_batch[indices_b, indices_h, indices_w,:].view(-1, 3)
        mask_invalid_norm = torch.isnan(norm_sample[...,0])
        mask_valid_depth = torch.logical_and(mask_valid_depth, ~mask_invalid_norm)
        #? norm_sample
        norm_sample = norm_sample[mask_valid_depth]

    depth_sample = depth_sample[mask_valid_depth]
    indices_b = indices_b[mask_valid_depth]
    indices_h = indices_h[mask_valid_depth]
    indices_w = indices_w[mask_valid_depth]
    T_WC_sample = T_WC_batch[indices_b]
    dirs_C_sample = dirs_C[0, indices_h, indices_w, :].view(-1, 3)

    return dirs_C_sample, depth_sample, T_WC_sample,indices_b, indices_h, indices_w

def stratified_sample(
    min_depth,
    max_depth,
    n_rays,
    device,
    n_stratified_samples,
    bin_length=None,
):
    """
    Random samples between min and max depth
    One sample from within each bin.

    If n_stratified_samples is passed then use fixed number of bins,
    else if bin_length is passed use fixed bin size.
    """
    if n_stratified_samples is not None:  # fixed number of bins
        n_bins = n_stratified_samples
        if isinstance(max_depth, torch.Tensor):
            sample_range = (max_depth - min_depth)[:, None]
            bin_limits = torch.linspace(
                0, 1, n_bins + 1,
                device=device)[None, :]
            bin_limits = bin_limits.repeat(n_rays, 1) * sample_range
            if isinstance(min_depth, torch.Tensor):
                bin_limits = bin_limits + min_depth[:, None]
            else:
                bin_limits = bin_limits + min_depth
            bin_length = sample_range / (n_bins)
        else:
            bin_limits = torch.linspace(
                min_depth,
                max_depth,
                n_bins + 1,
                device=device,
            )[None, :]
            bin_length = (max_depth - min_depth) / (n_bins)

    elif bin_length is not None:  # fixed size of bins
        bin_limits = torch.arange(
            min_depth,
            max_depth,
            bin_length,
            device=device,
        )[None, :]
        n_bins = bin_limits.size(1) - 1

    increments = torch.rand(n_rays, n_bins, device=device) * bin_length
    # increments = 0.5 * torch.ones(n_rays, n_bins, device=device) * bin_length
    lower_limits = bin_limits[..., :-1]
    z_vals = lower_limits + increments

    return z_vals

def sample_along_rays(
    T_WC,
    min_depth,
    max_depth,
    n_stratified_samples,
    n_surf_samples,
    dirs_C,
    gt_depth
):
    #rays in world frame    
    origins, dirs_W = transform.origin_dirs_W(T_WC, dirs_C)

    origins = origins.view(-1, 3)
    dirs_W = dirs_W.view(-1, 3)
    n_rays = dirs_W.shape[0]
    
    #TODO: need to check and read stratified samples and surface samples
    #stratified samples along rays
    z_vals = stratified_sample(min_depth, max_depth, n_rays, T_WC.device, n_stratified_samples)

    # if gt_depth is given, first sample at surface then around surface
    if gt_depth is not None and n_surf_samples > 0:
        surface_z_vals = gt_depth
        offsets = torch.normal(
            torch.zeros(gt_depth.shape[0], n_surf_samples - 1), 0.1
        ).to(z_vals.device)
        near_surf_z_vals = gt_depth[:, None] + offsets
        if not isinstance(min_depth, torch.Tensor):
            min_depth = torch.full(near_surf_z_vals.shape, min_depth).to(
                z_vals.device)[..., 0]
        near_surf_z_vals = torch.clamp(
            near_surf_z_vals,
            min_depth[:, None],
            max_depth[:, None],
        )
        z_vals = torch.cat(
            (surface_z_vals[:, None], near_surf_z_vals, z_vals), dim=1)

    # point cloud of 3d sample locations
    pc = origins[:, None, :] + (dirs_W[:, None, :] * z_vals[:, :, None])
    return pc, z_vals

def sample_along_lidar_rays(T_WC, min_depth, max_depth, n_stratified_samples, n_surf_samples, dirs_W, gt_depth):
    #rays in world frame    
    origins = T_WC[:, :3, 3]
    dirs_W = dirs_W.view(-1, 3)
    n_rays = dirs_W.shape[0]

    #stratified samples along rays
    z_vals = stratified_sample(min_depth, max_depth, n_rays, T_WC.device, n_stratified_samples)


    # if gt_depth is given, first sample at surface then around surface
    if gt_depth is not None and n_surf_samples > 0 and False:
        surface_z_vals = gt_depth 
        offsets = torch.normal(torch.zeros(gt_depth.shape[0], n_surf_samples-1), 0.1).to(z_vals.device)
        near_surf_z_vals = gt_depth[:,None] + offsets
        if not isinstance(min_depth, torch.Tensor):
            min_depth = torch.full(near_surf_z_vals.shape, min_depth).to(z_vals.device)[...,0]
        near_surf_z_vals = torch.clamp(near_surf_z_vals, min_depth[:,None], max_depth[:,None])

        z_vals = torch.cat((near_surf_z_vals, z_vals), dim=1)

    # point cloud of 3d sample locations
    pc = origins[:, None, :] + (dirs_W[:, None, :] * z_vals[:, :, None])
    
    # surf_pc filter out points on the ground
    surf_pc = origins + (dirs_W * gt_depth[:, None])
    # isNotGround = surf_pc[:, 2] > 0.01
    isNotGround = surf_pc[:, 2] > -0.11
    # surf_pc = surf_pc[isNotGround]

    return pc, z_vals, surf_pc

def surface_sample(T_WC, dirs_W, depth, device):
    n_rays = 500
    torch.manual_seed(0)
    indices = torch.randint(0, depth.shape[1], (n_rays,), device=device)
    depth_sample = depth[0][indices] #? could cause problem, check dimension
    mask_valid_depth = depth_sample > 0
    depth_sample = depth_sample[mask_valid_depth]
    indices = indices[mask_valid_depth]
    dirs_W_sample = dirs_W[:, indices].view(-1, 3)
    origins = T_WC[:, :3, 3]

    surf_pc = origins + (dirs_W_sample * depth_sample[:, None])
    return surf_pc

# naive ray distance
def bounds_ray(depth_sample, z_vals, dirs_C_sample):
    bounds = depth_sample[:, None] - z_vals
    z_to_euclidean_depth = dirs_C_sample.norm(dim=-1)
    bounds = z_to_euclidean_depth[:, None] * bounds
    return bounds

def bounds_pc(pc, z_vals, surf_pc, depth_sample):
    # surf_pc = pc[:, 0]
    diff = pc[:, :, None] - surf_pc 
    dists = diff.norm(dim=-1)
    dists, closest_ixs = dists.min(axis=-1)
    behind_surf = z_vals > depth_sample[:, None]
    dists[behind_surf] *= -1
    #! set to 0 if behind surface
    # dists[behind_surf] = 0
    bounds = dists

    return bounds

def sample_pixels(n_rays, n_frames, h, w, device):
    """
    Sample pixels from the image
    """
    torch.manual_seed(0)
    total_rays = n_rays * n_frames 
    indices_b = torch.arange(n_frames, device=device).repeat_interleave(n_rays)
    indices_h = torch.randint(0, h, (total_rays,), device=device)
    indices_w = torch.randint(0, w, (total_rays,), device=device)

    return indices_b, indices_h, indices_w


def sample_points(depth_batch, T_WC_batch, n_rays, dirs_C,dist_behind_surf, n_strat_samples, n_surf_samples, min_depth, max_depth, norm_batch, device="cpu"):
    """
    sample points by first sampling pixels and then sampling along the rays
    """
    n_frames = depth_batch.shape[0]
    indices_b, indices_h, indices_w = sample_pixels(n_rays, n_frames, depth_batch.shape[1], depth_batch.shape[2], T_WC_batch.device)


    (dirs_C_sample, depth_sample, T_WC_sample, indices_b, indices_h, indices_w) = get_batch_data(depth_batch, T_WC_batch, dirs_C, indices_b, indices_h, indices_w, norm_batch)

    max_depth = depth_sample + dist_behind_surf
    pc, z_vals= sample_along_rays(T_WC_sample, min_depth, max_depth, n_strat_samples, n_surf_samples, dirs_C_sample, depth_sample)

    sample_pts = {
        "depth_batch": depth_batch,
        "pc": pc, #TODO: may need to remove surf_pc
        "z_vals": z_vals,
        "indices_b": indices_b,
        "indices_h": indices_h,
        "indices_w": indices_w,
        "dirs_C_sample": dirs_C_sample,
        "depth_sample": depth_sample,
        "T_WC_sample": T_WC_sample,
        "surf_pc": pc[:, 0],
    }
    return sample_pts


def sample_lidar_points(depth, T_WC, n_rays, dirs_W, dist_behind_surf, n_strat_samples, n_surf_samples, min_depth, max_depth, device='cpu'):
    torch.manual_seed(0)
    indices = torch.randint(0, depth.shape[1], (n_rays,), device=device)
    depth_sample = depth[0][indices] #? could cause problem, check dimension
    mask_valid_depth = depth_sample > min_depth
    depth_sample = depth_sample[mask_valid_depth]
    indices = indices[mask_valid_depth]
    dirs_W_sample = dirs_W[:, indices].view(-1, 3)

    max_sample_depth = torch.min(depth_sample + dist_behind_surf, torch.tensor(max_depth))
    # max_sample_depth = depth_sample

    if True:
        pc, z_vals, surf_pc = sample_along_lidar_rays(T_WC, min_depth, max_sample_depth, n_strat_samples, n_surf_samples, dirs_W_sample, depth_sample)
    else:
        #TODO: need a new sampling strategy, output surf_pc and pc
        numsamples = 5000 
        dim = 3
        OutsideSize = numsamples + 2
        WholeSize = 0
        origin = T_WC[0, :3, 3]
        origins = T_WC[:, :3, 3]
        surf_pc = origins + (dirs_W_sample * depth_sample[:, None])
        while OutsideSize > 0:
            P  = origin + torch.rand((15*numsamples,dim),dtype=torch.float32, device='cuda')-0.5 # random start point
            dP = torch.rand((15*numsamples,dim),dtype=torch.float32, device='cuda')-0.5 # random direction
            rL = (torch.rand((15*numsamples,1),dtype=torch.float32, device='cuda'))*(max_depth) # random length
            nP = P + torch.nn.functional.normalize(dP,dim=1)*rL

            # need our own PointsInside
            # PointsInside = torch.all((nP <= 0.5),dim=1) & torch.all((nP >= -0.5),dim=1)
            print(depth_sample)
            bounds = bounds_pc(P, surf_pc, depth_sample)

            x0 = P[PointsInside, :]
            x1 = nP[PointsInside, :]
            if (x0.shape[0]<=1):
                continue 

            # obs_distance0 = bounds_pc(x0, surf_pc)
            # where_d = (obs_distance0 > minimum) & (obs_distance0 < maximum)
            OutsideSize = OutsideSize - x0.shape[0]
            WholeSize = WholeSize + x0.shape[0]

            if WholeSize > numsamples:
                break


    sample_pts = {
        "depth_batch": depth, 
        "pc": pc,
        "z_vals": z_vals,
        "surf_pc": surf_pc,
        "indices": indices,
        "dirs_W_sample": dirs_W_sample,
        "depth_sample": depth_sample,
        "T_WC": T_WC,
    }

    return sample_pts 


def generate_points_and_speed(pc, speeds, num):
    pc = pc.view(-1, 3)
    speeds = speeds.view(-1, 1)
    valid_indices = torch.where(speeds < 1)[0]
    start_indices = torch.randint(0, valid_indices.shape[0], (num,))
    end_indices = torch.randint(0, pc.shape[0], (num,))
    x0 = pc[start_indices]
    x1 = pc[end_indices]
    x = torch.cat((x0, x1), dim=1)
    y = torch.cat((speeds[start_indices], speeds[end_indices]), dim=1)




    #TODO: catenate qs and qe and generate points and speed
    # all_point_pairs = []
    # all_distances = []
    # for pc_sample, bound in pc_bound_pairs:
    #     point_pairs = []
    #     distances = []
    #     # Loop through the examples in pc_sample
    #     for i in range(pc_sample.shape[0]):
    #         qs = pc_sample[i][0]  # start point
    #         bound_qs = bound[i][0]
            
    #         for j in range(1, pc_sample.shape[1]):
    #             qe = pc_sample[i][j]  # end point
    #             bound_qe = bound[i][j]
                
    #             # Concatenate qs and qe to form a point pair and add it to the list
    #             point_pairs.append(np.concatenate([qs, qe]))

    #             distances.append([bound_qs, bound_qe])

    #     # Convert the list of point pairs to a numpy array
    #     point_pairs_array = np.array(point_pairs)
    #     distances_array = np.array(distances)
    #     distances_array[:,0] = np.clip(distances_array[:,0] * speed_scale, 0, 1)
    #     distances_array[:,1] = np.clip(distances_array[:,1] * speed_scale, 0, 1)

    #     all_point_pairs.append(point_pairs_array)
    #     all_distances.append(distances_array)
    
    # all_point_pairs = np.concatenate(all_point_pairs, axis=0)
    # all_distances = np.concatenate(all_distances, axis=0)

    out_path = './datasets/isdf-seqs'
    np.save('{}/sampled_points_test'.format(out_path),x.cpu().numpy())
    np.save('{}/speed_test'.format(out_path),y.cpu().numpy())


def meshvis(meshpath, scale, pc, speeds):
    import trimesh
    pc = pc.cpu().numpy()
    speeds = speeds.cpu().numpy()
    # # Vis evaluation points
    mesh_gt = trimesh.load(meshpath)
    # Create a scaling matrix
    scaling_matrix = np.eye(4)
    scaling_matrix[0, 0] = scale
    scaling_matrix[1, 1] = scale
    scaling_matrix[2, 2] = scale
    
    # Apply the scaling to the mesh
    mesh_gt.apply_transform(scaling_matrix)

    scene = trimesh.Scene(mesh_gt)
    distances = speeds.reshape(-1, 1)
    colors = (np.outer(1.0 - distances, [255, 0, 0, 80]) + 
          np.outer(distances, [255, 255, 255, 80])).astype(np.uint8)
    # alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)    
    # rgba_colors = np.concatenate([colors, alpha], axis=1)

    point_cloud = trimesh.PointCloud(pc.reshape(-1, 3), colors.reshape(-1, 4))
    scene.add_geometry([point_cloud])
    scene.show() 


def uniform_gt_pts_speeds(center, offsetxyz, meshpath="datasets/isdf-seqs/mesh.obj", minimum=0.07, maximum=0.3, num=10000):
    random_offsets = 2*torch.rand(num*10, 3)-1
    random_offsets[:,0] = random_offsets[:,0]*offsetxyz[0]+center[0]
    random_offsets[:,1] = random_offsets[:,1]*offsetxyz[1]+center[1]
    random_offsets[:,2] = random_offsets[:,2]*offsetxyz[2]+center[2]
    pc0 = random_offsets

    dP = torch.rand((num*10, 3))-0.5
    rL = (torch.rand(num*10, 1))*2
    pc1 = pc0 + torch.nn.functional.normalize(dP, dim=1)*rL


    v, f = igl.read_triangle_mesh(meshpath)
    t_obs = v[f].reshape(-1, 3)

    device = pc0.device
    x0 = pc0.cpu().numpy()
    y0 = unsigned_distance_without_bvh(t_obs, x0)
    y0 = torch.from_numpy(y0).float().to(device).unsqueeze(1)
    # bounds to speeds
    y0 = torch.clip((y0 - minimum) / (maximum - minimum), 0, 1)

    valid_indices = torch.where((y0 < 1) & (y0 > 0))[0]
    if num <= len(valid_indices):
        # Select without replacement if num is less than or equal to the size of valid_indices
        start_indices = valid_indices[torch.randperm(len(valid_indices))[:num]]
    else:
        # Select with replacement if num is greater than the size of valid_indices
        rand_indices = torch.randint(0, len(valid_indices), (num,))
        start_indices = valid_indices[rand_indices]

    x1 = pc1[start_indices].cpu().numpy()
    y1 = unsigned_distance_without_bvh(t_obs, x1)
    y1 = torch.from_numpy(y1).float().to(device).unsqueeze(1)
    y1 = torch.clip((y1 - minimum) / (maximum - minimum), 0, 1)

    x0 = pc0[start_indices]
    x1 = pc1[start_indices]
    x = torch.cat((x0, x1), dim=1)
    y = torch.cat((y0[start_indices], y1), dim=1)

    # import matplotlib.pyplot as plt
    # plt.scatter(all_bounds, all_speeds)
    # plt.xlabel("ground truth distance")
    # plt.ylabel("speed")
    # plt.show()
    # temp = speeds[start_indices]
    # plt.hist(temp.cpu().numpy(), bins=100)
    # plt.title("speed distribution")
    # plt.show()
    return x, y

def origin_data_gen(meshpath, num, offset, margin):
    v, f = igl.read_triangle_mesh(meshpath)
    t_obs = v[f].reshape(-1, 3)

    X_list = []
    Y_list = []
    X_list, Y_list = point_append_list(X_list, Y_list, t_obs, num, offset, margin)

    X = torch.cat(X_list, dim=0)[:num]
    Y = torch.cat(Y_list, dim=0)[:num]

    points = X.detach().cpu().numpy()
    bounds = Y.detach().cpu().numpy()

    bound0 = bounds[:,0]
    bound1 = bounds[:,1]
    speeds = np.zeros((bounds.shape[0],2))
    speeds[:,0] = np.clip(bound0, a_min=offset, a_max=margin)/margin 
    speeds[:,1] = np.clip(bound1, a_min=offset, a_max=margin)/margin

    return points, speeds



def point_append_list(X_list, Y_list, t_obs, num, offset, margin):
    OutsideSize = num + 2 
    WholeSize = 0 
    device = "cuda:0"
    center = [0, 0, 0]
    offsetxyz = [1, 0.5, 1]
    while OutsideSize > 0:
        P = (torch.rand((8*num, 3))-0.5)
        P[:,0] = P[:,0]*offsetxyz[0]+center[0]
        P[:,1] = P[:,1]*offsetxyz[1]+center[1]
        P[:,2] = P[:,2]*offsetxyz[2]+center[2]

        dP = torch.rand((8*num, 3))-0.5
        rL = (torch.rand(8*num, 1))*offsetxyz[0]

        nP = P + torch.nn.functional.normalize(dP, dim=1)*rL

        # PointsInside = torch.all((nP<=0.5) & (nP>=-0.5), dim=1)

        PointsInside = (nP[:,0] <= center[0]+offsetxyz[0]/2) & (nP[:,0] >= center[0]-offsetxyz[0]/2) & (nP[:,1] <= center[1]+offsetxyz[1]/2) & (nP[:,1] >= center[1]-offsetxyz[1]/2) & (nP[:,2] <= center[2]+offsetxyz[2]/2) & (nP[:,2] >= center[2]-offsetxyz[2]/2) 

        x0 = P[PointsInside, :]
        x1 = nP[PointsInside, :]

        obs_dist0 = unsigned_distance_without_bvh(t_obs, x0.cpu().numpy())
        where_d = (obs_dist0 > offset) & (obs_dist0 < margin)
        # where_d = (obs_dist0 > -10)
        x0 = x0[where_d]
        x1 = x1[where_d]
        y0 = obs_dist0[where_d]
        y0 = torch.from_numpy(y0).float().to(device)

        y1 = unsigned_distance_without_bvh(t_obs, x1.cpu().numpy())
        y1 = torch.from_numpy(y1).float().to(device)
        
        x = torch.cat((x0, x1), dim=1)
        y = torch.cat((y0.unsqueeze(1), y1.unsqueeze(1)), dim=1)

        X_list.append(x)
        Y_list.append(y)
        
        OutsideSize = OutsideSize - x.shape[0]
        WholeSize = WholeSize + x.shape[0]

        if(WholeSize > num):
            break 
    return X_list, Y_list


if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # folder = "datasets/isdf-seqs/"
    # meshpath = "./datasets/isdf-seqs/mesh.obj"
    # config_file = os.path.join(folder,"replicaCAD_info.json")
    # with open(config_file) as f:
    #     configs = json.load(f)
    # fx = configs["dataset"]["camera"]["fx"]
    # fy = configs["dataset"]["camera"]["fy"]
    # cx = configs["dataset"]["camera"]["cx"]
    # cy = configs["dataset"]["camera"]["cy"]
    # W = configs["dataset"]["camera"]["w"]
    # H = configs["dataset"]["camera"]["h"]
    # depth_scale = configs["dataset"]["depth_scale"]
    # inv_depth_scale = 1. / depth_scale
    # min_depth = configs["sample"]["depth_range"][0]
    # max_depth = configs["sample"]["depth_range"][1]
    # n_rays = configs["sample"]["n_rays"]
    # n_rays = 5000 #5000
    # n_strat_samples = configs["sample"]["n_strat_samples"]
    # n_surf_samples = configs["sample"]["n_surf_samples"]
    # dist_behind_surf = configs["sample"]["dist_behind_surf"]  

    # """ params for replica dataset
    #         root_dir,
    #         traj_file=None,
    #         rgb_transform=None,
    #         depth_transform=None,
    #         noisy_depth=False,
    #         col_ext=".jpg",
    # """
    # root_dir = os.path.join(folder,"apt_2_nav/results")
    # traj_file = os.path.join(folder, "apt_2_nav/traj.txt")
    # depth_transform = transforms.Compose([DepthScale(inv_depth_scale), DepthFilter(max_depth)])

    # col_ext = ".png"
    # up_vec = np.array([0., 1., 0.])
    # noisy_depth = 1

    # dataset = ReplicaDataset(root_dir=root_dir, traj_file=traj_file, rgb_transform=None, depth_transform=depth_transform, noisy_depth=noisy_depth, col_ext=col_ext)

    # dirs_C = transform.ray_dirs_C(1,H,W,fx,fy,cx,cy,device,depth_type="z")
    dataset = ReplicaDataset(root_dir="datasets/isdf-seqs/apt_2_nav/results", traj_file="datasets/isdf-seqs/apt_2_nav/traj.txt", config="datasets/isdf-seqs/replicaCAD_info.json")

    # # print("dataset length: ", len(dataset))

    FRAMES = [1600, 1400, 2000, 1700, 1900, 1300, 250, 1080]
    # FRAMES = [2020]

    points, speeds = dataset.get_speeds(FRAMES, maximum=0.3, num=100000) 
    meshvis("datasets/isdf-seqs/mesh.obj", 1, points, speeds)
    # pc, bound = get_bounds(FRAMES) 

    # scaler = 1
    # scaled_pc = pc * scaler
    # # scaled_bound = bound * scaler
    # min_depth = 0.1
    # speeds = get_speeds(bound, minimum=min_depth, maximum=0.3)

    
    # meshvis(meshpath, scaler, scaled_pc, speeds)
    # plot_distribution(scaled_pc, speeds)
    
    camera = dataset[FRAMES[0]]["T"][ :3, -1]
    print("camera: ", camera)
    # print("pc shape: ", scaled_pc.shape)
    # print("speed shape: ", speeds.shape)

    # generate_points_and_speed(pc, speeds, num=100000)     # ! uncomment this to generate points and speed






