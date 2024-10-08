#! this file read from depth npy files and pose npy files, and create a point cloud in trimesh

import numpy as np
import os
import re 
import cv2
import torch

RotY90 = np.array([[0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0]], dtype=np.float32)
RotX90_ = np.array([[1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]], dtype=np.float32)

def natural_sort_key(s):
    """
    Compute a key for natural sort order.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def pointcloud_from_depth(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_type: str = "z",
    skip=1,
) -> np.ndarray:
    assert depth_type in ["z", "euclidean"], "Unexpected depth_type"

    rows, cols = depth.shape
    c, r = np.meshgrid(
        np.arange(cols, step=skip), np.arange(rows, step=skip), sparse=True)
    depth = depth[::skip, ::skip]
    valid = ~np.isnan(depth)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - cx) / fx, np.nan)
    y = np.where(valid, z * (r - cy) / fy, np.nan)
    pc = np.dstack((x, y, z))

    if depth_type == "euclidean":
        norm = np.linalg.norm(pc, axis=2)
        pc = pc * (z / norm)[:, :, None]
    return pc

def pointcloud_from_depth_torch(
    depth,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_type: str = "z",
    skip=1,
) -> np.ndarray:
    assert depth_type in ["z", "euclidean"], "Unexpected depth_type"

    rows, cols = depth.shape
    c, r = np.meshgrid(
        np.arange(cols, step=skip), np.arange(rows, step=skip), sparse=True)
    c = torch.from_numpy(c).to(depth.device)
    r = torch.from_numpy(r).to(depth.device)
    depth = depth[::skip, ::skip]
    valid = ~torch.isnan(depth)
    nan_tensor = torch.FloatTensor([float('nan')]).to(depth.device)
    z = torch.where(valid, depth, nan_tensor)
    x = torch.where(valid, z * (c - cx) / fx, nan_tensor)
    y = torch.where(valid, z * (r - cy) / fy, nan_tensor)
    pc = torch.dstack((x, y, z))

    if depth_type == "euclidean":
        norm = torch.linalg.norm(pc, axis=2)
        pc = pc * (z / norm)[:, :, None]
    return pc

def cvt_transform_to_opengl_format(transform) -> np.ndarray:
    """
    Convert camera-space-2-world-space transform into OpenGL format (for iSDF)
    """
    if transform is None:
        transform = np.eye(4)
    T = np.array([[1, 0, 0, 0],
                [0, np.cos(np.pi), -np.sin(np.pi), 0],
                [0, np.sin(np.pi), np.cos(np.pi), 0],
                [0, 0, 0, 1]])
    return transform @ T

def pointcloud_world(depth, pose, fx, fy, cx, cy, scale=1000, opengl=False):
    pc = pointcloud_from_depth_torch(depth, fx, fy, cx, cy)
    pc = pc / scale

    # pose = convert_transform_odom_to_cam(pose)
    pose = convert_transform_odom_to_cam_torch(pose)
    if opengl: # opengl format
        pose = cvt_transform_to_opengl_format(pose)

    origin = pose[:3, 3]
    rotation = pose[:3, :3]
    # pc_W = np.einsum('ij,kj->ki', rotation, pc.reshape(-1, 3)) + origin
    pc_W = torch.matmul(rotation, pc.reshape(-1, 3).T).T + origin
    return pc_W

def get_filenames_for_tbot(folder):
    files = os.listdir(folder)
    depth_files = filter(lambda x: x.endswith("depth.npy"), files)
    sorted_depth_filenames = sorted(depth_files, key=natural_sort_key)
    sorted_depth_filenames = np.array([os.path.join(folder, file) for file in sorted_depth_filenames])
    
    pose_files = filter(lambda x: x.endswith("pose.npy"), files)
    sorted_pose_filenames = sorted(pose_files, key=natural_sort_key)
    sorted_pose_filenames = np.array([os.path.join(folder, file) for file in sorted_pose_filenames])

    rgb_files = filter(lambda x: x.endswith("rgb.jpg"), files)
    sorted_rgb_filenames = sorted(rgb_files, key=natural_sort_key)
    sorted_rgb_filenames = np.array([os.path.join(folder, file) for file in sorted_rgb_filenames])
    return sorted_depth_filenames, sorted_pose_filenames, sorted_rgb_filenames

def main():
    import trimesh
    # obtain the depth and pose files in sorted order
    folder = "kitchen3/data"
    sorted_depth_filenames, sorted_pose_filenames, sorted_rgb_filenames = get_filenames_for_tbot(folder=folder)
    # print("sorted_depth_filenames: ", sorted_depth_filenames)
    # print("sorted_pose_filenames: ", sorted_pose_filenames)

    #! transform the depth npy files to point cloud in world frame
    # pc_W = point_cloud_world()
    scene = trimesh.Scene()
    # for i in range(len(sorted_pose_filenames)):
    for i in range(0, len(sorted_pose_filenames), 100):
        # load the depth npy file
        depth = np.load(sorted_depth_filenames[i]).astype(np.int32)
        depth_torch = torch.from_numpy(depth).cuda()
        # print("depth: ", depth.shape) # (680, 1200)
        # print("depth: ", depth)
        # load the pose npy file
        pose = np.load(sorted_pose_filenames[i])
        pose_torch = torch.from_numpy(pose).cuda()
        # print("pose: ", pose.shape) # (4, 4)
        # print("pose: ", pose)
        # load the rgb image
        rgb = cv2.imread(sorted_rgb_filenames[i])

        # - 909.9520874023438 fx
        # - 0.0
        # - 641.1693115234375 cx
        # - 0.0
        # - 909.8970336914062 fy
        # - 352.7703552246094 cy
        fx = 909.9520874023438
        fy = 909.8970336914062
        cx = 641.1693115234375
        cy = 352.7703552246094
        pc_W = pointcloud_world(depth_torch, pose_torch, fx, fy, cx, cy, scale=1000, opengl=False)
        pc_W = pc_W.cpu().numpy()
        #! filter the point cloud with z value smaller than 0.1
        filter_mask = (pc_W[:, 2] < 10.25) & (pc_W[:, 2] > -10.1)
        pc_W = pc_W[filter_mask]
        point_cloud = trimesh.PointCloud(pc_W.reshape(-1, 3), colors=rgb.reshape(-1, 3)[filter_mask])
        
        scene.add_geometry(point_cloud)
    scene.show()




def convert_transform_odom_to_cam(_tform: np.ndarray) -> np.ndarray:
    print("this is original xyz", _tform[:3, 3])
    old_tform = _tform.copy()
    _tform = _tform.copy()

    x_offset = -0.045  # -4.5cm
    y_offset = 0.035  # # 3.5cm
    z_offset = 0.23  # 23cm
    corr_vec = np.array([x_offset, y_offset, z_offset],
                        dtype=np.float32)

    rotmat = _tform[:3, :3]
    corr_vec_aligned = np.matmul(rotmat, corr_vec)
    _tform[:3, 3] += corr_vec_aligned

    # conversion
    coord_adjust = np.eye(4, dtype=np.float32)

    coord_adjust[:3, :3] = RotX90_ @ RotY90
    ans = _tform @ coord_adjust
    print("_tform", _tform)
    print("coord", coord_adjust)
    print("this is new xyz", ans[:3, 3])
    print("this is offset xyz ****", ans[:3, 3]-(old_tform@coord_adjust)[:3,3])
    return ans


def convert_cam_xyz_to_odom_xyz(pose: np.ndarray) -> np.ndarray:
    # Inverse translation offsets
    x_offset = -0.045  # -4.5cm
    y_offset = 0.035  # 3.5cm
    z_offset = 0.23   # 23cm
    translation = np.array([x_offset, y_offset, z_offset], dtype=np.float32)

    # Apply inverse translation
    ans = pose[:3, 3] - translation
    return ans


import torch

def convert_transform_odom_to_cam_torch(_tform: torch.Tensor) -> torch.Tensor:
    _tform = _tform.clone()

    x_offset = -0.045  # -4.5cm
    y_offset = 0.035   # 3.5cm
    z_offset = 0.23    # 23cm
    corr_vec = torch.tensor([x_offset, y_offset, z_offset], dtype=torch.float32, device=_tform.device)

    rotmat = _tform[:3, :3]
    corr_vec_aligned = torch.matmul(rotmat, corr_vec)
    _tform[:3, 3] += corr_vec_aligned

    # conversion
    RotY90 = np.array([[0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0]], dtype=np.float32)
    RotX90_ = np.array([[1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]], dtype=np.float32)
    RotY90 = torch.from_numpy(RotY90).cuda()
    RotX90_ = torch.from_numpy(RotX90_).cuda()
    coord_adjust = torch.eye(4, dtype=torch.float32, device=_tform.device)
    coord_adjust[:3, :3] = RotX90_ @ RotY90
    return torch.matmul(_tform, coord_adjust)


# def convert_rgbdt(depth_img, rgb_img, tform_mat):
#     # convert depth
#     depth_img = depth_img.astype(np.float32) * 0.001  # depth scaling
#     # get robot x,y position
#     x, y = float(tform_mat[0, 3]), float(tform_mat[1, 3])
#     # get camera transform
#     tform_cam = convert_transform_odom_to_cam(_tform=tform_mat)
#     return np.array([x, y], dtype=np.float32), rgb_img, depth_img, tform_cam



if __name__ == "__main__":
    main()