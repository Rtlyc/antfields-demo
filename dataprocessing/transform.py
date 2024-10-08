# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import typing
from scipy import interpolate
import torch
import math


def ray_dirs_C(B, H, W, fx, fy, cx, cy, device, depth_type='z'):
    c, r = torch.meshgrid(torch.arange(W, device=device),
                          torch.arange(H, device=device))
    c, r = c.t().float(), r.t().float()
    size = [B, H, W]

    C = torch.empty(size, device=device)
    R = torch.empty(size, device=device)
    C[:, :, :] = c[None, :, :]
    R[:, :, :] = r[None, :, :]

    z = torch.ones(size, device=device)
    x = (C - cx) / fx
    y = (R - cy) / fy

    dirs = torch.stack((x, y, z), dim=3)
    if depth_type == 'euclidean':
        norm = torch.norm(dirs, dim=3)
        dirs = dirs * (1. / norm)[:, :, :, None]

    return dirs

def lidar_ray_dirs_C(B, H=468, V=16, horizontal_fov=45, vertical_fov=15, device='cuda:0'):
    lidar_vertical_low = -15 / 180.0 * np.pi
    lidar_vertical_high = 15 / 180.0 * np.pi
    lidar_vertical_n_beams = 16
    lidar_vertical_beams = np.arange(
        lidar_vertical_low,
        lidar_vertical_high + (lidar_vertical_high - lidar_vertical_low) / (lidar_vertical_n_beams - 1),
        (lidar_vertical_high - lidar_vertical_low) / (lidar_vertical_n_beams - 1),
    )

    lidar_horizontal_low = -45 / 180.0 * np.pi
    lidar_horizontal_high = 45 / 180.0 * np.pi
    lidar_horizontal_n_beams = 468
    lidar_horizontal_beams = np.arange(
        lidar_horizontal_low,
        lidar_horizontal_high,
        (lidar_horizontal_high - lidar_horizontal_low) / (lidar_horizontal_n_beams),
    )

    # Creating a grid of vertical and horizontal beam angles
    vv, hh = np.meshgrid(lidar_vertical_beams, lidar_horizontal_beams)

    # Calculating Cartesian coordinates from spherical coordinates
    x = np.cos(vv) * np.cos(hh)
    y = np.cos(vv) * np.sin(hh)
    z = np.sin(vv)

    # Stacking the direction vectors
    dirs = np.stack((x, y, z))

    #! visualize the directions
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    ray_dirs = dirs

    # Flatten the ray directions for plotting
    ray_dirs_flat = ray_dirs.reshape(-1, 3)

    # Creating the 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the rays
    ax.quiver(0, 0, 0, 
            ray_dirs_flat[:, 0], ray_dirs_flat[:, 1], ray_dirs_flat[:, 2], 
            length=0.1, normalize=True)

    # Setting labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('LiDAR Ray Directions')

    plt.show()

    return dirs


def sample_lidar_rays(vertical_fov=30, horizontal_fov=90, vertical_n_beams=16, horizontal_n_beams=468):
    # Calculate the angular spacing between beams
    vertical_spacing = vertical_fov / (vertical_n_beams - 1)
    horizontal_spacing = horizontal_fov / (horizontal_n_beams - 1)

    # Initialize an empty tensor to store the sampled rays
    sampled_rays = torch.zeros((vertical_n_beams * horizontal_n_beams, 3))

    # Loop through the vertical and horizontal angles
    ray_idx = 0
    for v_idx in range(vertical_n_beams):
        for h_idx in range(horizontal_n_beams):
            # Calculate the vertical and horizontal angles
            vertical_angle = -vertical_fov / 2.0 + v_idx * vertical_spacing
            horizontal_angle = -horizontal_fov / 2.0 + h_idx * horizontal_spacing

            # Convert angles to radians
            vertical_angle_rad = math.radians(vertical_angle)
            horizontal_angle_rad = math.radians(horizontal_angle)

            # Calculate the direction vector for the ray
            vertical_direction = torch.tensor([math.sin(vertical_angle_rad), math.cos(vertical_angle_rad), 0.0])
            horizontal_direction = torch.tensor([math.sin(horizontal_angle_rad), 0.0, math.cos(horizontal_angle_rad)])

            # Combine the vertical and horizontal directions
            ray_direction = torch.nn.functional.normalize(vertical_direction + horizontal_direction, dim=0)

            # Store the ray direction in the tensor
            sampled_rays[ray_idx] = ray_direction
            ray_idx += 1

    return sampled_rays


def sample_lidar_rays_igib():
    width = 1200
    height = 680
    vertical_fov = 90
    horizontal_fov = (
        2 * np.arctan(np.tan(vertical_fov / 180.0 * np.pi / 2.0) * width / height) / np.pi * 180.0
    )

def origin_dirs_W(T_WC, dirs_C):
    R_WC = T_WC[:, :3, :3]
    dirs_W = (R_WC * dirs_C[..., None, :]).sum(dim=-1)
    origins = T_WC[:, :3, -1]

    return origins, dirs_W


def normalize(x):
    assert x.ndim == 1, "x must be a vector (ndim: 1)"
    return x / np.linalg.norm(x)


def look_at(
    eye,
    target: typing.Optional[typing.Any] = None,
    up: typing.Optional[typing.Any] = None,
) -> np.ndarray:
    """Returns transformation matrix with eye, at and up.

    Parameters
    ----------
    eye: (3,) float
        Camera position.
    target: (3,) float
        Camera look_at position.
    up: (3,) float
        Vector that defines y-axis of camera (z-axis is vector from eye to at).

    Returns
    -------
    T_cam2world: (4, 4) float (if return_homography is True)
        Homography transformation matrix from camera to world.
        Points are transformed like below:
            # x: camera coordinate, y: world coordinate
            y = trimesh.transforms.transform_points(x, T_cam2world)
            x = trimesh.transforms.transform_points(
                y, np.linalg.inv(T_cam2world)
            )
    """
    eye = np.asarray(eye, dtype=float)

    if target is None:
        target = np.array([0, 0, 0], dtype=float)
    else:
        target = np.asarray(target, dtype=float)

    if up is None:
        up = np.array([0, 0, -1], dtype=float)
    else:
        up = np.asarray(up, dtype=float)

    assert eye.shape == (3,), "eye must be (3,) float"
    assert target.shape == (3,), "target must be (3,) float"
    assert up.shape == (3,), "up must be (3,) float"

    # create new axes
    z_axis: np.ndarray = normalize(target - eye)
    x_axis: np.ndarray = normalize(np.cross(up, z_axis))
    y_axis: np.ndarray = normalize(np.cross(z_axis, x_axis))

    # create rotation matrix: [bs, 3, 3]
    R: np.ndarray = np.vstack((x_axis, y_axis, z_axis))
    t: np.ndarray = eye

    return R.T, t


def to_trimesh(transform=None):
    import trimesh
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(-180), [1, 0, 0]
    )


def to_replica(transform=None):
    import trimesh
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(180), [0, 0, 1]
    )


def interpolation(keypoints, n_points):
    tick, _ = interpolate.splprep(keypoints.T, s=0)
    points = interpolate.splev(np.linspace(0, 1, n_points), tick)
    points = np.array(points, dtype=np.float64).T
    return points


def backproject_pointclouds(depths, fx, fy, cx, cy):
    pcs = []
    batch_size = depths.shape[0]
    for batch_i in range(batch_size):
        pcd = pointcloud_from_depth(
            depths[batch_i], fx, fy, cx, cy
        )
        pc_flat = pcd.reshape(-1, 3)
        pcs.append(pc_flat)

    pcs = np.stack(pcs, axis=0)
    return pcs


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
    assert depth.dtype.kind == "f", "depth must be float and have meter values"

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


def pc_bounds(pc):
    min_x = np.min(pc[:, 0])
    max_x = np.max(pc[:, 0])
    min_y = np.min(pc[:, 1])
    max_y = np.max(pc[:, 1])
    min_z = np.min(pc[:, 2])
    max_z = np.max(pc[:, 2])
    extents = np.array([max_x - min_x, max_y - min_y, max_z - min_z])
    centroid = np.array(
        [(max_x + min_x) / 2.0, (max_y + min_y) / 2.0, (max_z + min_z) / 2]
    )

    return extents, centroid


# adapted from https://github.com/wkentaro/morefusion/blob/main/morefusion/geometry/estimate_pointcloud_normals.py
def estimate_pointcloud_normals(points):
    # These lookups denote yx offsets from the anchor point for 8 surrounding
    # directions from the anchor A depicted below.
    #  -----------
    # | 7 | 6 | 5 |
    #  -----------
    # | 0 | A | 4 |
    #  -----------
    # | 1 | 2 | 3 |
    #  -----------
    assert points.shape[2] == 3

    d = 2
    H, W = points.shape[:2]
    points = torch.nn.functional.pad(
        points,
        pad=(0, 0, d, d, d, d),
        mode="constant",
        value=float('nan'),
    )

    lookups = torch.tensor(
        [(-d, 0), (-d, d), (0, d), (d, d), (d, 0), (d, -d), (0, -d), (-d, -d)]
    ).to(points.device)

    j, i = torch.meshgrid(torch.arange(W), torch.arange(H))
    i = i.transpose(0, 1).to(points.device)
    j = j.transpose(0, 1).to(points.device)
    k = torch.arange(8).to(points.device)

    i1 = i + d
    j1 = j + d
    points1 = points[i1, j1]

    lookup = lookups[k]
    i2 = i1[None, :, :] + lookup[:, 0, None, None]
    j2 = j1[None, :, :] + lookup[:, 1, None, None]
    points2 = points[i2, j2]

    lookup = lookups[(k + 2) % 8]
    i3 = i1[None, :, :] + lookup[:, 0, None, None]
    j3 = j1[None, :, :] + lookup[:, 1, None, None]
    points3 = points[i3, j3]

    diff = torch.linalg.norm(points2 - points1, dim=3) + torch.linalg.norm(
        points3 - points1, dim=3
    )
    diff[torch.isnan(diff)] = float('inf')
    indices = torch.argmin(diff, dim=0)

    normals = torch.cross(
        points2[indices, i, j] - points1[i, j],
        points3[indices, i, j] - points1[i, j],
    )
    normals /= torch.linalg.norm(normals, dim=2, keepdims=True)
    return normals


def make_3D_grid(grid_range, dim, device, transform=None, scale=None):
    t = torch.linspace(grid_range[0], grid_range[1], steps=dim, device=device)
    grid = torch.meshgrid(t, t, t)
    grid_3d = torch.cat(
        (grid[0][..., None],
         grid[1][..., None],
         grid[2][..., None]), dim=3
    )

    grid_3d = transform_3D_grid(grid_3d, transform=transform, scale=scale)

    return grid_3d


def transform_3D_grid(grid_3d, transform=None, scale=None):
    if scale is not None:
        grid_3d = grid_3d * scale
    if transform is not None:
        R1 = transform[None, None, None, 0, :3]
        R2 = transform[None, None, None, 1, :3]
        R3 = transform[None, None, None, 2, :3]

        grid1 = (R1 * grid_3d).sum(-1, keepdim=True)
        grid2 = (R2 * grid_3d).sum(-1, keepdim=True)
        grid3 = (R3 * grid_3d).sum(-1, keepdim=True)

        grid_3d = torch.cat([grid1, grid2, grid3], dim=-1)

        trans = transform[None, None, None, :3, 3]
        grid_3d = grid_3d + trans

    return grid_3d


class RotExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        ctx.save_for_backward(w)

        theta = torch.norm(w, dim=1)
        e_w_x = torch.eye(
            3, device=w.device).unsqueeze(0).repeat(w.shape[0], 1, 1)

        if torch.any(theta != 0):
            mask = theta != 0
            n_valid_thetas = mask.sum()
            w_x = torch.zeros((n_valid_thetas, 3, 3), device=w.device)
            valid_w = w[mask]
            w_x[:, 0, 1] = -valid_w[:, 2]
            w_x[:, 1, 0] = valid_w[:, 2]
            w_x[:, 0, 2] = valid_w[:, 1]
            w_x[:, 2, 0] = -valid_w[:, 1]
            w_x[:, 1, 2] = -valid_w[:, 0]
            w_x[:, 2, 1] = valid_w[:, 0]

            valid_theta = theta[mask]
            e_w_x[mask] = (
                e_w_x[mask]
                + (torch.sin(valid_theta) / valid_theta)[:, None, None] * w_x
                + ((1 - torch.cos(valid_theta)) / (valid_theta * valid_theta))[
                    :, None, None
                ]
                * w_x @ w_x
            )

        return e_w_x

    @staticmethod
    def backward(ctx, grad_output):
        (w,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        G1 = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            device=grad_output.device,
        ).view([1, -1])

        G2 = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            device=grad_output.device,
        ).view([1, -1])

        G3 = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            device=grad_output.device,
        ).view([1, -1])

        grad_input_flat = grad_input.view([grad_input.shape[0], -1])

        p1 = (grad_input_flat * G1).sum(1, keepdim=True)
        p2 = (grad_input_flat * G2).sum(1, keepdim=True)
        p3 = (grad_input_flat * G3).sum(1, keepdim=True)

        grad_input = torch.cat((p1, p2, p3), 1)

        return grad_input
