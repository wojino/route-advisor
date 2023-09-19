import sys, os; sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.config import *

import cv2
import numpy as np

def rotation_from_euler(roll=1., pitch=1., yaw=1.): # in radians
    si, sj, sk = np.sin(roll), np.sin(pitch), np.sin(yaw)
    ci, cj, ck = np.cos(roll), np.cos(pitch), np.cos(yaw)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    R = np.identity(4)
    R[0, 0] = cj * ck
    R[0, 1] = sj * sc - cs
    R[0, 2] = sj * cc + ss
    R[1, 0] = cj * sk
    R[1, 1] = sj * ss + cc
    R[1, 2] = sj * cs - sc
    R[2, 0] = -sj
    R[2, 1] = cj * si
    R[2, 2] = cj * ci

    return R # [4, 4]

def translation_matrix(x=0., y=0., z=0.): # in meters
    T = np.identity(4)
    T[:3, 3] = [x, y, z]

    return T # [4, 4]

def load_camera_params():
    # Extrinsic
    pitch, roll, yaw = np.deg2rad(CAM_PITCH), np.deg2rad(CAM_ROLL), np.deg2rad(CAM_YAW)
    x, y, z = CAM_X, CAM_Y, CAM_Z
    
    R_veh2cam = np.transpose(rotation_from_euler(roll, pitch, yaw))
    T_veh2cam = translation_matrix(-x, -y, -z)

    # Rotate to camera coordinates
    R = np.array([[0., -1., 0., 0.],
                  [0., 0., -1., 0.],
                  [1., 0., 0., 0.],
                  [0., 0., 0., 1.]])

    extrinsic = R @ R_veh2cam @ T_veh2cam

    # Intrinsic
    fx, fy = float(IMG_WIDTH), float(IMG_HEIGHT)
    u0, v0 = fx / 2, fy / 2

    intrinsic = np.array([[fx, 0, u0, 0],
                  [0, fy, v0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    return extrinsic, intrinsic

def generate_direct_backward_mapping(extrinsic, intrinsic):
    world_x_coords = np.arange(WORLD_X_MIN, WORLD_X_MAX, WORLD_X_INTERVAL)
    world_y_coords = np.arange(WORLD_Y_MIN, WORLD_Y_MAX, WORLD_Y_INTERVAL)
    
    output_height = len(world_x_coords)
    output_width = len(world_y_coords)
    
    map_x = np.zeros((output_height, output_width)).astype(np.float32)
    map_y = np.zeros((output_height, output_width)).astype(np.float32)
    
    for i, world_x in enumerate(world_x_coords):
        for j, world_y in enumerate(world_y_coords):
            world_coord = [world_x, world_y, 0, 1]
            camera_coord = extrinsic[:3, :] @ world_coord
            uv_coord = intrinsic[:3, :3] @ camera_coord
            uv_coord /= uv_coord[2]

            map_x[i][j] = uv_coord[0]
            map_y[i][j] = uv_coord[1]
            
    return map_x, map_y

def bilinear_sampler(imgs, pix_coords):
    img_h, img_w, img_c = imgs.shape
    pix_h, pix_w, pix_c = pix_coords.shape
    out_shape = (pix_h, pix_w, img_c)

    pix_x, pix_y = np.split(pix_coords, [1], axis=-1)  # [pix_h, pix_w, 1]
    pix_x = pix_x.astype(np.float32)
    pix_y = pix_y.astype(np.float32)

    # Rounding
    pix_x0 = np.floor(pix_x)
    pix_x1 = pix_x0 + 1
    pix_y0 = np.floor(pix_y)
    pix_y1 = pix_y0 + 1

    # Clip within image boundary
    y_max = (img_h - 1)
    x_max = (img_w - 1)
    zero = np.zeros([1])

    pix_x0 = np.clip(pix_x0, zero, x_max)
    pix_y0 = np.clip(pix_y0, zero, y_max)
    pix_x1 = np.clip(pix_x1, zero, x_max)
    pix_y1 = np.clip(pix_y1, zero, y_max)

    # Weights [pix_h, pix_w, 1]
    wt_x0 = pix_x1 - pix_x
    wt_x1 = pix_x - pix_x0
    wt_y0 = pix_y1 - pix_y
    wt_y1 = pix_y - pix_y0

    # indices in the image to sample from
    dim = img_w

    # Apply the lower and upper bound pix coord
    base_y0 = pix_y0 * dim
    base_y1 = pix_y1 * dim

    # 4 corner vertices
    idx00 = (pix_x0 + base_y0).flatten().astype(np.int32)
    idx01 = (pix_x0 + base_y1).astype(np.int32)
    idx10 = (pix_x1 + base_y0).astype(np.int32)
    idx11 = (pix_x1 + base_y1).astype(np.int32)

    # Gather pixels from image using vertices
    imgs_flat = imgs.reshape([-1, img_c]).astype(np.float32)
    im00 = imgs_flat[idx00].reshape(out_shape)
    im01 = imgs_flat[idx01].reshape(out_shape)
    im10 = imgs_flat[idx10].reshape(out_shape)
    im11 = imgs_flat[idx11].reshape(out_shape)

    # Apply weights [pix_h, pix_w, 1]
    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1
    output = w00 * im00 + w01 * im01 + w10 * im10 + w11 * im11
    return output

def remap_bilinear(image, map_x, map_y):
    pix_coords = np.concatenate([np.expand_dims(map_x, -1), np.expand_dims(map_y, -1)], axis=-1)
    bilinear_output = bilinear_sampler(image, pix_coords)
    output = np.round(bilinear_output).astype(np.int32)
    return output

def front_to_bev(front_camera):
    extrinsic, intrinsic = load_camera_params()
    map_x, map_y = generate_direct_backward_mapping(extrinsic, intrinsic)
    bev = remap_bilinear(front_camera, map_x, map_y)
    bev = bev.astype(np.float32)
    bev = cv2.rotate(bev, cv2.ROTATE_180)

    return bev