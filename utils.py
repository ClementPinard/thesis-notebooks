from __future__ import division
import torch

pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1,h,w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def invert_mat(matrices):
    rot_matrices = matrices[:,:,:-1].transpose(1,2)
    translation_vectors = - rot_matrices @ matrices[:,:,-1:]
    return(torch.cat([rot_matrices, translation_vectors], dim=-1))


def pose_vec2mat(vec):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, S, 3, 1]
    rot = vec[:, 3:]
    rot_mat = euler2mat(rot)  # [B, S, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=-1)  # [B, S, 3, 4]
    return transform_mat


def pixel2cam(depth):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    pixel_coords.type_as(depth)
    cam_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w)*depth.unsqueeze(1)
    return cam_coords.contiguous()


def cam2pixel(cam_coords, proj_c2p_rot=None, proj_c2p_tr=None):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
    X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
    Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
    Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.view(b,h,w,2)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=-1).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=-1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=-1).view(B, 3, 3)
    rotMat = xmat @ ymat @ zmat
    return rotMat


def inverse_warp(img, depth, pose_matrix, intrinsics, intrinsics_inv):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """

    assert(intrinsics_inv.size() == intrinsics.size())

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth)  # [B,3,H,W]

    # Get projection matrix for tgt camera frame to source pixel frame
    rot_cam_to_src_pixel = intrinsics @ pose_matrix[:,:,:-1] @ intrinsics_inv  # [B, 3, 3]
    trans_cam_to_src_pixel = intrinsics @ pose_matrix[:,:,-1:]  # [B, 3, 1]

    src_pixel_coords = cam2pixel(cam_coords, rot_cam_to_src_pixel, trans_cam_to_src_pixel)  # [B,H,W,2]
    projected_img = torch.nn.functional.grid_sample(img, src_pixel_coords)

    return projected_img
