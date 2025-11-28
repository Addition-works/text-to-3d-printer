"""
Patch for utils3d: adds depth_map_to_point_map function.

MoGe v2 (used by SAM 3D Body for FOV estimation) requires this function,
but it's not present in the older pinned version of utils3d that SAM 3D Objects needs.

This file gets copied into the utils3d/torch directory during Docker build.
"""

import torch


def depth_map_to_point_map(depth, intrinsics=None, extrinsics=None):
    """
    Unproject depth map to 3D point map.
    
    Args:
        depth: (H, W) or (B, H, W) depth map tensor
        intrinsics: (3, 3) or (B, 3, 3) camera intrinsic matrix
        extrinsics: (4, 4) or (B, 4, 4) camera extrinsic matrix (optional)
    
    Returns:
        points: (H, W, 3) or (B, H, W, 3) 3D point map
    """
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    B, H, W = depth.shape
    device = depth.device
    
    # Create pixel coordinate grid
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=depth.dtype),
        torch.arange(W, device=device, dtype=depth.dtype),
        indexing='ij'
    )
    
    # Get intrinsic parameters
    if intrinsics is None:
        # Default: assume centered principal point, fx=fy=max(H,W)
        fx = fy = float(max(H, W))
        cx, cy = W / 2.0, H / 2.0
    else:
        if intrinsics.dim() == 2:
            intrinsics = intrinsics.unsqueeze(0)
        fx = intrinsics[:, 0, 0]  # (B,)
        fy = intrinsics[:, 1, 1]  # (B,)
        cx = intrinsics[:, 0, 2]  # (B,)
        cy = intrinsics[:, 1, 2]  # (B,)
        
        # Reshape for broadcasting: (B, 1, 1)
        fx = fx.view(B, 1, 1)
        fy = fy.view(B, 1, 1)
        cx = cx.view(B, 1, 1)
        cy = cy.view(B, 1, 1)
    
    # Unproject to camera coordinates
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy  
    # Z = depth
    x = (u.unsqueeze(0) - cx) * depth / fx
    y = (v.unsqueeze(0) - cy) * depth / fy
    z = depth
    
    # Stack to (B, H, W, 3)
    points = torch.stack([x, y, z], dim=-1)
    
    # Apply extrinsics if provided (transform from camera to world coordinates)
    if extrinsics is not None:
        if extrinsics.dim() == 2:
            extrinsics = extrinsics.unsqueeze(0)
        R = extrinsics[:, :3, :3]  # (B, 3, 3)
        t = extrinsics[:, :3, 3]   # (B, 3)
        
        # points_world = R @ points + t
        points_flat = points.view(B, -1, 3)  # (B, H*W, 3)
        points_flat = torch.bmm(points_flat, R.transpose(1, 2)) + t.unsqueeze(1)
        points = points_flat.view(B, H, W, 3)
    
    if squeeze_batch:
        points = points.squeeze(0)
    
    return points