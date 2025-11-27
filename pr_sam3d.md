SAM 3D Objects Integration Fixes - Summary for PR
Overview
While integrating SAM 3D Objects into a pipeline, we encountered several issues that required workarounds. These fixes may be useful for the SAM 3D Objects repository to improve compatibility and ease of integration.

Issue 1: Mask Boolean Indexing Bug
Problem
The _compute_scale_and_shift method in sam3d_objects/data/dataset/tdfy/img_and_mask_transforms.py fails when the mask tensor is float32 or uint8 instead of bool.
Error:
RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0. 
Specify the reduction dim with the 'dim' argument.
Location: Line ~550 in img_and_mask_transforms.py
pythonmask_points = pointmap[mask]  # Fails when mask is not boolean
if mask_points.isfinite().max() == 0:  # Crashes because mask_points is empty
Root Cause
When the mask tensor has dtype float32 or uint8, PyTorch interprets the indexing pointmap[mask] differently than boolean indexing. Instead of selecting elements where mask is True, it treats the mask values as indices, resulting in an empty tensor.
Fix
Convert mask to boolean before indexing:
pythondef _compute_scale_and_shift(self, pointmap, mask):
    # Ensure mask is boolean for proper indexing
    if mask.dtype != torch.bool:
        mask = mask > 0
    
    # Rest of the method...
    mask_points = pointmap[mask]
Suggested PR
Add a boolean conversion at the start of _compute_scale_and_shift (and potentially other methods that use mask indexing).

Issue 2: Output Structure Documentation
Problem
The inference output dictionary structure is not well-documented, leading to confusion about how to access results.
Actual output structure:
python{
    'gs': Gaussian,              # Gaussian splat object - use .save_ply()
    'gaussian': list[Gaussian],  # List wrapper
    'glb': Trimesh,              # ← Proper Trimesh object for mesh export
    'mesh': list[MeshExtractResult],  # Internal type, NOT directly exportable
    'pointmap': Tensor,
    'pointmap_colors': Tensor,
    'rotation': Tensor,
    'translation': Tensor,
    'scale': Tensor,
    # ... other tensors
}
Key Findings

output["mesh"] is a MeshExtractResult (internal type) that does NOT have an .export() method
output["glb"] is a Trimesh object that CAN be exported directly
output["gs"] is a single Gaussian object with .save_ply() method
output["gaussian"] is a list containing the same Gaussian(s)

Suggested PR
Update documentation/README to clarify:
python# Export Gaussian splat
output["gs"].save_ply("model.ply")

# Export mesh (use 'glb', not 'mesh')
output["glb"].export("model.obj")  # or .glb, .stl, etc.
```

---

## Issue 3: utils3d Dependency Conflict

### Problem
MoGe (depth estimation) requires a specific version of `utils3d` from GitHub (EasternJournalist/utils3d), but PyPI has a different, incompatible package with the same name.

**Error:**
```
ModuleNotFoundError: No module named 'utils3d.pt'
Root Cause

MoGe specifies: utils3d @ git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6...
PyPI has a different utils3d package (point cloud utilities by different author)
Installing other packages can pull in the wrong utils3d from PyPI, overwriting the correct one

Fix

Install the correct utils3d after all other pip installs:

bashpip install "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38"

Create module aliases for pt and np:

python# In utils3d/pt.py
def __getattr__(name):
    from . import torch
    return getattr(torch, name)
```

### Suggested PR
- Pin the exact utils3d dependency more explicitly in setup/requirements
- Add installation troubleshooting note about PyPI conflict
- Consider renaming internal imports to avoid collision

---

## Issue 4: nvdiffrast Dependency

### Problem
`utils3d.torch.rasterization` requires `nvdiffrast`, which is not on PyPI.

**Error:**
```
ModuleNotFoundError: No module named 'nvdiffrast'
Fix
Install from NVIDIA's GitHub:
bashpip install git+https://github.com/NVlabs/nvdiffrast.git
Suggested PR
Add nvdiffrast to documented dependencies or optional dependencies.

Summary of Runtime Workaround
For anyone integrating SAM 3D Objects, here's the monkey-patch we used:
pythonimport torch
from sam3d_objects.data.dataset.tdfy import img_and_mask_transforms

# Patch all normalizer classes that have _compute_scale_and_shift
for name in dir(img_and_mask_transforms):
    cls = getattr(img_and_mask_transforms, name)
    if isinstance(cls, type) and hasattr(cls, '_compute_scale_and_shift'):
        original_method = cls._compute_scale_and_shift
        
        def make_patched_method(orig):
            def patched(self, pointmap, mask):
                if mask.dtype != torch.bool:
                    mask = mask > 0
                if mask.sum() == 0:
                    # Handle empty mask gracefully
                    return pointmap, torch.tensor(1.0, device=pointmap.device), torch.tensor(0.0, device=pointmap.device)
                return orig(self, pointmap, mask)
            return patched
        
        cls._compute_scale_and_shift = make_patched_method(original_method)

Environment

SAM 3D Objects version: Latest as of Nov 2025 (commit from HuggingFace)
Python: 3.11
PyTorch: 2.5.1
CUDA: 12.4