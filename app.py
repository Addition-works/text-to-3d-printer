"""
Text-to-3D-Printer Pipeline
A Gradio app that generates 3D printable models from text descriptions.

Pipeline:
1. Text prompt → Generate 4 image variants (Replicate/Stable Diffusion)
2. User selects best image
3. Auto-segment object from background (rembg)
4. User confirms mask
5. SAM 3D Objects OR SAM 3D Body reconstructs 3D model
6. Convert to STL for 3D printing
7. Preview and download
"""

import os
import sys
import tempfile
from io import BytesIO

import gradio as gr
import numpy as np
import replicate
import requests
import trimesh
from PIL import Image
from rembg import remove

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from dotenv import load_dotenv
load_dotenv()

REPLICATE_MODEL = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
IMAGE_SIZE = 1024
NUM_VARIANTS = 2  # Reduced from 4 to save costs

# SAM 3D Objects paths (set via environment variables in Docker)
SAM3D_REPO_PATH = os.environ.get("SAM3D_REPO_PATH", "/app/sam-3d-objects")
SAM3D_CHECKPOINT_PATH = os.environ.get("SAM3D_CHECKPOINT_PATH", "/app/checkpoints/checkpoints/pipeline.yaml")

# SAM 3D Body paths (set via environment variables in Docker)
SAM3D_BODY_REPO_PATH = os.environ.get("SAM3D_BODY_REPO_PATH", "/app/sam-3d-body")
SAM3D_BODY_HF_REPO = os.environ.get("SAM3D_BODY_HF_REPO", "facebook/sam-3d-body-dinov3")

# ---------------------------------------------------------------------------
# SAM 3D Objects Setup
# ---------------------------------------------------------------------------

# SAM 3D code expects CONDA_PREFIX to set CUDA_HOME - set it if not present
if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = os.environ.get("CUDA_HOME", "/usr/local/cuda")

# Add SAM 3D Objects notebook folder to path for imports
sam3d_notebook_path = os.path.join(SAM3D_REPO_PATH, "notebook")
if os.path.exists(sam3d_notebook_path):
    sys.path.insert(0, sam3d_notebook_path)
    SAM3D_OBJECTS_AVAILABLE = True
    try:
        from inference import Inference as SAM3DInference
        print("✓ SAM 3D Objects loaded successfully")
        
        # ---------------------------------------------------------------------------
        # MONKEY-PATCH: Fix mask boolean indexing issue in SAM 3D Objects
        # The _compute_scale_and_shift function uses pointmap[mask] which fails
        # when mask is uint8 (0/255) instead of boolean.
        # ---------------------------------------------------------------------------
        try:
            import torch
            from sam3d_objects.data.dataset.tdfy import img_and_mask_transforms
            
            # Find and patch all normalizer classes
            for name in dir(img_and_mask_transforms):
                cls = getattr(img_and_mask_transforms, name)
                if isinstance(cls, type) and hasattr(cls, '_compute_scale_and_shift'):
                    original_method = cls._compute_scale_and_shift
                    
                    def make_patched_method(orig):
                        def patched_compute_scale_and_shift(self, pointmap, mask):
                            # Convert mask to boolean if it's not already
                            if mask.dtype != torch.bool:
                                print(f"[MASK PATCH] Converting mask from {mask.dtype} to bool")
                                mask = mask > 0
                            
                            mask_sum = mask.sum().item()
                            print(f"[MASK PATCH] pointmap: {pointmap.shape}, mask: {mask.shape}, True pixels: {int(mask_sum)}")
                            
                            # Handle empty mask case
                            if mask_sum == 0:
                                print("[MASK PATCH] WARNING: Mask has zero foreground pixels! Returning defaults.")
                                return pointmap, torch.tensor(1.0, device=pointmap.device), torch.tensor(0.0, device=pointmap.device)
                            
                            return orig(self, pointmap, mask)
                        return patched_compute_scale_and_shift
                    
                    cls._compute_scale_and_shift = make_patched_method(original_method)
                    print(f"✓ Patched {name}._compute_scale_and_shift for boolean mask handling")
                    
        except Exception as patch_error:
            print(f"⚠ Could not apply mask patch: {patch_error}")
            
    except ImportError as e:
        print(f"✗ Failed to import SAM 3D Objects: {e}")
        SAM3D_OBJECTS_AVAILABLE = False
        SAM3DInference = None
else:
    print(f"✗ SAM 3D Objects not found at {sam3d_notebook_path}")
    SAM3D_OBJECTS_AVAILABLE = False
    SAM3DInference = None

# ---------------------------------------------------------------------------
# SAM 3D Body Setup
# ---------------------------------------------------------------------------

# Add SAM 3D Body to path
sam3d_body_path = SAM3D_BODY_REPO_PATH
if os.path.exists(sam3d_body_path):
    sys.path.insert(0, sam3d_body_path)
    # Also add notebook folder for utils
    sam3d_body_notebook_path = os.path.join(sam3d_body_path, "notebook")
    if os.path.exists(sam3d_body_notebook_path):
        sys.path.insert(0, sam3d_body_notebook_path)
    
    SAM3D_BODY_AVAILABLE = True
    try:
        from notebook.utils import setup_sam_3d_body
        print("✓ SAM 3D Body loaded successfully")
    except ImportError as e:
        print(f"✗ Failed to import SAM 3D Body: {e}")
        SAM3D_BODY_AVAILABLE = False
        setup_sam_3d_body = None
else:
    print(f"✗ SAM 3D Body not found at {sam3d_body_path}")
    SAM3D_BODY_AVAILABLE = False
    setup_sam_3d_body = None

# Global inference objects (loaded once)
_sam3d_objects_inference = None
_sam3d_body_estimator = None


def get_sam3d_objects_inference():
    """Lazy-load SAM 3D Objects inference model."""
    global _sam3d_objects_inference
    if _sam3d_objects_inference is None and SAM3D_OBJECTS_AVAILABLE:
        print("Loading SAM 3D Objects model (this may take a moment)...")
        # First arg is positional (config path), compile is keyword
        _sam3d_objects_inference = SAM3DInference(SAM3D_CHECKPOINT_PATH, compile=False)
        print("✓ SAM 3D Objects model loaded")
    return _sam3d_objects_inference


def get_sam3d_body_estimator():
    """Lazy-load SAM 3D Body estimator model."""
    global _sam3d_body_estimator
    if _sam3d_body_estimator is None and SAM3D_BODY_AVAILABLE:
        print("Loading SAM 3D Body model (this may take a moment)...")
        _sam3d_body_estimator = setup_sam_3d_body(hf_repo_id=SAM3D_BODY_HF_REPO)
        print("✓ SAM 3D Body model loaded")
    return _sam3d_body_estimator


# ---------------------------------------------------------------------------
# Step 1: Image Generation (Replicate / Stable Diffusion)
# ---------------------------------------------------------------------------

def generate_images(prompt: str, mode: str = "objects", num_images: int = NUM_VARIANTS) -> list[Image.Image]:
    """
    Generate images from a text prompt using Stable Diffusion.
    Returns a list of PIL Images.
    
    Args:
        prompt: The user's text prompt
        mode: "objects" for product-style or "body" for human poses
        num_images: Number of variants to generate
    """
    if mode == "body":
        # Enhance prompt for clear, full-body human images
        enhanced_prompt = (
            f"{prompt}, full body visible, standing pose, "
            f"plain background, studio lighting, high detail, sharp focus, "
            f"single person, no cropping, professional photo"
        )
        negative_prompt = (
            "blurry, multiple people, cluttered, busy background, "
            "text, watermark, logo, cropped body, partial body, "
            "low quality, distorted, deformed, face closeup"
        )
    else:
        # Enhance prompt for clean, product-style images that work well for 3D reconstruction
        enhanced_prompt = (
            f"{prompt}, product photography, centered in frame, "
            f"plain white background, studio lighting, high detail, sharp focus, "
            f"single object, no shadows, professional product shot"
        )
        negative_prompt = (
            "blurry, multiple objects, cluttered, busy background, "
            "text, watermark, logo, human hands, person, shadow, "
            "low quality, distorted, deformed"
        )
    
    images = []
    for i in range(num_images):
        print(f"Generating image {i + 1}/{num_images}...")
        try:
            output = replicate.run(
                REPLICATE_MODEL,
                input={
                    "prompt": enhanced_prompt,
                    "negative_prompt": negative_prompt,
                    "width": IMAGE_SIZE,
                    "height": IMAGE_SIZE,
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                    "seed": i * 42 + 12345,  # Different seed for variety
                }
            )
            
            # Output is a list of URLs
            if output and len(output) > 0:
                image_url = output[0]
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                images.append(img)
            else:
                print(f"  Warning: No output for image {i + 1}")
                
        except Exception as e:
            print(f"  Error generating image {i + 1}: {e}")
            # Create a placeholder error image
            error_img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(200, 200, 200))
            images.append(error_img)
    
    return images


# ---------------------------------------------------------------------------
# Step 2: Segmentation (rembg for background removal)
# ---------------------------------------------------------------------------

def segment_object(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """
    Remove background and create a binary mask for the object.
    Returns (RGBA image with transparent background, binary mask as numpy array).
    """
    # Use rembg to remove background
    rgba_image = remove(image)
    
    # Extract alpha channel as mask
    alpha = np.array(rgba_image)[:, :, 3]
    mask = (alpha > 128).astype(np.uint8) * 255
    
    return rgba_image, mask


def create_mask_preview(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Create a visual preview of the mask overlaid on the image."""
    img_array = np.array(image.convert("RGB"))
    
    # Create red overlay where mask is active
    overlay = img_array.copy()
    mask_bool = mask > 128
    overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    return Image.fromarray(overlay.astype(np.uint8))


# ---------------------------------------------------------------------------
# Step 3a: 3D Reconstruction (SAM 3D Objects)
# ---------------------------------------------------------------------------

def reconstruct_3d_objects(image: Image.Image, mask: np.ndarray, seed: int = 42) -> dict:
    """
    Use SAM 3D Objects to reconstruct a 3D model from image + mask.
    Returns dict with paths to output files.
    
    SAM 3D Objects inference API format (based on load_image/load_single_mask):
    - image: numpy array with shape (H, W, 3) - RGB image
    - mask: numpy array with shape (H, W) - 2D binary mask (0-255 uint8)
    
    The inference code internally calls merge_mask_to_rgba which combines them:
        rgba_image = np.concatenate([image[..., :3], mask], axis=-1)
    Note: The inference code handles adding the channel dimension to the mask.
    """
    if not SAM3D_OBJECTS_AVAILABLE:
        raise RuntimeError(
            "SAM 3D Objects is not available. "
            "Make sure you're running in the Docker container with GPU support."
        )
    
    inference = get_sam3d_objects_inference()
    if inference is None:
        raise RuntimeError("Failed to load SAM 3D Objects model")
    
    # Convert PIL Image to numpy array (H, W, 3)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_np = np.array(image)
    
    # Ensure mask is 2D (H, W) - NOT 3D!
    # load_single_mask returns: np.array(Image.open(mask_path).convert('L')) which is 2D
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    elif len(mask.shape) == 4:
        mask = mask[0, :, :, 0]
    
    # Ensure mask is uint8 with 0-255 values
    if mask.dtype != np.uint8:
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
    
    # Keep mask as 2D (H, W) - the inference code will handle adding dimension
    mask_2d = mask
    
    # Debug: Print detailed mask statistics
    foreground_pixels = np.sum(mask_2d > 128)
    total_pixels = mask_2d.size
    foreground_pct = foreground_pixels / total_pixels * 100
    
    print(f"Image shape: {image_np.shape}, dtype: {image_np.dtype}")
    print(f"Mask shape: {mask_2d.shape}, dtype: {mask_2d.dtype}")
    print(f"Mask value range: [{mask_2d.min()}, {mask_2d.max()}]")
    print(f"Mask unique values: {np.unique(mask_2d)[:10]}...")  # First 10 unique values
    print(f"Mask foreground pixels (>128): {foreground_pixels} / {total_pixels} ({foreground_pct:.1f}%)")
    
    # Save debug images
    debug_dir = "/tmp/sam3d_debug"
    os.makedirs(debug_dir, exist_ok=True)
    Image.fromarray(image_np).save(f"{debug_dir}/input_image.png")
    Image.fromarray(mask_2d).save(f"{debug_dir}/input_mask.png")
    print(f"Debug images saved to {debug_dir}")
    
    # Check if mask has enough foreground
    if foreground_pct < 1.0:
        print("WARNING: Mask has very few foreground pixels (<1%) - object may not have been detected")
        # Check if we should invert
        background_pixels = total_pixels - foreground_pixels
        if background_pixels < foreground_pixels:
            print("Mask seems inverted (more foreground than background), not inverting")
        else:
            print("Trying inverted mask...")
            mask_2d_inv = 255 - mask_2d
            fg_inv = np.sum(mask_2d_inv > 128)
            fg_inv_pct = fg_inv / total_pixels * 100
            print(f"  Inverted mask foreground: {fg_inv} ({fg_inv_pct:.1f}%)")
            if fg_inv_pct > foreground_pct and fg_inv_pct < 90:
                print("  Using inverted mask")
                mask_2d = mask_2d_inv
                foreground_pixels = fg_inv
                foreground_pct = fg_inv_pct
                Image.fromarray(mask_2d).save(f"{debug_dir}/input_mask_inverted.png")
    
    # IMPORTANT: SAM 3D internally uses the mask for:
    # 1. Creating RGBA image (mask as alpha channel, needs 0-255 range)
    # 2. Boolean indexing for pointmap normalization
    #
    # The mask from load_single_mask is strictly binary (0 or 255).
    # rembg might produce anti-aliased edges with intermediate values.
    # We ensure strictly binary values for consistency.
    
    # Convert mask to strict binary 0/255 (threshold at 128)
    mask_binary = np.where(mask_2d > 128, 255, 0).astype(np.uint8)
    print(f"Converted mask to binary 0/255: unique values = {np.unique(mask_binary)}")
    print(f"Binary mask foreground pixels: {np.sum(mask_binary == 255)}")
    
    # Save debug image
    Image.fromarray(mask_binary).save(f"{debug_dir}/input_mask_binary.png")
    
    print("Running SAM 3D Objects reconstruction...")
    
    # Pass numpy array image (H,W,3) and 2D binary mask (H,W) with values 0/255
    output = inference(image_np, mask_binary, seed=seed)
    print("✓ Reconstruction complete")
    
    # Debug: show what's in the output
    print(f"  Output keys: {list(output.keys())}")
    for key, value in output.items():
        if isinstance(value, list):
            print(f"    {key}: list of {len(value)} items")
        else:
            print(f"    {key}: {type(value).__name__}")
    
    # Save outputs to temp files
    output_dir = tempfile.mkdtemp(prefix="sam3d_objects_")
    output_paths = {}
    
    # Save Gaussian Splat PLY (may be a single object or a list)
    if "gs" in output:
        ply_path = os.path.join(output_dir, "model.ply")
        gs_data = output["gs"]
        
        if isinstance(gs_data, list):
            print(f"  Got {len(gs_data)} Gaussian splat(s)")
            if len(gs_data) > 0:
                gs_data[0].save_ply(ply_path)
                output_paths["ply"] = ply_path
                print(f"  Saved PLY: {ply_path}")
        else:
            gs_data.save_ply(ply_path)
            output_paths["ply"] = ply_path
            print(f"  Saved PLY: {ply_path}")
    
    # Save mesh - prefer 'glb' (Trimesh) over 'mesh' (MeshExtractResult)
    # The 'glb' output is a proper Trimesh object we can export directly
    if "glb" in output:
        obj_path = os.path.join(output_dir, "model.obj")
        glb_data = output["glb"]
        
        if isinstance(glb_data, list):
            print(f"  Got {len(glb_data)} GLB mesh(es)")
            if len(glb_data) > 0:
                glb_data[0].export(obj_path)
                output_paths["obj"] = obj_path
                print(f"  Saved OBJ from GLB: {obj_path}")
        else:
            glb_data.export(obj_path)
            output_paths["obj"] = obj_path
            print(f"  Saved OBJ from GLB: {obj_path}")
    
    elif "mesh" in output:
        # Fallback: try to extract trimesh from MeshExtractResult
        obj_path = os.path.join(output_dir, "model.obj")
        mesh_data = output["mesh"]
        
        if isinstance(mesh_data, list):
            mesh_data = mesh_data[0] if len(mesh_data) > 0 else None
        
        if mesh_data is not None:
            # MeshExtractResult may have a .mesh attribute or similar
            if hasattr(mesh_data, 'mesh'):
                mesh_data.mesh.export(obj_path)
                output_paths["obj"] = obj_path
                print(f"  Saved OBJ from mesh.mesh: {obj_path}")
            elif hasattr(mesh_data, 'export'):
                mesh_data.export(obj_path)
                output_paths["obj"] = obj_path
                print(f"  Saved OBJ: {obj_path}")
            else:
                print(f"  Warning: Could not export mesh (type: {type(mesh_data).__name__})")
    
    return output_paths


# ---------------------------------------------------------------------------
# Step 3b: 3D Reconstruction (SAM 3D Body)
# ---------------------------------------------------------------------------

def reconstruct_3d_body(image: Image.Image, mask: np.ndarray = None, seed: int = 42) -> dict:
    """
    Use SAM 3D Body to reconstruct a 3D human body mesh from image.
    Returns dict with paths to output files.
    
    SAM 3D Body API:
    - Uses setup_sam_3d_body() to create estimator
    - estimator.process_one_image() takes RGB numpy array
    - Returns outputs with pred_vertices and estimator.faces for mesh
    
    Note: SAM 3D Body does NOT require a mask - it detects humans automatically.
    The mask parameter is ignored but kept for API consistency.
    """
    import cv2
    
    if not SAM3D_BODY_AVAILABLE:
        raise RuntimeError(
            "SAM 3D Body is not available. "
            "Make sure you're running in the Docker container with GPU support."
        )
    
    estimator = get_sam3d_body_estimator()
    if estimator is None:
        raise RuntimeError("Failed to load SAM 3D Body model")
    
    # Convert PIL Image to numpy array (RGB)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_np = np.array(image)
    
    print(f"Image shape: {image_np.shape}, dtype: {image_np.dtype}")
    
    # Save debug image
    debug_dir = "/tmp/sam3d_debug"
    os.makedirs(debug_dir, exist_ok=True)
    Image.fromarray(image_np).save(f"{debug_dir}/body_input_image.png")
    print(f"Debug image saved to {debug_dir}")
    
    print("Running SAM 3D Body reconstruction...")
    
    # SAM 3D Body expects RGB input
    outputs = estimator.process_one_image(image_np)
    print("✓ SAM 3D Body reconstruction complete")
    
    # Debug: show what's in the output
    if isinstance(outputs, dict):
        print(f"  Output keys: {list(outputs.keys())}")
    elif isinstance(outputs, list):
        print(f"  Got {len(outputs)} detected person(s)")
        if len(outputs) > 0 and isinstance(outputs[0], dict):
            print(f"  First person keys: {list(outputs[0].keys())}")
    
    # Save outputs to temp files
    output_dir = tempfile.mkdtemp(prefix="sam3d_body_")
    output_paths = {}
    
    # Get the mesh data - outputs can be a list of dicts (one per detected person)
    # or a single dict with pred_vertices
    if isinstance(outputs, list) and len(outputs) > 0:
        # Multiple people detected, use first one
        person_output = outputs[0]
    else:
        person_output = outputs
    
    # Extract vertices and faces
    # pred_vertices contains 3D mesh vertices in camera coordinates
    if 'pred_vertices' in person_output:
        import torch
        
        vertices = person_output['pred_vertices']
        faces = estimator.faces  # The face indices are stored on the estimator
        
        # Convert to numpy if tensor
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        
        # Handle batch dimension if present
        if len(vertices.shape) == 3:
            vertices = vertices[0]  # Take first batch item
        if len(faces.shape) == 3:
            faces = faces[0]
        
        print(f"  Vertices shape: {vertices.shape}")
        print(f"  Faces shape: {faces.shape}")
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Save as PLY
        ply_path = os.path.join(output_dir, "body_model.ply")
        mesh.export(ply_path)
        output_paths["ply"] = ply_path
        print(f"  Saved PLY: {ply_path}")
        
        # Save as OBJ
        obj_path = os.path.join(output_dir, "body_model.obj")
        mesh.export(obj_path)
        output_paths["obj"] = obj_path
        print(f"  Saved OBJ: {obj_path}")
    else:
        print("  Warning: No pred_vertices in output, trying alternative extraction...")
        # Try to find mesh data in other formats
        for key in ['vertices', 'mesh', 'body_mesh']:
            if key in person_output:
                print(f"  Found alternative key: {key}")
                break
        else:
            raise RuntimeError("Could not find mesh vertices in SAM 3D Body output")
    
    return output_paths


# ---------------------------------------------------------------------------
# Unified reconstruction function
# ---------------------------------------------------------------------------

def reconstruct_3d(image: Image.Image, mask: np.ndarray, mode: str = "objects", seed: int = 42) -> dict:
    """
    Unified 3D reconstruction that dispatches to Objects or Body model.
    
    Args:
        image: Input PIL Image
        mask: Binary mask (used by Objects, ignored by Body)
        mode: "objects" or "body"
        seed: Random seed
    
    Returns dict with paths to output files.
    """
    if mode == "body":
        return reconstruct_3d_body(image=image, mask=mask, seed=seed)
    else:
        return reconstruct_3d_objects(image=image, mask=mask, seed=seed)


# ---------------------------------------------------------------------------
# Step 4: Mesh Conversion (PLY/OBJ → STL)
# ---------------------------------------------------------------------------

def convert_to_stl(input_path: str, output_path: str = None) -> str:
    """
    Convert a 3D model (PLY, OBJ, GLB) to STL format for 3D printing.
    Returns path to the STL file.
    """
    if output_path is None:
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, "model.stl")
    
    print(f"Converting {input_path} to STL...")
    
    # Load mesh with trimesh
    mesh = trimesh.load(input_path)
    
    # If it's a scene (multiple meshes), concatenate them
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise ValueError("No valid meshes found in the file")
    
    # Basic mesh repair for 3D printing
    if hasattr(mesh, 'fill_holes'):
        mesh.fill_holes()
    if hasattr(mesh, 'fix_normals'):
        mesh.fix_normals()
    
    # Export to STL
    mesh.export(output_path, file_type="stl")
    print(f"✓ Saved STL: {output_path}")
    
    return output_path


def convert_to_glb(input_path: str, output_path: str = None) -> str:
    """Convert mesh to GLB format for web viewing."""
    if output_path is None:
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, "model.glb")
    
    mesh = trimesh.load(input_path)
    
    # Flip vertically (rotate 180° around X-axis) - SAM 3D outputs are often inverted
    # This rotation matrix flips Y and Z coordinates
    flip_matrix = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])
    
    if isinstance(mesh, trimesh.Scene):
        # Apply transform to each geometry in the scene
        for name, geometry in mesh.geometry.items():
            if hasattr(geometry, 'apply_transform'):
                geometry.apply_transform(flip_matrix)
        mesh.export(output_path, file_type="glb")
    else:
        mesh.apply_transform(flip_matrix)
        scene = trimesh.Scene([mesh])
        scene.export(output_path, file_type="glb")
    
    return output_path


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

# State to hold data between steps
class PipelineState:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.prompt = ""
        self.mode = "objects"  # "objects" or "body"
        self.generated_images = []
        self.selected_image = None
        self.selected_index = None
        self.mask = None
        self.rgba_image = None
        self.output_paths = {}


state = PipelineState()


def step1_generate(prompt: str, mode: str):
    """Generate images from prompt."""
    if not prompt.strip():
        raise gr.Error("Please enter a description of what you want to create.")
    
    # Validate mode selection against available models
    if mode == "body" and not SAM3D_BODY_AVAILABLE:
        raise gr.Error(
            "SAM 3D Body is not available in this installation. "
            "Please select 'Objects' mode or run with full Docker setup."
        )
    if mode == "objects" and not SAM3D_OBJECTS_AVAILABLE:
        raise gr.Error(
            "SAM 3D Objects is not available in this installation. "
            "Please select 'Human Body' mode or run with full Docker setup."
        )
    
    state.reset()
    state.prompt = prompt
    state.mode = mode
    
    gr.Info("Generating images... This may take 30-60 seconds.")
    state.generated_images = generate_images(prompt=prompt, mode=mode, num_images=NUM_VARIANTS)
    
    return state.generated_images


def step2_select(evt: gr.SelectData):
    """Handle image selection from gallery."""
    state.selected_index = evt.index
    state.selected_image = state.generated_images[evt.index]
    
    # Auto-segment the selected image
    gr.Info("Segmenting object from background...")
    state.rgba_image, state.mask = segment_object(state.selected_image)
    
    # Create preview
    preview = create_mask_preview(state.selected_image, state.mask)
    
    return state.selected_image, preview, gr.update(visible=True)


def step3_regenerate_mask():
    """Regenerate mask (placeholder for manual adjustment)."""
    if state.selected_image is None:
        raise gr.Error("Please select an image first.")
    
    # For now, just re-run segmentation
    # In a more advanced version, you could allow manual mask editing
    state.rgba_image, state.mask = segment_object(state.selected_image)
    preview = create_mask_preview(state.selected_image, state.mask)
    
    return preview


def step4_reconstruct():
    """Run 3D reconstruction."""
    if state.selected_image is None:
        raise gr.Error("Please select an image first.")
    
    # Body mode doesn't strictly require mask, but Objects mode does
    if state.mode == "objects" and state.mask is None:
        raise gr.Error("Please select an image and confirm the mask first.")
    
    # Check model availability
    if state.mode == "body" and not SAM3D_BODY_AVAILABLE:
        raise gr.Error(
            "SAM 3D Body is not available. "
            "Please run this app in the Docker container with GPU support."
        )
    if state.mode == "objects" and not SAM3D_OBJECTS_AVAILABLE:
        raise gr.Error(
            "SAM 3D Objects is not available. "
            "Please run this app in the Docker container with GPU support."
        )
    
    mode_name = "Human Body" if state.mode == "body" else "Object"
    gr.Info(f"Reconstructing 3D {mode_name}... This may take 30-60 seconds.")
    
    try:
        state.output_paths = reconstruct_3d(
            image=state.selected_image, 
            mask=state.mask, 
            mode=state.mode
        )
        
        # Convert to formats we need
        if "ply" in state.output_paths:
            # Convert PLY to GLB for viewing
            glb_path = convert_to_glb(state.output_paths["ply"])
            state.output_paths["glb"] = glb_path
            
            # Convert to STL for 3D printing
            stl_path = convert_to_stl(state.output_paths["ply"])
            state.output_paths["stl"] = stl_path
        
        elif "obj" in state.output_paths:
            glb_path = convert_to_glb(state.output_paths["obj"])
            state.output_paths["glb"] = glb_path
            
            stl_path = convert_to_stl(state.output_paths["obj"])
            state.output_paths["stl"] = stl_path
        
        # Return GLB for 3D viewer
        glb_path = state.output_paths.get("glb")
        stl_path = state.output_paths.get("stl")
        
        return (
            glb_path,
            gr.update(value=stl_path, visible=True),
            gr.update(visible=True)
        )
        
    except Exception as e:
        raise gr.Error(f"3D reconstruction failed: {str(e)}")


def step5_download_stl():
    """Get STL file for download."""
    if "stl" not in state.output_paths:
        raise gr.Error("No STL file available. Please run reconstruction first.")
    return state.output_paths["stl"]


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def create_ui():
    with gr.Blocks(
        title="Text to 3D Printer",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # 🖨️ Text to 3D Printer
            
            Transform your ideas into 3D printable models in 4 simple steps:
            1. **Choose mode** - Objects (products, items) or Human Body (poses)
            2. **Describe** what you want to create
            3. **Select** the best generated image
            4. **Download** your 3D printable STL file
            """
        )
        
        # Step 1: Mode Selection & Text Input
        with gr.Group():
            gr.Markdown("### Step 1: Choose Mode & Describe")
            with gr.Row():
                mode_selector = gr.Radio(
                    choices=[
                        ("🎁 Objects (products, items, things)", "objects"),
                        ("🧍 Human Body (poses, figures)", "body")
                    ],
                    value="objects",
                    label="What do you want to 3D print?",
                    interactive=True,
                )
            with gr.Row():
                prompt_input = gr.Textbox(
                    label="Describe your object or person",
                    placeholder="e.g., a ceramic coffee mug with geometric patterns",
                    lines=2,
                    scale=4,
                )
                generate_btn = gr.Button("🎨 Generate Images", variant="primary", scale=1)
            
            # Show availability status
            status_parts = []
            if SAM3D_OBJECTS_AVAILABLE:
                status_parts.append("✓ Objects mode available")
            else:
                status_parts.append("✗ Objects mode not available")
            if SAM3D_BODY_AVAILABLE:
                status_parts.append("✓ Body mode available")
            else:
                status_parts.append("✗ Body mode not available")
            gr.Markdown(f"*Model status: {' | '.join(status_parts)}*")
        
        # Step 2: Image Selection
        with gr.Group():
            gr.Markdown("### Step 2: Select an Image")
            gr.Markdown("*Click on the image you like best*")
            gallery = gr.Gallery(
                label="Generated Images",
                columns=2,
                rows=1,
                height="auto",
                object_fit="contain",
                allow_preview=False,
            )
        
        # Step 3: Mask Confirmation
        with gr.Group(visible=False) as mask_group:
            gr.Markdown("### Step 3: Confirm Selection")
            gr.Markdown("*The red overlay shows what will be converted to 3D*")
            with gr.Row():
                selected_image = gr.Image(label="Selected Image", type="pil")
                mask_preview = gr.Image(label="Object Mask Preview", type="pil")
            with gr.Row():
                regenerate_btn = gr.Button("🔄 Regenerate Mask", variant="secondary")
                confirm_btn = gr.Button("✅ Confirm & Generate 3D", variant="primary")
        
        # Step 4: 3D Preview & Download
        with gr.Group(visible=False) as result_group:
            gr.Markdown("### Step 4: Your 3D Model")
            model_viewer = gr.Model3D(
                label="3D Preview (drag to rotate)",
                clear_color=[0.15, 0.15, 0.15, 1.0],  # Dark background for better visibility
            )
            with gr.Row():
                stl_download = gr.File(label="📥 Download STL for 3D Printing", visible=False)
            gr.Markdown(
                """
                **Tips for 3D printing:**
                - Import the STL file into your slicer software (Cura, PrusaSlicer, etc.)
                - You may need to scale the model to your desired size
                - Check for any mesh errors using your slicer's repair tools
                """
            )
        
        # Wire up events
        generate_btn.click(
            fn=step1_generate,
            inputs=[prompt_input, mode_selector],
            outputs=[gallery],
        )
        
        gallery.select(
            fn=step2_select,
            inputs=[],
            outputs=[selected_image, mask_preview, mask_group],
        )
        
        regenerate_btn.click(
            fn=step3_regenerate_mask,
            inputs=[],
            outputs=[mask_preview],
        )
        
        confirm_btn.click(
            fn=step4_reconstruct,
            inputs=[],
            outputs=[model_viewer, stl_download, result_group],
        )
    
    return app


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("✗ No GPU detected by PyTorch")
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
    
    # Check for required environment variables
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("⚠️  Warning: REPLICATE_API_TOKEN not set. Image generation will fail.")
    
    # Print model availability
    print("\n--- Model Availability ---")
    print(f"SAM 3D Objects: {'✓ Available' if SAM3D_OBJECTS_AVAILABLE else '✗ Not available'}")
    print(f"SAM 3D Body: {'✓ Available' if SAM3D_BODY_AVAILABLE else '✗ Not available'}")
    print("--------------------------\n")
    
    # Create and launch the app
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 8080)),
        share=False,
    )