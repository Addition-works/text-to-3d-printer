"""
Text-to-3D-Printer Pipeline
A Gradio app that generates 3D printable models from text descriptions.

Pipeline:
1. Text prompt → Generate 4 image variants (Replicate/Stable Diffusion)
2. User selects best image
3. Auto-segment object from background (rembg)
4. User confirms mask
5. SAM 3D Objects reconstructs 3D model
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
NUM_VARIANTS = 4

# SAM 3D Objects paths (set via environment variables in Docker)
SAM3D_REPO_PATH = os.environ.get("SAM3D_REPO_PATH", "/app/sam-3d-objects")
SAM3D_CHECKPOINT_PATH = os.environ.get("SAM3D_CHECKPOINT_PATH", "/app/checkpoints/hf/pipeline.yaml")

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
    SAM3D_AVAILABLE = True
    try:
        from inference import Inference as SAM3DInference
        print("✓ SAM 3D Objects loaded successfully")
    except ImportError as e:
        print(f"✗ Failed to import SAM 3D Objects: {e}")
        SAM3D_AVAILABLE = False
        SAM3DInference = None
else:
    print(f"✗ SAM 3D Objects not found at {sam3d_notebook_path}")
    SAM3D_AVAILABLE = False
    SAM3DInference = None

# Global inference object (loaded once)
_sam3d_inference = None


def get_sam3d_inference():
    """Lazy-load SAM 3D inference model."""
    global _sam3d_inference
    if _sam3d_inference is None and SAM3D_AVAILABLE:
        print("Loading SAM 3D Objects model (this may take a moment)...")
        _sam3d_inference = SAM3DInference(config_path=SAM3D_CHECKPOINT_PATH, compile=False)
        print("✓ SAM 3D Objects model loaded")
    return _sam3d_inference


# ---------------------------------------------------------------------------
# Step 1: Image Generation (Replicate / Stable Diffusion)
# ---------------------------------------------------------------------------

def generate_images(prompt: str, num_images: int = NUM_VARIANTS) -> list[Image.Image]:
    """
    Generate product-style images from a text prompt using Stable Diffusion.
    Returns a list of PIL Images.
    """
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
# Step 3: 3D Reconstruction (SAM 3D Objects)
# ---------------------------------------------------------------------------

def reconstruct_3d(image: Image.Image, mask: np.ndarray, seed: int = 42) -> dict:
    """
    Use SAM 3D Objects to reconstruct a 3D model from image + mask.
    Returns dict with paths to output files.
    """
    if not SAM3D_AVAILABLE:
        raise RuntimeError(
            "SAM 3D Objects is not available. "
            "Make sure you're running in the Docker container with GPU support."
        )
    
    inference = get_sam3d_inference()
    if inference is None:
        raise RuntimeError("Failed to load SAM 3D Objects model")
    
    # Ensure image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Ensure mask is the right shape and type
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    # Normalize mask to 0-1 range for SAM 3D
    mask_normalized = (mask > 128).astype(np.float32)
    
    print("Running SAM 3D Objects reconstruction...")
    output = inference(image, mask_normalized, seed=seed)
    print("✓ Reconstruction complete")
    
    # Save outputs to temp files
    output_dir = tempfile.mkdtemp(prefix="sam3d_")
    output_paths = {}
    
    # Save Gaussian Splat PLY
    if "gs" in output:
        ply_path = os.path.join(output_dir, "model.ply")
        output["gs"].save_ply(ply_path)
        output_paths["ply"] = ply_path
        print(f"  Saved PLY: {ply_path}")
    
    # Save mesh if available
    if "mesh" in output:
        obj_path = os.path.join(output_dir, "model.obj")
        output["mesh"].export(obj_path)
        output_paths["obj"] = obj_path
        print(f"  Saved OBJ: {obj_path}")
    
    return output_paths


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
    
    if isinstance(mesh, trimesh.Scene):
        mesh.export(output_path, file_type="glb")
    else:
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
        self.generated_images = []
        self.selected_image = None
        self.selected_index = None
        self.mask = None
        self.rgba_image = None
        self.output_paths = {}


state = PipelineState()


def step1_generate(prompt: str):
    """Generate images from prompt."""
    if not prompt.strip():
        raise gr.Error("Please enter a description of the object you want to create.")
    
    state.reset()
    state.prompt = prompt
    
    gr.Info("Generating images... This may take 30-60 seconds.")
    state.generated_images = generate_images(prompt, num_images=NUM_VARIANTS)
    
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
    if state.selected_image is None or state.mask is None:
        raise gr.Error("Please select an image and confirm the mask first.")
    
    if not SAM3D_AVAILABLE:
        raise gr.Error(
            "SAM 3D Objects is not available. "
            "Please run this app in the Docker container with GPU support."
        )
    
    gr.Info("Reconstructing 3D model... This may take 30-60 seconds.")
    
    try:
        state.output_paths = reconstruct_3d(state.selected_image, state.mask)
        
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
            1. **Describe** the object you want to create
            2. **Select** the best generated image
            3. **Confirm** the object mask
            4. **Download** your 3D printable STL file
            """
        )
        
        # Step 1: Text Input & Image Generation
        with gr.Group():
            gr.Markdown("### Step 1: Describe Your Object")
            with gr.Row():
                prompt_input = gr.Textbox(
                    label="What do you want to 3D print?",
                    placeholder="e.g., a ceramic coffee mug with geometric patterns",
                    lines=2,
                    scale=4,
                )
                generate_btn = gr.Button("🎨 Generate Images", variant="primary", scale=1)
        
        # Step 2: Image Selection
        with gr.Group():
            gr.Markdown("### Step 2: Select an Image")
            gr.Markdown("*Click on the image you like best*")
            gallery = gr.Gallery(
                label="Generated Images",
                columns=4,
                rows=1,
                height="auto",
                object_fit="contain",
                allow_preview=False,
            )
        
        # Step 3: Mask Confirmation
        with gr.Group(visible=False) as mask_group:
            gr.Markdown("### Step 3: Confirm Object Mask")
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
                clear_color=[0.9, 0.9, 0.9, 1.0],
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
            inputs=[prompt_input],
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
    
    # Create and launch the app
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 8080)),
        share=False,
    )