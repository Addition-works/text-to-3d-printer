"""
Text-to-3D-Printer Pipeline
A Gradio app that generates 3D printable models from text descriptions.

Pipeline:
1. Text prompt -> Generate image variants (Replicate/Nano Banana)
2. User selects best image
3. Trellis 2 reconstructs 3D model with PBR materials (handles background removal internally)
4. Export to GLB/GLTF and 3MF for 3D printing
5. Preview and download
"""

import base64
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import gradio as gr
import numpy as np
import replicate
import requests
import trimesh
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from dotenv import load_dotenv
load_dotenv()

REPLICATE_MODEL = "google/nano-banana"
IMAGE_SIZE = 1024
NUM_VARIANTS = 2  # Reduced from 4 to save costs

# Default system prompt for 3D-friendly image generation
DEFAULT_SYSTEM_PROMPT = (
    "Product photography style, angled view, isolated object, "
    "plain white background, studio lighting, high detail, sharp focus, drop shadow, "
    "single object, professional product shot, clean edges, suitable for 3D reconstruction. "
    "No thin surfaces. Simple colors and textures (i.e. no color gradients or complex geometry/topology). "
    "Very 3d-printer friendly."
)

# Trellis 2 model path (set via environment variable in Docker)
TRELLIS_MODEL_PATH = os.environ.get("TRELLIS_MODEL_PATH", "/app/models/TRELLIS.2-4B")

# ---------------------------------------------------------------------------
# Trellis 2 Setup
# ---------------------------------------------------------------------------

# Set environment variables for Trellis 2
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

TRELLIS_AVAILABLE = False
_trellis_pipeline = None

try:
    import torch
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel
    TRELLIS_AVAILABLE = True
    print("Trellis 2 modules loaded successfully")
except ImportError as e:
    print(f"Failed to import Trellis 2: {e}")
    Trellis2ImageTo3DPipeline = None
    o_voxel = None


def get_trellis_pipeline():
    """Lazy-load Trellis 2 pipeline."""
    global _trellis_pipeline
    if _trellis_pipeline is None and TRELLIS_AVAILABLE:
        print("Loading Trellis 2 model (this may take a moment)...")

        # Set HF token for authentication (needed for gated models like DINOv3)
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            print("Using HF_TOKEN for authentication")
            # Set token for huggingface_hub
            try:
                from huggingface_hub import login
                login(token=hf_token, add_to_git_credential=False)
            except Exception as e:
                print(f"Warning: Could not login to HuggingFace: {e}")

        try:
            # Load from local path if available, otherwise from HuggingFace
            if os.path.exists(TRELLIS_MODEL_PATH) and os.path.isdir(TRELLIS_MODEL_PATH):
                print(f"Loading from local path: {TRELLIS_MODEL_PATH}")
                _trellis_pipeline = Trellis2ImageTo3DPipeline.from_pretrained(TRELLIS_MODEL_PATH)
            else:
                print("Loading from HuggingFace: microsoft/TRELLIS.2-4B")
                _trellis_pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
            _trellis_pipeline.cuda()
            print("Trellis 2 model loaded successfully")
        except Exception as e:
            print(f"Failed to load Trellis 2 model: {e}")
            return None
    return _trellis_pipeline


# ---------------------------------------------------------------------------
# Step 1: Image Generation (Replicate / Nano Banana)
# ---------------------------------------------------------------------------

def upload_image_to_temp_url(image: Image.Image) -> str:
    """
    Convert a PIL Image to a data URI for use with Replicate.
    Nano Banana accepts base64 data URIs directly.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


def generate_images(
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    reference_images: list[Image.Image] | None = None,
    num_images: int = NUM_VARIANTS
) -> list[Image.Image]:
    """
    Generate images from a text prompt using Nano Banana.
    Returns a list of PIL Images.

    Args:
        prompt: The user's text prompt
        system_prompt: Additional context/style instructions
        reference_images: Optional list of reference images (up to 5)
        num_images: Number of variants to generate
    """
    # Combine user prompt with system prompt for nano-banana
    # Nano Banana works best with descriptive, conversational prompts
    full_prompt = f"{prompt}. {system_prompt}"

    # Prepare reference images if provided
    image_inputs = []
    if reference_images:
        for img in reference_images[:5]:  # Limit to 5 images
            if img is not None:
                # Convert to data URI
                data_uri = upload_image_to_temp_url(img)
                image_inputs.append(data_uri)

    images = []
    for i in range(num_images):
        print(f"Generating image {i + 1}/{num_images}...")
        try:
            # Build input dict for nano-banana
            input_dict = {
                "prompt": full_prompt,
                "output_format": "png",
            }

            # Add reference images if provided
            if image_inputs:
                input_dict["image_input"] = image_inputs

            output = replicate.run(
                REPLICATE_MODEL,
                input=input_dict
            )

            print(f"  Output type: {type(output)}, value: {output}")

            # nano-banana returns a single URI string (not a list)
            # But also handle FileOutput objects from replicate library
            if output:
                # Get the URL from the output
                if hasattr(output, 'url'):
                    # FileOutput object
                    image_url = output.url
                elif hasattr(output, 'read'):
                    # File-like object - read directly
                    img = Image.open(output).convert("RGB")
                    images.append(img)
                    continue
                elif isinstance(output, str):
                    # Direct URL string
                    image_url = output
                else:
                    print(f"  Warning: Unexpected output format for image {i + 1}: {type(output)}")
                    print(f"  Output value: {output}")
                    continue

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
# Step 2: 3D Reconstruction (Trellis 2)
# ---------------------------------------------------------------------------

def reconstruct_3d(image: Image.Image, seed: int = 42) -> dict:
    """
    Use Trellis 2 to reconstruct a 3D model from an image.

    Trellis 2 handles background removal internally via rembg during preprocessing.
    It generates a full 3D mesh with PBR materials (base color, roughness, metallic, opacity).

    Returns dict with paths to output files.
    """
    if not TRELLIS_AVAILABLE:
        raise RuntimeError(
            "Trellis 2 is not available. "
            "Make sure you're running in the Docker container with GPU support."
        )

    pipeline = get_trellis_pipeline()
    if pipeline is None:
        raise RuntimeError("Failed to load Trellis 2 model")

    # Debug directory
    debug_dir = "/tmp/trellis_debug"
    os.makedirs(debug_dir, exist_ok=True)

    # Save debug image
    image.save(f"{debug_dir}/input_image.png")
    print(f"Debug images saved to {debug_dir}")

    # Preprocess image for Trellis 2
    print("Preprocessing image for Trellis 2...")
    processed_image = pipeline.preprocess_image(image)
    processed_image.save(f"{debug_dir}/preprocessed_image.png")

    print("Running Trellis 2 reconstruction...")

    # Run the pipeline with default parameters optimized for quality
    import torch
    torch.manual_seed(seed)

    try:
        # Run Trellis 2 inference with return_latent=True
        # Returns: (outputs, latents) where:
        #   - outputs is a list of MeshWithVoxel objects for preview rendering
        #   - latents is a tuple (shape_slat, tex_slat, res) for GLB export
        outputs, latents = pipeline.run(
            processed_image,
            seed=seed,
            return_latent=True,  # Required to get latents for proper GLB export
        )

        mesh = outputs[0]
        shape_slat, tex_slat, res = latents

        print(f"Trellis 2 reconstruction complete (resolution: {res})")

        # Simplify mesh to respect nvdiffrast limits
        mesh.simplify(16777216)

    except Exception as e:
        print(f"Trellis 2 inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"3D reconstruction failed: {e}")

    # Save outputs to temp files
    output_dir = tempfile.mkdtemp(prefix="trellis2_")
    output_paths = {}

    # ---------------------------------------------------------------------------
    # Export to GLB (with full PBR materials and colors)
    # ---------------------------------------------------------------------------
    try:
        glb_path = os.path.join(output_dir, "model.glb")

        # Decode latents to get the mesh with proper attributes for GLB export
        print("Decoding latents for GLB export...")
        glb_mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]

        # Use o_voxel for high-quality GLB export with PBR materials
        print("Exporting GLB with PBR materials...")
        glb = o_voxel.postprocess.to_glb(
            vertices=glb_mesh.vertices,
            faces=glb_mesh.faces,
            attr_volume=glb_mesh.attrs,
            coords=glb_mesh.coords,
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=res,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            texture_size=2048,  # Good balance of quality and file size
            decimation_target=500000,  # 500k faces for good quality
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=True,
        )

        # Export with PNG textures (WebP not supported by PyMeshLab for 3MF conversion)
        glb.export(glb_path, extension_webp=False)
        output_paths["glb"] = glb_path
        print(f"  Saved GLB: {glb_path}")

        # Clean up GPU memory
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  Warning: GLB export with o_voxel failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: try direct trimesh export if mesh has vertices/faces
        try:
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                glb_path = os.path.join(output_dir, "model.glb")
                tm = trimesh.Trimesh(
                    vertices=mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, 'cpu') else mesh.vertices,
                    faces=mesh.faces.cpu().numpy() if hasattr(mesh.faces, 'cpu') else mesh.faces
                )
                tm.export(glb_path, file_type="glb")
                output_paths["glb"] = glb_path
                print(f"  Saved GLB (fallback - no colors): {glb_path}")
        except Exception as e2:
            print(f"  GLB fallback also failed: {e2}")

    return output_paths


# ---------------------------------------------------------------------------
# Step 4: Mesh Conversion (GLB/OBJ -> STL, 3MF)
# ---------------------------------------------------------------------------

def convert_to_stl(input_path: str, output_path: str = None) -> str:
    """
    Convert a 3D model (GLB, OBJ) to STL format for 3D printing.
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
    print(f"  Saved STL: {output_path}")

    return output_path


def convert_to_3mf(input_path: str, output_path: str = None) -> str:
    """
    Convert a GLB model to 3MF format with full vertex colors using lib3mf.

    lib3mf is the official 3MF Consortium library and properly supports
    vertex colors via ColorGroups, which PyMeshLab does not.

    Returns path to the 3MF file.
    Raises an exception if conversion fails or color data is lost.
    """
    if output_path is None:
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, "model.3mf")

    print(f"Converting {input_path} to 3MF with lib3mf...")

    import lib3mf
    from lib3mf import get_wrapper

    # Load the GLB with trimesh to get vertices, faces, and colors
    # force='mesh' converts Scene to single Trimesh by concatenating all geometries
    mesh = trimesh.load(input_path, force='mesh')

    print(f"  Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # Get vertex colors - handle both vertex color and texture-based visuals
    vertex_colors = None

    # Check for direct vertex colors first
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        vc = mesh.visual.vertex_colors
        # Verify it's not all the same default color
        if len(vc) == len(mesh.vertices) and not np.all(vc == vc[0]):
            vertex_colors = vc
            print(f"  Has vertex colors: True ({len(vertex_colors)} colors)")

    # If no vertex colors, try to convert from texture
    if vertex_colors is None and hasattr(mesh.visual, 'to_color'):
        print("  Converting texture to vertex colors...")
        try:
            color_visual = mesh.visual.to_color()
            if hasattr(color_visual, 'vertex_colors') and color_visual.vertex_colors is not None:
                vertex_colors = color_visual.vertex_colors
                print(f"  Converted to vertex colors: {len(vertex_colors)} colors")
        except Exception as e:
            print(f"  Warning: Could not convert texture to vertex colors: {e}")

    if vertex_colors is None:
        raise RuntimeError(
            "GLB file has no vertex colors and texture conversion failed. "
            "Check that O-Voxel exported colors correctly."
        )

    # Create lib3mf model using recommended get_wrapper() function
    wrapper = get_wrapper()
    model = wrapper.CreateModel()

    # Create mesh object
    mesh_object = model.AddMeshObject()
    mesh_object.SetName("ColoredMesh")

    # Add vertices - lib3mf Position uses .Coordinates array
    vertices = mesh.vertices.astype(np.float64)
    for v in vertices:
        pos = lib3mf.Position()
        pos.Coordinates[0] = float(v[0])
        pos.Coordinates[1] = float(v[1])
        pos.Coordinates[2] = float(v[2])
        mesh_object.AddVertex(pos)

    # Add triangles - lib3mf Triangle uses .Indices array
    faces = mesh.faces.astype(np.uint32)
    for f in faces:
        tri = lib3mf.Triangle()
        tri.Indices[0] = int(f[0])
        tri.Indices[1] = int(f[1])
        tri.Indices[2] = int(f[2])
        mesh_object.AddTriangle(tri)

    # Create color group for vertex colors
    color_group = model.AddColorGroup()

    # Build a map of unique colors to color IDs to avoid duplicates
    # Colors are RGBA 0-255, convert to lib3mf Color format
    color_to_id = {}

    def get_color_id(rgba):
        """Get or create a color ID for the given RGBA tuple."""
        key = tuple(rgba[:3])  # Use RGB as key (ignore alpha for now)
        if key not in color_to_id:
            # lib3mf Color uses .Red, .Green, .Blue, .Alpha fields
            color = lib3mf.Color()
            color.Red = int(rgba[0])
            color.Green = int(rgba[1])
            color.Blue = int(rgba[2])
            color.Alpha = int(rgba[3]) if len(rgba) > 3 else 255
            color_id = color_group.AddColor(color)
            color_to_id[key] = color_id
        return color_to_id[key]

    # Set triangle properties with per-vertex colors
    print("  Assigning vertex colors to triangles...")
    resource_id = color_group.GetResourceID()

    for i, face in enumerate(faces):
        # Get the color for each vertex of this triangle
        c0 = vertex_colors[face[0]]
        c1 = vertex_colors[face[1]]
        c2 = vertex_colors[face[2]]

        id0 = get_color_id(c0)
        id1 = get_color_id(c1)
        id2 = get_color_id(c2)

        # Create triangle properties - uses .ResourceID and .PropertyIDs array
        props = lib3mf.TriangleProperties()
        props.ResourceID = resource_id
        props.PropertyIDs[0] = id0
        props.PropertyIDs[1] = id1
        props.PropertyIDs[2] = id2
        mesh_object.SetTriangleProperties(i, props)

    print(f"  Created {len(color_to_id)} unique colors")

    # Set object-level property (required for colors to work)
    mesh_object.SetObjectLevelProperty(resource_id, 1)

    # Add mesh to build items
    model.AddBuildItem(mesh_object, wrapper.GetIdentityTransform())

    # Write to file
    writer = model.QueryWriter("3mf")
    writer.WriteToFile(output_path)

    print(f"  Saved 3MF with vertex colors: {output_path}")

    return output_path


def convert_to_3mf_printable(input_path: str, output_path: str = None) -> str:
    """
    Convert a GLB model to a print-ready 3MF format using PyMeshLab.

    This version applies mesh repair operations to fix common issues that
    cause problems with 3D printing, such as holes, non-manifold geometry,
    and floating debris.

    Repair steps applied:
    ─────────────────────────────────────────────────────────────────────────
    1. Remove duplicate vertices
       - Filter: meshing_remove_duplicate_vertices
       - Merges vertices closer than 0.01% of bounding box diagonal

    2. Remove duplicate faces
       - Filter: meshing_remove_duplicate_faces

    3. Remove zero-area faces
       - Filter: meshing_remove_null_faces

    4. Remove unreferenced vertices
       - Filter: meshing_remove_unreferenced_vertices

    5. Remove small disconnected components
       - Filter: meshing_remove_connected_component_by_diameter
       - Removes floating pieces smaller than 5% of main model diameter

    6. Repair non-manifold edges
       - Filter: meshing_repair_non_manifold_edges
       - Fixes edges shared by more than 2 faces (required for watertight mesh)

    7. Repair non-manifold vertices
       - Filter: meshing_repair_non_manifold_vertices
       - Splits vertices shared by non-adjacent face fans

    8. Close small holes
       - Filter: meshing_close_holes
       - Closes holes with up to 100 edges
       - Preserves vertex colors on filled regions
    ─────────────────────────────────────────────────────────────────────────

    Returns path to the 3MF file.
    Raises an exception if conversion fails or color data is lost.
    """
    if output_path is None:
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, "model_printable.3mf")

    print(f"Converting {input_path} to print-ready 3MF...")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Use trimesh to convert texture to vertex colors
    # PyMeshLab can't access embedded textures in GLB files, so we use trimesh
    # which properly handles embedded textures via to_color()
    # ─────────────────────────────────────────────────────────────────────────
    print("  Loading GLB with trimesh for color extraction...")
    tm_mesh = trimesh.load(input_path, force='mesh')
    print(f"  Loaded mesh: {len(tm_mesh.vertices)} vertices, {len(tm_mesh.faces)} faces")

    # Convert texture to vertex colors using trimesh
    vertex_colors = None
    if hasattr(tm_mesh.visual, 'vertex_colors') and tm_mesh.visual.vertex_colors is not None:
        vc = tm_mesh.visual.vertex_colors
        if len(vc) == len(tm_mesh.vertices) and not np.all(vc == vc[0]):
            vertex_colors = vc
            print(f"  Found existing vertex colors: {len(vertex_colors)} colors")

    if vertex_colors is None and hasattr(tm_mesh.visual, 'to_color'):
        print("  Converting texture to vertex colors...")
        try:
            color_visual = tm_mesh.visual.to_color()
            if hasattr(color_visual, 'vertex_colors') and color_visual.vertex_colors is not None:
                vertex_colors = color_visual.vertex_colors
                tm_mesh.visual = color_visual  # Apply the vertex colors to mesh
                print(f"  Converted to vertex colors: {len(vertex_colors)} colors")
        except Exception as e:
            print(f"  Warning: Could not convert texture to vertex colors: {e}")

    if vertex_colors is None:
        raise RuntimeError(
            "GLB file has no vertex colors and texture conversion failed. "
            "Check that O-Voxel exported colors correctly."
        )

    # Save as PLY with vertex colors for PyMeshLab
    # PLY format preserves vertex colors and PyMeshLab can read it
    temp_ply = os.path.join(os.path.dirname(input_path), "temp_colored.ply")
    tm_mesh.export(temp_ply, file_type='ply')
    print(f"  Saved temporary PLY with vertex colors")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Load into PyMeshLab for mesh repairs
    # ─────────────────────────────────────────────────────────────────────────
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(temp_ply)

    mesh = ms.current_mesh()
    print(f"  PyMeshLab loaded: {mesh.vertex_number()} vertices, {mesh.face_number()} faces")
    print(f"  Has vertex colors: {mesh.has_vertex_color()}")

    if not mesh.has_vertex_color():
        raise RuntimeError(
            "Vertex colors were not preserved in PLY transfer. "
            "This is unexpected - please report this issue."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Mesh repair operations for 3D printing
    # ─────────────────────────────────────────────────────────────────────────
    print("  Applying mesh repairs for 3D printing...")

    initial_vertices = ms.current_mesh().vertex_number()
    initial_faces = ms.current_mesh().face_number()

    # 1. Remove duplicate vertices (within 0.01% of bounding box diagonal)
    print("    - Removing duplicate vertices...")
    ms.meshing_remove_duplicate_vertices()

    # 2. Remove duplicate faces
    print("    - Removing duplicate faces...")
    ms.meshing_remove_duplicate_faces()

    # 3. Remove zero-area (null/degenerate) faces
    print("    - Removing zero-area faces...")
    ms.meshing_remove_null_faces()

    # 4. Remove unreferenced vertices (orphans)
    print("    - Removing unreferenced vertices...")
    ms.meshing_remove_unreferenced_vertices()

    # 5. Remove small floating components (debris)
    #    Removes components smaller than 5% of the largest component's diameter
    print("    - Removing small disconnected components (<5% diameter)...")
    try:
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pymeshlab.PercentageValue(5.0)
        )
    except Exception as e:
        print(f"      Warning: Could not remove small components: {e}")

    # 6. Repair non-manifold edges (edges shared by more than 2 faces)
    print("    - Repairing non-manifold edges...")
    try:
        ms.meshing_repair_non_manifold_edges()
    except Exception as e:
        print(f"      Warning: Could not repair non-manifold edges: {e}")

    # 7. Repair non-manifold vertices (vertices with non-adjacent face fans)
    print("    - Repairing non-manifold vertices...")
    try:
        ms.meshing_repair_non_manifold_vertices()
    except Exception as e:
        print(f"      Warning: Could not repair non-manifold vertices: {e}")

    # 8. Close small holes (up to 100 edges)
    print("    - Closing small holes (up to 100 edges)...")
    try:
        ms.meshing_close_holes(maxholesize=100)
    except Exception as e:
        print(f"      Warning: Could not close holes: {e}")

    # Report changes
    final_vertices = ms.current_mesh().vertex_number()
    final_faces = ms.current_mesh().face_number()
    print(f"  Mesh repair complete:")
    print(f"    Vertices: {initial_vertices} -> {final_vertices} ({final_vertices - initial_vertices:+d})")
    print(f"    Faces: {initial_faces} -> {final_faces} ({final_faces - initial_faces:+d})")

    # Verify colors still exist after repairs
    mesh = ms.current_mesh()
    if not mesh.has_vertex_color():
        raise RuntimeError(
            "Vertex colors were lost during mesh repair operations. "
            "This is unexpected - please report this issue."
        )

    # Extract data directly from PyMeshLab (avoids PLY texture save issues)
    import lib3mf
    from lib3mf import get_wrapper

    # Get the current mesh and compact it to ensure contiguous arrays
    # compact() is a Mesh method, not MeshSet method
    mesh = ms.current_mesh()
    mesh.compact()

    # Get vertices, faces, and colors directly from PyMeshLab
    vertices = mesh.vertex_matrix()  # #V x 3 numpy array
    faces = mesh.face_matrix()  # #F x 3 numpy array
    # vertex_color_matrix returns #V x 4 floats in [0,1] range (RGBA)
    vertex_colors_float = mesh.vertex_color_matrix()
    # Convert to 0-255 range for lib3mf
    vertex_colors = (vertex_colors_float * 255).astype(np.uint8)

    print(f"  Extracted from PyMeshLab: {len(vertices)} vertices, {len(faces)} faces, {len(vertex_colors)} colors")

    # Create lib3mf model using recommended get_wrapper() function
    wrapper = get_wrapper()
    model = wrapper.CreateModel()

    # Create mesh object
    mesh_object = model.AddMeshObject()
    mesh_object.SetName("PrintReadyMesh")

    # Add vertices - lib3mf Position uses .Coordinates array
    for v in vertices:
        pos = lib3mf.Position()
        pos.Coordinates[0] = float(v[0])
        pos.Coordinates[1] = float(v[1])
        pos.Coordinates[2] = float(v[2])
        mesh_object.AddVertex(pos)

    # Add triangles - lib3mf Triangle uses .Indices array
    for f in faces:
        tri = lib3mf.Triangle()
        tri.Indices[0] = int(f[0])
        tri.Indices[1] = int(f[1])
        tri.Indices[2] = int(f[2])
        mesh_object.AddTriangle(tri)

    # Create color group for vertex colors
    color_group = model.AddColorGroup()

    # Build a map of unique colors to color IDs
    color_to_id = {}

    def get_color_id(rgba):
        key = (int(rgba[0]), int(rgba[1]), int(rgba[2]))
        if key not in color_to_id:
            # lib3mf Color uses .Red, .Green, .Blue, .Alpha fields
            color = lib3mf.Color()
            color.Red = int(rgba[0])
            color.Green = int(rgba[1])
            color.Blue = int(rgba[2])
            color.Alpha = int(rgba[3]) if len(rgba) > 3 else 255
            color_id = color_group.AddColor(color)
            color_to_id[key] = color_id
        return color_to_id[key]

    # Set triangle properties with per-vertex colors
    print("  Assigning vertex colors to triangles...")
    resource_id = color_group.GetResourceID()

    for i, face in enumerate(faces):
        c0 = vertex_colors[face[0]]
        c1 = vertex_colors[face[1]]
        c2 = vertex_colors[face[2]]

        id0 = get_color_id(c0)
        id1 = get_color_id(c1)
        id2 = get_color_id(c2)

        # Create triangle properties - uses .ResourceID and .PropertyIDs array
        props = lib3mf.TriangleProperties()
        props.ResourceID = resource_id
        props.PropertyIDs[0] = id0
        props.PropertyIDs[1] = id1
        props.PropertyIDs[2] = id2
        mesh_object.SetTriangleProperties(i, props)

    print(f"  Created {len(color_to_id)} unique colors")

    # Set object-level property
    mesh_object.SetObjectLevelProperty(resource_id, 1)

    # Add mesh to build items
    model.AddBuildItem(mesh_object, wrapper.GetIdentityTransform())

    # Write to file
    writer = model.QueryWriter("3mf")
    writer.WriteToFile(output_path)

    # Clean up temporary PLY file
    try:
        os.remove(temp_ply)
    except Exception:
        pass

    print(f"  Saved print-ready 3MF with vertex colors: {output_path}")

    return output_path


def convert_to_glb_viewer(input_path: str, output_path: str = None) -> str:
    """Convert mesh to GLB format for web viewing (with optional transforms)."""
    if output_path is None:
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, "model_viewer.glb")

    mesh = trimesh.load(input_path)

    # Apply any necessary transforms for viewing
    # (Trellis 2 output should already be correctly oriented)

    if isinstance(mesh, trimesh.Scene):
        mesh.export(output_path, file_type="glb")
    else:
        scene = trimesh.Scene([mesh])
        scene.export(output_path, file_type="glb")

    return output_path


# ---------------------------------------------------------------------------
# Pipeline State (per-session via gr.State)
# ---------------------------------------------------------------------------

class PipelineState:
    """
    Holds data between pipeline steps for a single user session.
    Each browser session gets its own instance via gr.State.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.prompt = ""
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.reference_images = []
        self.generated_images = []
        self.selected_image = None
        self.selected_index = None
        self.output_paths = {}


# ---------------------------------------------------------------------------
# Gradio Step Functions (take and return state for session isolation)
# ---------------------------------------------------------------------------

def step1_generate(
    prompt: str,
    system_prompt: str,
    ref_img_1: Image.Image | None,
    ref_img_2: Image.Image | None,
    ref_img_3: Image.Image | None,
    ref_img_4: Image.Image | None,
    ref_img_5: Image.Image | None,
    state: PipelineState
):
    """Generate images from prompt."""
    if not prompt.strip():
        raise gr.Error("Please enter a description of what you want to create.")

    if not TRELLIS_AVAILABLE:
        raise gr.Error(
            "Trellis 2 is not available in this installation. "
            "Please run with full Docker setup."
        )

    state.reset()
    state.prompt = prompt
    state.system_prompt = system_prompt

    # Collect reference images (filter out None)
    reference_images = [img for img in [ref_img_1, ref_img_2, ref_img_3, ref_img_4, ref_img_5] if img is not None]
    state.reference_images = reference_images

    gr.Info("Generating images... This may take 30-60 seconds.")
    state.generated_images = generate_images(
        prompt=prompt,
        system_prompt=system_prompt,
        reference_images=reference_images,
        num_images=NUM_VARIANTS
    )

    return state.generated_images, state


def step2_select(evt: gr.SelectData, state: PipelineState):
    """Handle image selection from gallery."""
    state.selected_index = evt.index
    state.selected_image = state.generated_images[evt.index]

    return state.selected_image, gr.update(visible=True), state


def step3_reconstruct(state: PipelineState):
    """Run 3D reconstruction with Trellis 2."""
    if state.selected_image is None:
        raise gr.Error("Please select an image first.")

    if not TRELLIS_AVAILABLE:
        raise gr.Error(
            "Trellis 2 is not available. "
            "Please run this app in the Docker container with GPU support."
        )

    gr.Info("Reconstructing 3D Object with Trellis 2... This may take 30-90 seconds.")

    try:
        state.output_paths = reconstruct_3d(image=state.selected_image)

        # Primary output is GLB with full colors and materials
        if "glb" not in state.output_paths:
            raise gr.Error("No 3D model output found")

        glb_path = state.output_paths["glb"]

        # Convert GLB to 3MF for 3D printing (standard version)
        gr.Info("Converting to 3MF format for 3D printing...")
        threemf_path = convert_to_3mf(glb_path)
        if threemf_path:
            state.output_paths["3mf"] = threemf_path

        # Convert GLB to print-ready 3MF (with mesh repairs)
        gr.Info("Creating print-ready 3MF with mesh repairs...")
        threemf_printable_path = convert_to_3mf_printable(glb_path)
        if threemf_printable_path:
            state.output_paths["3mf_printable"] = threemf_printable_path

        return (
            glb_path,  # Use GLB for preview (Gradio doesn't support 3MF rendering)
            threemf_path,  # 3MF download
            threemf_printable_path,  # Print-ready 3MF
            gr.update(visible=True),  # Result group
            state,
        )

    except Exception as e:
        raise gr.Error(f"3D reconstruction failed: {str(e)}")


def step5_download_stl(state: PipelineState):
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
        # Per-session state - each user gets their own PipelineState instance
        state = gr.State(PipelineState)

        gr.Markdown(
            """
            # Text to 3D Printer

            Transform your ideas into 3D printable models in 3 simple steps:
            1. **Describe** what you want to create
            2. **Select** the best generated image
            3. **Download** your 3D printable files (GLB, 3MF)

            *Powered by Microsoft Trellis 2 - generates 3D models with full PBR materials and colors*
            """
        )

        # Step 1: Text Input and Configuration
        with gr.Group():
            gr.Markdown("### Step 1: Describe Your Object")

            with gr.Row():
                prompt_input = gr.Textbox(
                    label="Describe your object",
                    placeholder="e.g., a ceramic coffee mug with geometric patterns",
                    lines=2,
                    scale=4,
                )
                generate_btn = gr.Button("Generate Images", variant="primary", scale=1)

            # System prompt (collapsible)
            with gr.Accordion("Advanced: Style Prompt", open=False):
                system_prompt_input = gr.Textbox(
                    label="Style/System Prompt",
                    value=DEFAULT_SYSTEM_PROMPT,
                    lines=3,
                    info="Additional instructions for the image style. Modify this to change how images are generated.",
                )

            # Reference images (collapsible)
            with gr.Accordion("Optional: Reference Images (up to 5)", open=False):
                gr.Markdown("*Upload reference images to guide the generation. Nano Banana can blend styles and elements from these.*")
                with gr.Row():
                    ref_img_1 = gr.Image(label="Reference 1", type="pil", height=150)
                    ref_img_2 = gr.Image(label="Reference 2", type="pil", height=150)
                    ref_img_3 = gr.Image(label="Reference 3", type="pil", height=150)
                with gr.Row():
                    ref_img_4 = gr.Image(label="Reference 4", type="pil", height=150)
                    ref_img_5 = gr.Image(label="Reference 5", type="pil", height=150)

            # Show availability status
            status = "Trellis 2 available" if TRELLIS_AVAILABLE else "Trellis 2 not available"
            gr.Markdown(f"*Model status: {status}*")

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

        # Step 2b: Confirm Selection
        with gr.Group(visible=False) as confirm_group:
            gr.Markdown("### Confirm Selection")
            selected_image = gr.Image(label="Selected Image", type="pil")
            confirm_btn = gr.Button("Generate 3D Model", variant="primary", size="lg")

        # Step 3: 3D Preview & Download
        with gr.Group(visible=False) as result_group:
            gr.Markdown("### Step 3: Your 3D Model")

            # 3D model preview (using GLB since Gradio doesn't support 3MF rendering)
            threemf_viewer = gr.Model3D(
                label="3D Model Preview",
                clear_color=[0.15, 0.15, 0.15, 1.0],
                height=500,
            )

            gr.Markdown("### Download")

            threemf_download = gr.File(label="Download 3MF (Original)")
            threemf_printable_download = gr.File(label="Download 3MF (Print-Ready, Repaired)")

            gr.Markdown(
                """
                **3MF Format:** 3D printing format with vertex colors. Compatible with Bambu Studio, PrusaSlicer, Cura, and most slicers.

                - **Original:** Direct conversion with colors, no mesh modifications
                - **Print-Ready:** Includes mesh repairs (hole filling, manifold fixes, debris removal)

                *Note: Preview shows GLB render. Download the 3MF files for printing.*
                """
            )

        # Wire up events - state is passed as input AND output for session isolation
        generate_btn.click(
            fn=step1_generate,
            inputs=[prompt_input, system_prompt_input, ref_img_1, ref_img_2, ref_img_3, ref_img_4, ref_img_5, state],
            outputs=[gallery, state],
        )

        gallery.select(
            fn=step2_select,
            inputs=[state],
            outputs=[selected_image, confirm_group, state],
        )

        confirm_btn.click(
            fn=step3_reconstruct,
            inputs=[state],
            outputs=[threemf_viewer, threemf_download, threemf_printable_download, result_group, state],
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
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("No GPU detected by PyTorch")
    except Exception as e:
        print(f"Error checking GPU: {e}")

    # Check for required environment variables
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Warning: REPLICATE_API_TOKEN not set. Image generation will fail.")

    # Print model availability
    print("\n--- Model Availability ---")
    print(f"Trellis 2: {'Available' if TRELLIS_AVAILABLE else 'Not available'}")
    print(f"Model path: {TRELLIS_MODEL_PATH}")
    print(f"Image Model: {REPLICATE_MODEL}")
    print("--------------------------\n")

    # Create and launch the app
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 8080)),
        share=False,
    )
