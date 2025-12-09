# Color 3D Printing Instructions

This PLY file contains a 3D model with embedded vertex colors.

---

## For Full-Color Printing Services

**Recommended services:** Shapeways, Sculpteo, Xometry, i.materialise

### Steps:

1. **Convert PLY to VRML format** (most services require this):
   - Download [MeshLab](https://www.meshlab.net/) (free)
   - File → Import Mesh → select the `.ply` file
   - File → Export Mesh As → save as `.wrl` (VRML 2.0)

2. **Upload to printing service**
   - Select "Full Color Sandstone" or "Multicolor" material
   - Scale the model to desired size (it may import very small)

3. **Order print**

---

## For In-House FDM Printers

Standard FDM printers cannot print vertex colors directly. Options:

- **Multi-filament printer** (Bambu AMS, Prusa MMU): Manually assign color regions in slicer
- **Single color printer**: Print in white/gray, then hand-paint using the on-screen preview as reference

---

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Colors not visible in MeshLab | Render → Color → Per Vertex |
| Service says "no color data" | Export as VRML (.wrl) instead of OBJ |
| Model too small | Scale up in slicer or MeshLab |
| Colors look faded | Normal for vertex colors; original preview has richer color |