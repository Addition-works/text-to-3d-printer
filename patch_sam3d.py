#!/usr/bin/env python3
"""
Patch SAM 3D Objects to fix mask boolean indexing issue.

The _compute_scale_and_shift function uses pointmap[mask] indexing which fails
when mask tensor is uint8 (0/255) instead of boolean. This patch ensures proper
boolean conversion before indexing, and handles empty mask cases gracefully.
"""
import os

TRANSFORMS_FILE = "/app/sam-3d-objects/sam3d_objects/data/dataset/tdfy/img_and_mask_transforms.py"
PREPROCESSOR_FILE = "/app/sam-3d-objects/sam3d_objects/data/dataset/tdfy/preprocessor.py"

def patch_transforms():
    """Patch the _compute_scale_and_shift function"""
    if not os.path.exists(TRANSFORMS_FILE):
        print(f"Warning: {TRANSFORMS_FILE} not found")
        return False
    
    with open(TRANSFORMS_FILE, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if '# PATCHED: Ensure mask is boolean' in content:
        print("Transforms file already patched, skipping")
        return True
    
    # Find the _compute_scale_and_shift method and add mask conversion
    lines = content.split('\n')
    new_lines = []
    found = False
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Insert boolean conversion right after function definition
        if 'def _compute_scale_and_shift(self, pointmap, mask):' in line:
            found = True
            # Get the indentation of the next non-empty line
            for j in range(i+1, min(i+10, len(lines))):
                if lines[j].strip() and not lines[j].strip().startswith('#') and not lines[j].strip().startswith('"""'):
                    indent = len(lines[j]) - len(lines[j].lstrip())
                    break
            else:
                indent = 8
            
            indent_str = ' ' * indent
            patch_lines = [
                f'{indent_str}# PATCHED: Ensure mask is boolean for proper indexing',
                f'{indent_str}if mask.dtype != torch.bool:',
                f'{indent_str}    print(f"[MASK PATCH] Converting mask from {{mask.dtype}} to bool")',
                f'{indent_str}    mask = mask > 0',
                f'{indent_str}mask_sum = mask.sum().item()',
                f'{indent_str}print(f"[MASK PATCH] pointmap: {{pointmap.shape}}, mask: {{mask.shape}}, True pixels: {{mask_sum}}")',
                f'{indent_str}if mask_sum == 0:',
                f'{indent_str}    print("[MASK PATCH] WARNING: Mask has zero foreground pixels!")',
                f'{indent_str}    # Return default values to avoid crash',
                f'{indent_str}    return pointmap, torch.tensor(1.0, device=pointmap.device), torch.tensor(0.0, device=pointmap.device)',
            ]
            new_lines.extend(patch_lines)
    
    if found:
        new_content = '\n'.join(new_lines)
        with open(TRANSFORMS_FILE, 'w') as f:
            f.write(new_content)
        print(f"✓ Patched _compute_scale_and_shift in {TRANSFORMS_FILE}")
        return True
    else:
        print(f"Warning: Could not find _compute_scale_and_shift function")
        if '_compute_scale_and_shift' in content:
            print("  (function name found but signature may differ)")
        return False


def patch_preprocessor():
    """Add debugging to the preprocessor to trace mask handling"""
    if not os.path.exists(PREPROCESSOR_FILE):
        print(f"Warning: {PREPROCESSOR_FILE} not found")
        return False
    
    with open(PREPROCESSOR_FILE, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if '# PATCHED: Debug mask' in content:
        print("Preprocessor file already patched, skipping")
        return True
    
    # Find _process_image_mask_pointmap_mess and add debug at start
    lines = content.split('\n')
    new_lines = []
    found = False
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        if 'def _process_image_mask_pointmap_mess(self' in line:
            found = True
            # Find the parameters and add debug after
            for j in range(i+1, min(i+20, len(lines))):
                if lines[j].strip().startswith('):') or (lines[j].strip() and not lines[j].strip().startswith('#') and not lines[j].strip().endswith(',') and ')' in lines[j]):
                    # End of function signature
                    indent = 8
                    patch_lines = [
                        f'{" " * indent}# PATCHED: Debug mask at preprocessing entry',
                        f'{" " * indent}import numpy as np',
                        f'{" " * indent}if mask is not None:',
                        f'{" " * indent}    if hasattr(mask, "shape"):',
                        f'{" " * indent}        _mask_info = f"shape={{mask.shape}}, dtype={{mask.dtype}}"',
                        f'{" " * indent}        if hasattr(mask, "sum"):',
                        f'{" " * indent}            _mask_info += f", sum={{mask.sum() if hasattr(mask.sum(), \"__call__\") else mask.sum}}"',
                        f'{" " * indent}        print(f"[PREPROCESS DEBUG] mask: {{_mask_info}}")',
                    ]
                    # Insert after the closing of function signature
                    insert_pos = len(new_lines)
                    for k in range(j - i):
                        new_lines.append(lines[i + 1 + k])
                    new_lines.extend(patch_lines)
                    # Skip the lines we already added
                    lines = lines[:i+1] + lines[j+1:]
                    break
            break
    
    if found:
        new_content = '\n'.join(new_lines)
        with open(PREPROCESSOR_FILE, 'w') as f:
            f.write(new_content)
        print(f"✓ Patched _process_image_mask_pointmap_mess in {PREPROCESSOR_FILE}")
        return True
    else:
        print(f"Warning: Could not find _process_image_mask_pointmap_mess function")
        return False


def main():
    print("Patching SAM 3D Objects for mask handling...")
    t1 = patch_transforms()
    # Skip preprocessor patch for now - it's complex
    # t2 = patch_preprocessor()
    if t1:
        print("✓ Patching complete")
    else:
        print("⚠ Some patches may have failed")


if __name__ == '__main__':
    main()