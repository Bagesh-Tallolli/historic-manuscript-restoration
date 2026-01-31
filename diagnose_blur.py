"""
Diagnostic script to identify blur issue
Tests with actual image to see what's happening
"""
import torch
import numpy as np
from PIL import Image
import cv2
import os
from models.vit_restorer import create_vit_restorer
from utils.image_restoration_enhanced import create_enhanced_restorer

print("=" * 70)
print("DIAGNOSING BLUR ISSUE")
print("=" * 70)

# Load model
print("\n1. Loading model...")
checkpoint_path = "checkpoints/latest_model.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
state_dict = checkpoint.get('model_state_dict', checkpoint)
use_simple_head = 'head.weight' in state_dict

model = create_vit_restorer(
    model_size='base',
    img_size=256,
    use_simple_head=use_simple_head,
    use_skip_connections='skip_fusion.weight' in state_dict
)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print(f"✅ Model loaded on {device}")
print(f"   Architecture: {'old (simple head)' if use_simple_head else 'new (patch_recon)'}")
print(f"   Skip connections: {'skip_fusion.weight' in state_dict}")

# Create enhanced restorer
enhanced_restorer = create_enhanced_restorer(model, device=device, patch_size=256, overlap=32)

# Check if there are test images
test_dirs = ['data/raw/test', 'data/raw', 'output', '.']
test_image = None

for dir_path in test_dirs:
    if os.path.exists(dir_path):
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']:
            import glob
            files = glob.glob(os.path.join(dir_path, ext))
            if files:
                test_image = files[0]
                break
    if test_image:
        break

if not test_image:
    print("\n⚠️  No test images found. Creating synthetic test image...")
    # Create test image with patterns
    test_img = np.ones((600, 800, 3), dtype=np.uint8) * 240

    # Add text-like patterns
    for i in range(15):
        y = 50 + i * 35
        cv2.rectangle(test_img, (50, y), (750, y+20), (50, 50, 50), -1)
        for j in range(10):
            x = 60 + j * 70
            cv2.rectangle(test_img, (x, y+5), (x+50, y+15), (180, 180, 180), 1)

    test_image = "test_synthetic.png"
    cv2.imwrite(test_image, cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
    print(f"   Created: {test_image}")
else:
    print(f"\n2. Using test image: {test_image}")

# Load and process image
img_pil = Image.open(test_image).convert("RGB")
img_np = np.array(img_pil)
h, w = img_np.shape[:2]

print(f"\n3. Image info:")
print(f"   Size: {w}×{h}")
print(f"   Will use: {'PATCH-BASED' if (h > 512 or w > 512) else 'SIMPLE'} method")

# Test restoration
print(f"\n4. Testing restoration...")

# Method 1: Direct model call (what might be causing blur)
print("\n   Method 1: Direct model (resize method)...")
img_resized = cv2.resize(img_np, (256, 256))
img_tensor = torch.from_numpy(img_resized).float() / 255.0
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    output_tensor = model(img_tensor)

output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
print(f"   Output range: [{output_np.min():.3f}, {output_np.max():.3f}]")
print(f"   Output mean: {output_np.mean():.3f}")
print(f"   Output std: {output_np.std():.3f}")

# Check if output is in wrong range
if output_np.max() > 2.0 or output_np.min() < -1.0:
    print(f"   ⚠️  WARNING: Output range is unusual!")
    print(f"   This might cause issues when converting back to image")

restored_direct = (output_np * 255).clip(0, 255).astype(np.uint8)
restored_direct = cv2.resize(restored_direct, (w, h))

# Method 2: Enhanced restorer
print("\n   Method 2: Enhanced restorer...")
use_patches = (h > 512 or w > 512)
restored_enhanced = enhanced_restorer.restore_image(
    img_np,
    use_patches=use_patches,
    apply_postprocess=True
)

# Method 3: Enhanced restorer WITHOUT post-processing
print("\n   Method 3: Enhanced restorer (no post-processing)...")
restored_no_post = enhanced_restorer.restore_image(
    img_np,
    use_patches=use_patches,
    apply_postprocess=False
)

# Method 4: Enhanced restorer ALWAYS using patches
print("\n   Method 4: Enhanced restorer (force patches)...")
restored_forced_patches = enhanced_restorer.restore_image(
    img_np,
    use_patches=True,  # Force patch-based even for small images
    apply_postprocess=True
)

# Calculate sharpness for each method
def calculate_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

print(f"\n5. Sharpness comparison:")
sharpness_original = calculate_sharpness(img_np)
sharpness_direct = calculate_sharpness(restored_direct)
sharpness_enhanced = calculate_sharpness(restored_enhanced)
sharpness_no_post = calculate_sharpness(restored_no_post)
sharpness_forced = calculate_sharpness(restored_forced_patches)

print(f"   Original:                {sharpness_original:.2f}")
print(f"   Direct (resize):         {sharpness_direct:.2f} ({sharpness_direct/sharpness_original*100:.1f}%)")
print(f"   Enhanced (auto):         {sharpness_enhanced:.2f} ({sharpness_enhanced/sharpness_original*100:.1f}%)")
print(f"   Enhanced (no post):      {sharpness_no_post:.2f} ({sharpness_no_post/sharpness_original*100:.1f}%)")
print(f"   Enhanced (forced patch): {sharpness_forced:.2f} ({sharpness_forced/sharpness_original*100:.1f}%)")

# Identify best method
best_sharpness = max(sharpness_direct, sharpness_enhanced, sharpness_no_post, sharpness_forced)
best_method = "None"
if best_sharpness == sharpness_direct:
    best_method = "Direct (resize) ⚠️ This shouldn't be best!"
elif best_sharpness == sharpness_enhanced:
    best_method = "Enhanced (auto) ✅"
elif best_sharpness == sharpness_no_post:
    best_method = "Enhanced (no post) ⚠️ Post-processing might be too aggressive"
elif best_sharpness == sharpness_forced:
    best_method = "Enhanced (forced patch) ✅"

print(f"\n   Best method: {best_method}")

# Save outputs for visual comparison
output_dir = "output/diagnostics"
os.makedirs(output_dir, exist_ok=True)

cv2.imwrite(f"{output_dir}/0_original.png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{output_dir}/1_direct_resize.png", cv2.cvtColor(restored_direct, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{output_dir}/2_enhanced_auto.png", cv2.cvtColor(restored_enhanced, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{output_dir}/3_enhanced_no_post.png", cv2.cvtColor(restored_no_post, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{output_dir}/4_enhanced_forced_patch.png", cv2.cvtColor(restored_forced_patches, cv2.COLOR_RGB2BGR))

print(f"\n6. Saved comparison images to: {output_dir}/")
print(f"   View these images to see the difference!")

# Diagnosis
print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

if sharpness_enhanced < sharpness_original * 0.8:
    print("\n❌ ISSUE FOUND: Enhanced restoration is making images LESS sharp!")
    print("\n   Possible causes:")
    print("   1. Model output is not in expected range")
    print("   2. Post-processing is too aggressive")
    print("   3. Image size threshold (512px) might not be optimal")
    print("   4. Resize interpolation method might be wrong")

    if sharpness_no_post > sharpness_enhanced:
        print("\n   🔍 Post-processing is reducing sharpness!")
        print("   Solution: Adjust post-processing parameters")

    if sharpness_forced > sharpness_enhanced:
        print("\n   🔍 Using patches even for small images improves quality!")
        print("   Solution: Always use patch-based method, or reduce threshold")

    print("\n   Recommended fixes:")
    if sharpness_forced > sharpness_enhanced:
        print("   ✅ ALWAYS use patch-based processing (set use_patches=True)")
    if sharpness_no_post > sharpness_enhanced:
        print("   ✅ Disable or reduce post-processing sharpening")
    print("   ✅ Check model output normalization")

elif sharpness_enhanced > sharpness_original * 0.95:
    print("\n✅ Enhanced restoration is working correctly!")
    print(f"   Sharpness retained: {sharpness_enhanced/sharpness_original*100:.1f}%")
else:
    print(f"\n⚠️  Enhanced restoration has minor quality loss")
    print(f"   Sharpness retained: {sharpness_enhanced/sharpness_original*100:.1f}%")
    print("   This is acceptable but could be improved")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)

print("\n1. View the saved images in: output/diagnostics/")
print("2. Compare visually to see which method looks best")
print("3. Check the sharpness scores above")

if sharpness_forced > sharpness_enhanced:
    print("\n💡 SOLUTION FOUND:")
    print("   Modify restore_manuscript() to ALWAYS use patches:")
    print("   Change: use_patches = (original_h > 512 or original_w > 512)")
    print("   To:     use_patches = True")

