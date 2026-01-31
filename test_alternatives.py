"""
Alternative solution: Maybe the issue is that the model isn't actually helping
Let's test if we should just enhance the original image instead of using the model
"""
import cv2
import numpy as np
from PIL import Image
import os

print("=" * 70)
print("TESTING ALTERNATIVE: Skip Model, Just Enhance Image")
print("=" * 70)

# Load test image
test_image = "data/raw/test/test_0009.jpg"
img = cv2.imread(test_image)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

print(f"\nOriginal image: {w}×{h}")

def calculate_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def enhance_image(img):
    """
    Simple enhancement without ViT model
    - Denoise
    - Sharpen
    - Contrast
    """
    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Sharpen
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # CLAHE for contrast
    lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    return enhanced

def enhance_simple(img):
    """Simpler enhancement"""
    # Unsharp mask
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharpened = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

# Test different enhancement methods
print("\nTesting enhancement methods...")

# Method 1: Original (no processing)
sharp_original = calculate_sharpness(img_rgb)

# Method 2: Simple enhancement
enhanced_simple = enhance_simple(img_rgb)
sharp_simple = calculate_sharpness(enhanced_simple)

# Method 3: Full enhancement
enhanced_full = enhance_image(img_rgb)
sharp_full = calculate_sharpness(enhanced_full)

print(f"\nSharpness comparison:")
print(f"  Original:          {sharp_original:.2f} (100%)")
print(f"  Simple enhance:    {sharp_simple:.2f} ({sharp_simple/sharp_original*100:.1f}%)")
print(f"  Full enhance:      {sharp_full:.2f} ({sharp_full/sharp_original*100:.1f}%)")

# Compare with ViT model output
if os.path.exists("output/diagnostics/2_enhanced_auto.png"):
    vit_output = cv2.imread("output/diagnostics/2_enhanced_auto.png")
    vit_output_rgb = cv2.cvtColor(vit_output, cv2.COLOR_BGR2RGB)
    sharp_vit = calculate_sharpness(vit_output_rgb)
    print(f"  ViT model:         {sharp_vit:.2f} ({sharp_vit/sharp_original*100:.1f}%)")

    if sharp_simple > sharp_vit:
        print(f"\n⚠️  WARNING: Simple enhancement is better than ViT model!")
        print(f"   Simple: {sharp_simple/sharp_original*100:.1f}% vs ViT: {sharp_vit/sharp_original*100:.1f}%")

# Save outputs
output_dir = "output/alternatives"
os.makedirs(output_dir, exist_ok=True)

cv2.imwrite(f"{output_dir}/1_original.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{output_dir}/2_simple_enhance.png", cv2.cvtColor(enhanced_simple, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{output_dir}/3_full_enhance.png", cv2.cvtColor(enhanced_full, cv2.COLOR_RGB2BGR))

print(f"\nSaved to: {output_dir}/")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

if sharp_simple > sharp_original * 1.1:
    print("\n✅ SOLUTION: Use simple sharpening instead of ViT model!")
    print("\n   The ViT model might be:")
    print("   1. Not properly trained")
    print("   2. Trained on different type of manuscripts")
    print("   3. Designed for heavily degraded images (not clean scans)")
    print("\n   Recommendation:")
    print("   • Skip ViT restoration for clean manuscripts")
    print("   • Use simple sharpening/enhancement instead")
    print("   • Or train model on your specific manuscript type")
elif sharp_simple < sharp_original * 0.9:
    print("\n⚠️  Enhancement is making images WORSE")
    print("   Your images might already be high quality")
    print("   Recommendation: Don't process, use original images")
else:
    print("\n📊 Results are mixed")
    print("   Consider using ViT model only for degraded images")

