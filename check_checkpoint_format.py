#!/usr/bin/env python3
import torch

# Load checkpoint
ckpt = torch.load('checkpoints/kaggle/final_converted.pth', map_location='cpu', weights_only=False)

print("Checkpoint structure:")
print(f"Type: {type(ckpt)}")

if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    print("\nWrapped checkpoint - keys:", list(ckpt.keys()))
    state_dict = ckpt['model_state_dict']
else:
    print("\nDirect state_dict")
    state_dict = ckpt

print(f"\nTotal parameters: {len(state_dict)}")
print("\nFirst 20 keys:")
for i, key in enumerate(list(state_dict.keys())[:20]):
    print(f"  {i+1}. {key}")

print("\nLast 5 keys:")
for key in list(state_dict.keys())[-5:]:
    print(f"  - {key}")

print("\nChecking for decoder keys:")
print(f"  Has 'head.': {any(k.startswith('head.') for k in state_dict.keys())}")
print(f"  Has 'patch_recon.': {any(k.startswith('patch_recon.') for k in state_dict.keys())}")

