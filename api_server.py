#!/usr/bin/env python3
"""
Simple REST API for manuscript restoration
Usage: python api_server.py
Access: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from models.vit_restorer import ViTRestorer
from PIL import Image
import numpy as np
import io
import base64
from pathlib import Path
import tempfile
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global model variable
model = None
device = None

def load_model(checkpoint_path='checkpoints/best_psnr.pth'):
    """Load the trained model"""
    global model, device

    print(f"Loading model from: {checkpoint_path}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = ViTRestorer(
        img_size=256,
        patch_size=16,
        in_channels=3,
        out_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        use_skip_connections=True
    )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print("✓ Model loaded successfully")
    return model

def preprocess_image(image, target_size=256):
    """Preprocess image for model input"""
    # Resize
    image = image.resize((target_size, target_size), Image.LANCZOS)

    # Convert to numpy array
    img_array = np.array(image).astype(np.float32) / 255.0

    # Convert to tensor
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    return img_tensor

def postprocess_image(tensor):
    """Convert model output to PIL image"""
    # Remove batch dimension and convert to numpy
    img_array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Clip and convert to uint8
    img_array = np.clip(img_array, 0, 1)
    img_array = (img_array * 255).astype(np.uint8)

    # Convert to PIL
    image = Image.fromarray(img_array)

    return image

@app.route('/')
def index():
    """API documentation"""
    return jsonify({
        'name': 'Manuscript Restoration API',
        'version': '1.0.0',
        'description': 'REST API for historic manuscript restoration using Vision Transformer',
        'endpoints': {
            '/': 'This documentation',
            '/health': 'Health check',
            '/restore': 'POST - Restore a manuscript image',
            '/restore_base64': 'POST - Restore using base64 encoded image'
        },
        'usage': {
            'curl': 'curl -X POST -F "image=@manuscript.jpg" http://localhost:5000/restore --output restored.jpg',
            'python': 'See example_api_client.py'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'not set'
    })

@app.route('/restore', methods=['POST'])
def restore():
    """
    Restore a manuscript image

    Input: multipart/form-data with 'image' file
    Output: Restored image file
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read image
        image = Image.open(file.stream).convert('RGB')

        # Preprocess
        input_tensor = preprocess_image(image).to(device)

        # Inference
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Postprocess
        restored_image = postprocess_image(output_tensor)

        # Save to buffer
        img_buffer = io.BytesIO()
        restored_image.save(img_buffer, format='JPEG', quality=95)
        img_buffer.seek(0)

        return send_file(
            img_buffer,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='restored.jpg'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/restore_base64', methods=['POST'])
def restore_base64():
    """
    Restore a manuscript image from base64

    Input JSON: {"image": "base64_encoded_image"}
    Output JSON: {"image": "base64_restored_image"}
    """
    try:
        data = request.get_json()

        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode base64
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Preprocess
        input_tensor = preprocess_image(image).to(device)

        # Inference
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Postprocess
        restored_image = postprocess_image(output_tensor)

        # Encode to base64
        img_buffer = io.BytesIO()
        restored_image.save(img_buffer, format='JPEG', quality=95)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        return jsonify({
            'image': img_base64,
            'format': 'jpeg'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_restore', methods=['POST'])
def batch_restore():
    """
    Restore multiple images

    Input: multipart/form-data with multiple 'images[]' files
    Output: ZIP file with restored images
    """
    if 'images[]' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    files = request.files.getlist('images[]')

    if len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400

    try:
        import zipfile

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Process each image
            for idx, file in enumerate(files):
                # Read image
                image = Image.open(file.stream).convert('RGB')

                # Preprocess
                input_tensor = preprocess_image(image).to(device)

                # Inference
                with torch.no_grad():
                    output_tensor = model(input_tensor)

                # Postprocess
                restored_image = postprocess_image(output_tensor)

                # Save
                output_path = tmpdir_path / f'restored_{idx+1}.jpg'
                restored_image.save(output_path, format='JPEG', quality=95)

            # Create ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in tmpdir_path.glob('*.jpg'):
                    zipf.write(file_path, file_path.name)

            zip_buffer.seek(0)

            return send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True,
                download_name='restored_images.zip'
            )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Manuscript Restoration API Server')
    parser.add_argument('--model', type=str, default='checkpoints/best_psnr.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port number (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')

    args = parser.parse_args()

    # Load model
    load_model(args.model)

    print()
    print("=" * 70)
    print("🚀 Manuscript Restoration API Server")
    print("=" * 70)
    print(f"API URL: http://{args.host}:{args.port}")
    print(f"Documentation: http://{args.host}:{args.port}/")
    print(f"Health check: http://{args.host}:{args.port}/health")
    print()
    print("Example usage:")
    print(f"  curl -X POST -F 'image=@manuscript.jpg' \\")
    print(f"       http://localhost:{args.port}/restore \\")
    print(f"       --output restored.jpg")
    print("=" * 70)
    print()

    # Run server
    app.run(host=args.host, port=args.port, debug=args.debug)

