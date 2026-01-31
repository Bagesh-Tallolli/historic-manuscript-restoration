#!/usr/bin/env python3
"""
Example client for the Manuscript Restoration API
"""

import requests
import base64
from pathlib import Path
import argparse

class ManuscriptRestorationClient:
    """Client for the manuscript restoration API"""

    def __init__(self, api_url='http://localhost:5000'):
        self.api_url = api_url

    def health_check(self):
        """Check API health"""
        response = requests.get(f'{self.api_url}/health')
        return response.json()

    def restore_image(self, image_path, output_path=None):
        """
        Restore a single image

        Args:
            image_path: Path to input image
            output_path: Path to save restored image (optional)

        Returns:
            bytes: Restored image data
        """
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f'{self.api_url}/restore', files=files)

        if response.status_code == 200:
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Restored image saved to: {output_path}")
            return response.content
        else:
            error = response.json().get('error', 'Unknown error')
            raise Exception(f"API error: {error}")

    def restore_image_base64(self, image_path):
        """
        Restore image using base64 encoding

        Args:
            image_path: Path to input image

        Returns:
            str: Base64 encoded restored image
        """
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        # Send request
        response = requests.post(
            f'{self.api_url}/restore_base64',
            json={'image': image_data}
        )

        if response.status_code == 200:
            return response.json()['image']
        else:
            error = response.json().get('error', 'Unknown error')
            raise Exception(f"API error: {error}")

    def restore_batch(self, image_paths, output_zip=None):
        """
        Restore multiple images

        Args:
            image_paths: List of paths to input images
            output_zip: Path to save output ZIP file (optional)

        Returns:
            bytes: ZIP file data
        """
        files = [('images[]', open(path, 'rb')) for path in image_paths]

        try:
            response = requests.post(f'{self.api_url}/batch_restore', files=files)

            if response.status_code == 200:
                if output_zip:
                    with open(output_zip, 'wb') as f:
                        f.write(response.content)
                    print(f"✓ Restored images saved to: {output_zip}")
                return response.content
            else:
                error = response.json().get('error', 'Unknown error')
                raise Exception(f"API error: {error}")
        finally:
            # Close all file handles
            for _, f in files:
                f.close()


def main():
    parser = argparse.ArgumentParser(description='Manuscript Restoration API Client')
    parser.add_argument('--api-url', type=str, default='http://localhost:5000',
                       help='API server URL')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path or folder')
    parser.add_argument('--output', type=str, required=True,
                       help='Output image path or ZIP file')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple images')

    args = parser.parse_args()

    # Create client
    client = ManuscriptRestorationClient(args.api_url)

    # Check API health
    print("Checking API health...")
    health = client.health_check()
    print(f"✓ API Status: {health['status']}")
    print(f"✓ Model loaded: {health['model_loaded']}")
    print()

    # Process images
    if args.batch:
        # Batch processing
        input_path = Path(args.input)
        if not input_path.is_dir():
            print("Error: --batch requires a folder path")
            return

        image_paths = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))

        if not image_paths:
            print(f"No images found in: {input_path}")
            return

        print(f"Processing {len(image_paths)} images...")
        client.restore_batch(image_paths, args.output)
        print("✓ Batch processing complete!")

    else:
        # Single image
        print(f"Processing: {args.input}")
        client.restore_image(args.input, args.output)
        print("✓ Processing complete!")


if __name__ == '__main__':
    main()

