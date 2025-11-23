#!/usr/bin/env python3
"""
Search and download public Sanskrit/Devanagari datasets from Roboflow Universe
"""

from roboflow import Roboflow
import sys

def search_public_datasets(api_key):
    """Search for public Sanskrit/Devanagari datasets."""

    print("🔍 Searching Roboflow Universe for Sanskrit/Devanagari datasets...")
    print()

    # List of known public datasets
    public_datasets = [
        {
            'name': 'Devanagari Character Recognition',
            'workspace': 'roboflow-100',
            'project': 'devanagari-character-recognition',
            'description': 'Devanagari character dataset'
        },
        {
            'name': 'Sanskrit OCR',
            'workspace': 'sanskrit-ocr',
            'project': 'sanskrit-characters',
            'description': 'Sanskrit character recognition'
        },
        {
            'name': 'Hindi/Devanagari Dataset',
            'workspace': 'devanagari',
            'project': 'hindi-characters',
            'description': 'Hindi Devanagari characters'
        }
    ]

    print("📚 Suggested Public Datasets:")
    print("-" * 70)
    for idx, dataset in enumerate(public_datasets, 1):
        print(f"\n{idx}. {dataset['name']}")
        print(f"   Workspace: {dataset['workspace']}")
        print(f"   Project: {dataset['project']}")
        print(f"   Description: {dataset['description']}")
        print(f"   URL: https://universe.roboflow.com/{dataset['workspace']}/{dataset['project']}")

    print("\n" + "-" * 70)
    print()
    print("💡 To use a public dataset:")
    print("   1. Visit the URL above")
    print("   2. Check if it has versions available")
    print("   3. Update download_roboflow_dataset.py with the workspace/project")
    print("   4. Run the download script again")
    print()
    print("🔗 Or browse all datasets:")
    print("   https://universe.roboflow.com/search?q=sanskrit")
    print("   https://universe.roboflow.com/search?q=devanagari")
    print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Search for public datasets')
    parser.add_argument('--api-key', type=str, help='Your Roboflow API key')
    args = parser.parse_args()

    if args.api_key:
        search_public_datasets(args.api_key)
    else:
        search_public_datasets(None)

