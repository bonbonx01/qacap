#!/usr/bin/env python3
"""
Quick Demo Runner for QA-ViT Heat Map Generation

This script provides a simple way to test the QA-ViT + PNP-VQA model
with predefined examples or custom inputs.

Usage:
    python run_demo.py  # Run with default example
    python run_demo.py --custom --image path/to/image.jpg --question "Your question"
"""

import argparse
import os
import sys
import torch
from pathlib import Path
import urllib.request
from PIL import Image
import numpy as np

# Add current directory to path
sys.path.append('/workspace')

from demo_qavit_heatmap import QAViTHeatMapDemo


def download_sample_image():
    """Download a sample image for testing"""
    sample_url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/merlion.png"
    sample_path = "/workspace/sample_image.jpg"
    
    if not os.path.exists(sample_path):
        print("Downloading sample image...")
        try:
            urllib.request.urlretrieve(sample_url, sample_path)
            print(f"Sample image saved to: {sample_path}")
        except Exception as e:
            print(f"Could not download sample image: {e}")
            # Create a simple test image instead
            create_test_image(sample_path)
    
    return sample_path

def create_test_image(path):
    """Create a simple test image"""
    print("Creating a simple test image...")
    # Create a simple image with colored rectangles
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Add colored rectangles
    img_array[50:100, 50:150, 0] = 255  # Red rectangle
    img_array[120:170, 80:180, 1] = 255  # Green rectangle
    img_array[100:150, 120:200, 2] = 255  # Blue rectangle
    
    # Add some text-like patterns
    img_array[180:200, 30:200, :] = 128  # Gray bar
    
    img = Image.fromarray(img_array)
    img.save(path)
    print(f"Test image created: {path}")

def run_predefined_examples():
    """Run predefined examples"""
    # Sample image
    image_path = download_sample_image()
    
    # Predefined questions
    questions = [
        "What colors are in the image?",
        "What objects can you see?",
        "Where is the red object located?",
        "Is there any text in the image?"
    ]
    
    print("Running predefined examples...")
    print("="*60)
    
    try:
        # Initialize demo
        demo = QAViTHeatMapDemo()
        
        # Run examples
        for i, question in enumerate(questions, 1):
            print(f"\nExample {i}: {question}")
            print("-" * 40)
            
            try:
                fig, attention_map = demo.run_demo(
                    image_path=image_path,
                    question=question,
                    output_dir="/workspace/demo_outputs",
                    block_num=7
                )
                
                print(f"✓ Example {i} completed successfully")
                
            except Exception as e:
                print(f"✗ Example {i} failed: {e}")
                continue
        
        print("\n" + "="*60)
        print("All examples completed! Check /workspace/demo_outputs/ for results.")
        
    except Exception as e:
        print(f"Failed to initialize demo: {e}")
        return 1
    
    return 0

def run_custom_example(image_path, question):
    """Run custom example with user-provided image and question"""
    print(f"Running custom example...")
    print(f"Image: {image_path}")
    print(f"Question: {question}")
    print("="*60)
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return 1
    
    try:
        # Initialize demo
        demo = QAViTHeatMapDemo()
        
        # Run custom example
        fig, attention_map = demo.run_demo(
            image_path=image_path,
            question=question,
            output_dir="/workspace/demo_outputs",
            block_num=7
        )
        
        print("✓ Custom example completed successfully")
        print("Check /workspace/demo_outputs/ for results.")
        
    except Exception as e:
        print(f"✗ Custom example failed: {e}")
        return 1
    
    return 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="QA-ViT Demo Runner")
    parser.add_argument("--custom", action="store_true",
                       help="Run custom example instead of predefined ones")
    parser.add_argument("--image", type=str,
                       help="Path to custom image (required with --custom)")
    parser.add_argument("--question", type=str,
                       help="Custom question (required with --custom)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Check device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()
    
    # Create output directory
    os.makedirs("/workspace/demo_outputs", exist_ok=True)
    
    if args.custom:
        if not args.image or not args.question:
            print("Error: --image and --question are required with --custom")
            return 1
        return run_custom_example(args.image, args.question)
    else:
        return run_predefined_examples()

if __name__ == "__main__":
    exit(main())