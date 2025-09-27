#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dependency installation script for QA-ViT + PNP-VQA demo
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def install_dependencies():
    """Install all required dependencies"""
    print("="*60)
    print("QA-ViT + PNP-VQA Dependency Installation")
    print("="*60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 8):
        print("Warning: Python 3.8+ is recommended")
    
    # Update pip
    run_command("python3 -m pip install --upgrade pip", "Upgrading pip")
    
    # Core ML dependencies
    dependencies = [
        "torch",
        "torchvision", 
        "transformers",
        "numpy",
        "pillow",
        "matplotlib",
        "opencv-python",
        "omegaconf",  # Required by LAVIS
        "accelerate",  # Optional but recommended
    ]
    
    print(f"\nInstalling {len(dependencies)} core dependencies...")
    
    failed_installs = []
    for dep in dependencies:
        success = run_command(f"python3 -m pip install {dep}", f"Installing {dep}")
        if not success:
            failed_installs.append(dep)
    
    # Summary
    print("\n" + "="*60)
    if not failed_installs:
        print("✓ All dependencies installed successfully!")
        print("\nNext steps:")
        print("1. Run 'python3 test_setup.py' to verify installation")
        print("2. Run 'python3 run_demo.py' to start the demo")
    else:
        print(f"✗ Failed to install: {', '.join(failed_installs)}")
        print("\nTry installing manually:")
        for dep in failed_installs:
            print(f"  pip3 install {dep}")
    print("="*60)

if __name__ == "__main__":
    install_dependencies()