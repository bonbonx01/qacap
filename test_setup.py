#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify that all dependencies and components are working correctly.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import PIL
        print(f"✓ PIL/Pillow imported successfully")
    except ImportError as e:
        print(f"✗ PIL/Pillow import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    return True

def test_lavis_imports():
    """Test LAVIS components"""
    print("\nTesting LAVIS components...")
    
    try:
        # Add lavis to path
        sys.path.append('/workspace')
        
        from lavis.common.registry import registry
        print("✓ LAVIS registry imported successfully")
        
        from lavis.models.base_model import BaseModel
        print("✓ LAVIS BaseModel imported successfully")
        
        # Test QA-ViT components
        from lavis.models.qacap_models.clip_qavit import InstructCLIPVisionModel
        print("✓ QA-ViT InstructCLIPVisionModel imported successfully")
        
        from lavis.models.qacap_models.qavit_encoder import CLIPVisionTower
        print("✓ CLIPVisionTower imported successfully")
        
        from lavis.models.qacap_models.qacap_vqa import PNPVQAWithQAViT
        print("✓ PNPVQAWithQAViT imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ LAVIS import failed: {e}")
        return False

def test_model_initialization():
    """Test basic model initialization"""
    print("\nTesting model initialization...")
    
    try:
        import torch
        from transformers import BertModel, BertTokenizer
        
        # Test text encoder initialization
        text_encoder = BertModel.from_pretrained('bert-base-uncased')
        text_encoder.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print("✓ Text encoder initialized successfully")
        
        # Test tokenization
        sample_text = ["What is in the image?"]
        tokens = text_encoder.tokenizer(sample_text, return_tensors='pt', padding=True)
        print("✓ Text tokenization working")
        
        # Test encoding
        with torch.no_grad():
            outputs = text_encoder(**tokens)
            print(f"✓ Text encoding working, output shape: {outputs.last_hidden_state.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False

def test_image_processing():
    """Test image processing pipeline"""
    print("\nTesting image processing...")
    
    try:
        import numpy as np
        from PIL import Image
        import torch
        
        # Create a test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        print("✓ Test image created")
        
        # Test basic transformations
        tensor_image = torch.tensor(test_image).permute(2, 0, 1).float() / 255.0
        tensor_image = tensor_image.unsqueeze(0)  # Add batch dimension
        print(f"✓ Image tensor conversion working, shape: {tensor_image.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Image processing failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("QA-ViT + PNP-VQA Setup Test")
    print("="*60)
    
    all_passed = True
    
    # Test basic imports
    if not test_imports():
        all_passed = False
    
    # Test LAVIS imports
    if not test_lavis_imports():
        all_passed = False
    
    # Test model initialization
    if not test_model_initialization():
        all_passed = False
    
    # Test image processing
    if not test_image_processing():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! The setup is ready for the demo.")
        print("Run 'python3 run_demo.py' to start the demo.")
    else:
        print("✗ Some tests failed. Please install missing dependencies.")
        print("\nTo install missing dependencies:")
        print("pip3 install torch torchvision transformers pillow matplotlib opencv-python numpy")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())