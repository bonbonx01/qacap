#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: QA-ViT + PNP-VQA Heat Map Generation

This demo shows how to use the QA-ViT enhanced PNP-VQA model to generate
question-aware visual attention heat maps for Visual Question Answering.

Usage:
    python demo_qavit_heatmap.py --image path/to/image.jpg --question "What is in the image?"
    
Requirements:
    - PIL (Pillow)
    - matplotlib
    - numpy
    - torch
    - transformers
    - opencv-python (cv2)
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
import os
import sys
from pathlib import Path

# Add lavis to path
sys.path.append('/workspace')

from lavis.models.qacap_models.qacap_vqa import PNPVQAWithQAViT
from lavis.models.qacap_models.qavit_encoder import CLIPVisionTower
from lavis.processors.blip_processors import BlipImageBaseProcessor, BlipCaptionProcessor
from transformers import BertModel, BertTokenizer, T5ForConditionalGeneration, T5Tokenizer


class QAViTHeatMapDemo:
    """
    Demo class for generating heat maps using QA-ViT + PNP-VQA model
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.image_processor = None
        self.setup_model()
        
    def setup_model(self):
        """Initialize the QA-ViT enhanced PNP-VQA model"""
        print("Setting up QA-ViT + PNP-VQA model...")
        
        try:
            # Create mock captioning model (simplified for demo)
            self.captioning_model = self._create_mock_captioning_model()
            
            # Create mock QA model (simplified for demo) 
            self.qa_model = self._create_mock_qa_model()
            
            # Create text encoder
            self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
            self.text_encoder.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            # QA-ViT configuration
            qavit_config = {
                'vit_model': 'openai/clip-vit-base-patch16',
                'vit_layer': -1,
                'vit_type': 'qavit',
                'integration_point': 'all'
            }
            
            # Initialize the main model
            self.model = PNPVQAWithQAViT(
                image_captioning_model=self.captioning_model,
                question_answering_model=self.qa_model,
                qavit_model_config=qavit_config,
                text_encoder=self.text_encoder,
                offload_model=False
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            # Setup image processor
            self.image_processor = BlipImageBaseProcessor()
            
            print("Model setup completed successfully!")
            
        except Exception as e:
            print(f"Error setting up model: {e}")
            raise e
    
    def _create_mock_captioning_model(self):
        """Create a mock captioning model for demo purposes"""
        class MockCaptioningModel:
            def __init__(self):
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.prompt = "a picture of"
                self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
                # Create a simple encoder that returns random features
                self.vision_encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 768, kernel_size=16, stride=16),
                    torch.nn.Flatten(2),
                    torch.nn.Transpose(1, 2)
                )
                self.vision_encoder.to(self.device)
                
            def forward_encoder(self, samples):
                # Mock encoder that returns features for patch sampling
                images = samples['image'].to(self.device)
                # Resize to standard size if needed
                if images.shape[-1] != 224:
                    images = torch.nn.functional.interpolate(images, size=224, mode='bilinear')
                features = self.vision_encoder(images)
                return features  # [batch, num_patches, dim]
                
            def to(self, device):
                self.device = device
                self.vision_encoder.to(device)
                return self
                
        return MockCaptioningModel()
    
    def _create_mock_qa_model(self):
        """Create a mock QA model for demo purposes"""
        class MockQAModel:
            def __init__(self):
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
                
            def generate(self, **kwargs):
                # Mock generation - return dummy answers
                batch_size = kwargs['input_ids'].shape[0]
                dummy_answers = ["Yes"] * batch_size
                encoded = self.tokenizer(dummy_answers, return_tensors='pt', padding=True)
                return encoded.input_ids.to(self.device)
                
            def to(self, device):
                self.device = device
                return self
                
        return MockQAModel()
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess an image"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Preprocess using BLIP processor
            processed_image = self.image_processor(image, return_tensors="pt")['image']
            
            return image, processed_image
            
        except Exception as e:
            print(f"Error loading image: {e}")
            raise e
    
    def generate_heatmap(self, image_path, question, block_num=7):
        """Generate question-aware heat map for the given image and question"""
        print(f"Generating heat map for question: '{question}'")
        
        # Load and preprocess image
        original_image, processed_image = self.load_and_preprocess_image(image_path)
        
        # Prepare input samples
        samples = {
            'image': processed_image.to(self.device),
            'text_input': [question]
        }
        
        try:
            # Generate attention maps using QA-ViT
            with torch.no_grad():
                samples = self.model.forward_itm(samples, block_num=block_num)
            
            # Extract attention map
            attention_map = samples['gradcams'].cpu().numpy()[0]  # First (and only) sample
            
            # Reshape to 2D grid (assuming square patches)
            grid_size = int(np.sqrt(len(attention_map)))
            attention_map_2d = attention_map.reshape(grid_size, grid_size)
            
            return original_image, attention_map_2d, attention_map
            
        except Exception as e:
            print(f"Error generating heat map: {e}")
            raise e
    
    def visualize_heatmap(self, original_image, attention_map_2d, question, save_path=None):
        """Visualize the heat map overlaid on the original image"""
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Heat map only
        im = axes[1].imshow(attention_map_2d, cmap='hot', interpolation='bilinear')
        axes[1].set_title(f"QA-ViT Attention Map\nQuestion: {question}")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay heat map on original image
        # Resize attention map to match image size
        image_np = np.array(original_image)
        h, w = image_np.shape[:2]
        
        # Resize attention map
        attention_resized = cv2.resize(attention_map_2d, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Create overlay
        axes[2].imshow(image_np)
        axes[2].imshow(attention_resized, cmap='hot', alpha=0.6, interpolation='bilinear')
        axes[2].set_title("Attention Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heat map saved to: {save_path}")
        
        return fig
    
    def run_demo(self, image_path, question, output_dir=None, block_num=7):
        """Run the complete demo pipeline"""
        print("="*60)
        print("QA-ViT + PNP-VQA Heat Map Generation Demo")
        print("="*60)
        
        try:
            # Generate heat map
            original_image, attention_map_2d, attention_map = self.generate_heatmap(
                image_path, question, block_num=block_num
            )
            
            # Create output directory if specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"heatmap_{Path(image_path).stem}.png")
            else:
                save_path = f"heatmap_{Path(image_path).stem}.png"
            
            # Visualize
            fig = self.visualize_heatmap(original_image, attention_map_2d, question, save_path)
            
            # Show statistics
            print(f"\nAttention Statistics:")
            print(f"- Max attention: {attention_map.max():.4f}")
            print(f"- Min attention: {attention_map.min():.4f}")
            print(f"- Mean attention: {attention_map.mean():.4f}")
            print(f"- Std attention: {attention_map.std():.4f}")
            
            return fig, attention_map_2d
            
        except Exception as e:
            print(f"Demo failed: {e}")
            raise e


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="QA-ViT Heat Map Generation Demo")
    parser.add_argument("--image", type=str, required=True, 
                       help="Path to input image")
    parser.add_argument("--question", type=str, required=True,
                       help="Question about the image")
    parser.add_argument("--output_dir", type=str, default="./demo_outputs",
                       help="Output directory for results")
    parser.add_argument("--block_num", type=int, default=7,
                       help="Attention block number to use (0-11)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    try:
        # Initialize demo
        demo = QAViTHeatMapDemo(device=device)
        
        # Run demo
        fig, attention_map = demo.run_demo(
            image_path=args.image,
            question=args.question,
            output_dir=args.output_dir,
            block_num=args.block_num
        )
        
        # Show plot
        plt.show()
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Error running demo: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())