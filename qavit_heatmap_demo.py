"""
QA-ViT + PnP-VQA Heatmap Generation Demo
This demo shows how to use the QA-ViT + PnP-VQA model to generate question-aware heatmaps.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import cv2
import seaborn as sns
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class QAViTHeatmapDemo:
    """
    Demo class for QA-ViT + PnP-VQA heatmap generation
    """
    
    def __init__(self, model, device='cuda'):
        """
        Initialize the demo
        
        Args:
            model: The QA-ViT + PnP-VQA model
            device: Device to run the model on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def load_and_preprocess_image(self, image_path: str, size: Tuple[int, int] = (224, 224)) -> Tuple[torch.Tensor, Image.Image]:
        """
        Load and preprocess image for the model
        
        Args:
            image_path: Path to the image file
            size: Target size for the image
            
        Returns:
            Tuple of (processed_tensor, original_pil_image)
        """
        # Load original image
        original_image = Image.open(image_path).convert('RGB')
        
        # Resize image
        resized_image = original_image.resize(size, Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        # Using CLIP preprocessing
        image_tensor = torch.tensor(np.array(resized_image)).float().permute(2, 0, 1) / 255.0
        
        # CLIP normalization values
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        
        image_tensor = (image_tensor - mean) / std
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return image_tensor, original_image
    
    def generate_heatmap(self, image_tensor: torch.Tensor, question: str, block_num: int = 7) -> torch.Tensor:
        """
        Generate question-aware heatmap using QA-ViT
        
        Args:
            image_tensor: Preprocessed image tensor
            question: Question string
            block_num: Which attention layer to use for heatmap
            
        Returns:
            Attention map tensor
        """
        with torch.no_grad():
            # Prepare samples dictionary
            samples = {
                'image': image_tensor.to(self.device),
                'text_input': [question]
            }
            
            # Generate attention maps using forward_itm
            samples = self.model.forward_itm(samples, block_num=block_num)
            
            # Extract gradcams (attention maps)
            attention_maps = samples['gradcams']  # Shape: [batch_size, H*W]
            
            return attention_maps.cpu()
    
    def visualize_heatmap(self, image: Image.Image, attention_map: torch.Tensor, 
                         question: str, save_path: str = None, show_patches: bool = True) -> Image.Image:
        """
        Visualize the attention heatmap overlaid on the original image
        
        Args:
            image: Original PIL image
            attention_map: Attention weights tensor
            question: Question string for title
            save_path: Path to save the visualization
            show_patches: Whether to show patch boundaries
            
        Returns:
            PIL Image with heatmap visualization
        """
        # Convert attention map to numpy
        attention_np = attention_map.squeeze().numpy()
        
        # Calculate grid size (assuming square patches)
        grid_size = int(np.sqrt(len(attention_np)))
        
        # Reshape to 2D grid
        attention_grid = attention_np.reshape(grid_size, grid_size)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attention heatmap
        sns.heatmap(attention_grid, ax=axes[1], cmap='hot', cbar=True, square=True)
        axes[1].set_title('QA-ViT Attention Map')
        axes[1].axis('off')
        
        # Overlay heatmap on image
        # Resize image to match model input size
        img_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        
        # Create heatmap overlay
        heatmap_resized = cv2.resize(attention_grid, (224, 224))
        heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
        
        # Apply colormap
        heatmap_colored = plt.cm.hot(heatmap_normalized)[:, :, :3]  # Remove alpha channel
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Blend with original image
        alpha = 0.6
        blended = cv2.addWeighted(img_array, alpha, heatmap_colored, 1-alpha, 0)
        
        axes[2].imshow(blended)
        axes[2].set_title('Heatmap Overlay')
        axes[2].axis('off')
        
        # Add patch boundaries if requested
        if show_patches:
            patch_size = 224 // grid_size
            for i in range(grid_size + 1):
                axes[2].axhline(y=i * patch_size - 0.5, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
                axes[2].axvline(x=i * patch_size - 0.5, color='white', linestyle='--', alpha=0.3, linewidth=0.5)
        
        # Set main title
        fig.suptitle(f'Question: "{question}"', fontsize=14, y=0.95)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        # Convert matplotlib figure to PIL Image
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        result_image = Image.fromarray(buf)
        
        plt.close(fig)  # Clean up
        
        return result_image
    
    def analyze_attention_statistics(self, attention_map: torch.Tensor, top_k: int = 5) -> dict:
        """
        Analyze attention statistics
        
        Args:
            attention_map: Attention weights tensor
            top_k: Number of top patches to analyze
            
        Returns:
            Dictionary with attention statistics
        """
        attention_np = attention_map.squeeze().numpy()
        grid_size = int(np.sqrt(len(attention_np)))
        
        # Find top-k attended patches
        top_indices = np.argsort(attention_np)[-top_k:][::-1]
        top_values = attention_np[top_indices]
        
        # Convert indices to 2D coordinates
        top_coords = [(idx // grid_size, idx % grid_size) for idx in top_indices]
        
        stats = {
            'grid_size': grid_size,
            'attention_mean': float(attention_np.mean()),
            'attention_std': float(attention_np.std()),
            'attention_max': float(attention_np.max()),
            'attention_min': float(attention_np.min()),
            'top_patches': list(zip(top_coords, top_values.tolist())),
            'entropy': float(-np.sum(attention_np * np.log(attention_np + 1e-8)))
        }
        
        return stats
    
    def demo_single_image(self, image_path: str, question: str, save_dir: str = './', block_num: int = 7):
        """
        Run complete demo on a single image-question pair
        
        Args:
            image_path: Path to image file
            question: Question string
            save_dir: Directory to save results
            block_num: Which attention layer to use
        """
        print(f"Processing: {image_path}")
        print(f"Question: {question}")
        print("-" * 50)
        
        # Load and preprocess image
        image_tensor, original_image = self.load_and_preprocess_image(image_path)
        print(f"Image loaded: {original_image.size}")
        
        # Generate heatmap
        attention_map = self.generate_heatmap(image_tensor, question, block_num)
        print(f"Attention map generated: {attention_map.shape}")
        
        # Analyze attention statistics
        stats = self.analyze_attention_statistics(attention_map)
        print(f"Attention Statistics:")
        print(f"  Mean: {stats['attention_mean']:.4f}")
        print(f"  Std: {stats['attention_std']:.4f}")
        print(f"  Max: {stats['attention_max']:.4f}")
        print(f"  Min: {stats['attention_min']:.4f}")
        print(f"  Entropy: {stats['entropy']:.4f}")
        print(f"  Top-5 patches (row, col): {stats['top_patches'][:5]}")
        
        # Visualize heatmap
        save_path = f"{save_dir}/heatmap_{hash(question) % 10000}.png"
        result_image = self.visualize_heatmap(
            original_image, attention_map, question, save_path
        )
        
        return {
            'attention_map': attention_map,
            'statistics': stats,
            'visualization': result_image
        }
    
    def demo_multiple_questions(self, image_path: str, questions: List[str], save_dir: str = './'):
        """
        Run demo with multiple questions on the same image
        
        Args:
            image_path: Path to image file
            questions: List of question strings
            save_dir: Directory to save results
        """
        print(f"Processing image: {image_path}")
        print(f"Questions: {len(questions)}")
        print("-" * 50)
        
        results = {}
        
        for i, question in enumerate(questions):
            print(f"\nQuestion {i+1}: {question}")
            result = self.demo_single_image(
                image_path, question, 
                save_dir=f"{save_dir}/q{i+1}_", 
                block_num=7
            )
            results[question] = result
        
        # Create comparison visualization
        self._create_comparison_visualization(image_path, questions, results, save_dir)
        
        return results
    
    def _create_comparison_visualization(self, image_path: str, questions: List[str], 
                                       results: dict, save_dir: str):
        """
        Create a comparison visualization for multiple questions
        """
        image_tensor, original_image = self.load_and_preprocess_image(image_path)
        
        num_questions = len(questions)
        fig, axes = plt.subplots(2, num_questions + 1, figsize=(4 * (num_questions + 1), 8))
        
        if num_questions == 1:
            axes = axes.reshape(2, 2)
        
        # Original image in first column
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')  # Empty bottom cell
        
        # Heatmaps for each question
        for i, question in enumerate(questions):
            attention_map = results[question]['attention_map']
            attention_np = attention_map.squeeze().numpy()
            grid_size = int(np.sqrt(len(attention_np)))
            attention_grid = attention_np.reshape(grid_size, grid_size)
            
            # Top row: attention heatmap
            sns.heatmap(attention_grid, ax=axes[0, i+1], cmap='hot', cbar=True, square=True)
            axes[0, i+1].set_title(f'Q{i+1}: {question[:30]}{"..." if len(question) > 30 else ""}')
            
            # Bottom row: overlay
            img_resized = original_image.resize((224, 224), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized)
            heatmap_resized = cv2.resize(attention_grid, (224, 224))
            heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
            heatmap_colored = plt.cm.hot(heatmap_normalized)[:, :, :3]
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            blended = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
            
            axes[1, i+1].imshow(blended)
            axes[1, i+1].set_title('Overlay')
            axes[1, i+1].axis('off')
        
        plt.tight_layout()
        save_path = f"{save_dir}/comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison visualization saved to: {save_path}")


def create_simple_demo_model():
    """
    Create a simplified demo model for testing
    This is a mock implementation for demonstration purposes
    """
    class SimpleMockModel:
        def __init__(self):
            self.device = 'cpu'
            
        def to(self, device):
            self.device = device
            return self
            
        def eval(self):
            return self
        
        def forward_itm(self, samples, block_num=7):
            # Create mock attention map
            batch_size = samples['image'].shape[0]
            grid_size = 14  # 14x14 patches for 224x224 image with 16x16 patches
            
            # Create question-aware attention pattern
            question = samples['text_input'][0].lower()
            
            # Simple mock: different patterns for different question types
            attention = np.random.rand(grid_size * grid_size) * 0.3
            
            if 'car' in question or 'vehicle' in question:
                # Focus on center-bottom (where cars typically are)
                for i in range(grid_size):
                    for j in range(grid_size):
                        if i > grid_size // 2 and abs(j - grid_size // 2) < grid_size // 4:
                            attention[i * grid_size + j] += 0.5
            
            elif 'person' in question or 'man' in question or 'woman' in question:
                # Focus on center regions
                for i in range(grid_size):
                    for j in range(grid_size):
                        if abs(i - grid_size // 2) < grid_size // 3 and abs(j - grid_size // 2) < grid_size // 3:
                            attention[i * grid_size + j] += 0.4
            
            elif 'sky' in question or 'cloud' in question:
                # Focus on top regions
                for i in range(grid_size // 2):
                    for j in range(grid_size):
                        attention[i * grid_size + j] += 0.6
            
            # Normalize
            attention = attention / attention.sum()
            
            samples['gradcams'] = torch.tensor(attention).unsqueeze(0)
            return samples
    
    return SimpleMockModel()


def main():
    """
    Main demo function
    """
    print("QA-ViT + PnP-VQA Heatmap Generation Demo")
    print("=" * 50)
    
    # For this demo, we'll use a mock model since the real model requires specific setup
    print("Creating demo model...")
    model = create_simple_demo_model()
    
    # Initialize demo
    demo = QAViTHeatmapDemo(model, device='cpu')
    
    # You can test with your own images by updating these paths
    sample_images = [
        # Add your image paths here
        # "path/to/your/image1.jpg",
        # "path/to/your/image2.jpg",
    ]
    
    sample_questions = [
        "What is the main object in the image?",
        "Where is the person located?",
        "What color is the car?",
        "Is there a building in the image?",
        "What is in the background?"
    ]
    
    # Create a sample image if none provided
    if not sample_images:
        print("Creating sample image for demo...")
        # Create a simple test image
        sample_img = Image.new('RGB', (224, 224), color='lightblue')
        draw = ImageDraw.Draw(sample_img)
        
        # Draw some simple shapes
        draw.rectangle([50, 150, 100, 200], fill='red')  # Car-like shape
        draw.ellipse([120, 80, 160, 140], fill='yellow')  # Person-like shape
        draw.rectangle([10, 10, 214, 60], fill='white')  # Sky area
        
        sample_path = "/tmp/demo_image.png"
        sample_img.save(sample_path)
        sample_images = [sample_path]
        print(f"Sample image saved to: {sample_path}")
    
    # Run single image demo
    if sample_images:
        print("\n" + "=" * 50)
        print("SINGLE IMAGE DEMO")
        print("=" * 50)
        
        result = demo.demo_single_image(
            sample_images[0], 
            "Where is the car in the image?",
            save_dir="./demo_results/"
        )
        
        print("\n" + "=" * 50)
        print("MULTIPLE QUESTIONS DEMO")
        print("=" * 50)
        
        results = demo.demo_multiple_questions(
            sample_images[0],
            sample_questions[:3],  # Use first 3 questions
            save_dir="./demo_results/"
        )
    
    print("\n" + "=" * 50)
    print("Demo completed! Check the ./demo_results/ folder for visualizations.")
    print("=" * 50)


if __name__ == "__main__":
    main()