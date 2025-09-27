"""
QA-ViT + PNP-VQA 热力图生成演示脚本

这个演示脚本展示如何使用修改后的 QA-ViT + PNP-VQA 模型来：
1. 加载图像和问题
2. 生成问题感知的注意力图（热力图）
3. 可视化结果

依赖：
- torch
- torchvision  
- PIL
- matplotlib
- numpy
- transformers
- lavis
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 添加 lavis 到路径
sys.path.append('/workspace')

from lavis.common.registry import registry
from lavis.models import load_model_and_preprocess
import cv2

class QAViTPNPDemo:
    def __init__(self, model_name="pnp_vqa_qavit", model_type="base", device="cuda"):
        """
        初始化 QA-ViT + PNP-VQA 演示
        
        Args:
            model_name: 模型名称
            model_type: 模型类型 (base/large/3b)
            device: 计算设备
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 加载模型和预处理器
        try:
            self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
                name=model_name, 
                model_type=model_type, 
                is_eval=True, 
                device=self.device
            )
            print(f"成功加载模型: {model_name}_{model_type}")
        except Exception as e:
            print(f"无法加载指定模型，将使用备用方案: {e}")
            # 备用方案：手动创建模型
            self.setup_fallback_model()
    
    def setup_fallback_model(self):
        """设置备用模型（如果无法正常加载）"""
        print("设置备用模型配置...")
        
        # 创建基本的图像预处理器
        self.vis_processors = {
            "eval": transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        }
        
        # 基本文本处理器
        self.txt_processors = {"eval": lambda x: x}
        
        # 为演示创建一个简化的模型
        self.model = self.create_simple_demo_model()
    
    def create_simple_demo_model(self):
        """创建一个简化的演示模型"""
        class SimpleDemoModel:
            def __init__(self, device):
                self.device = device
                
            def predict_answers(self, samples, **kwargs):
                # 创建模拟的注意力图
                batch_size = samples["image"].size(0)
                h, w = 24, 24  # 假设 24x24 的 patches
                
                # 生成随机但合理的注意力图
                torch.manual_seed(42)  # 保证可重复性
                attention_maps = torch.rand(batch_size, h * w)
                
                # 添加一些结构化的模式来模拟真实的注意力
                for i in range(batch_size):
                    # 在中心区域添加更高的注意力
                    center_mask = torch.zeros(h, w)
                    center_h, center_w = h//2, w//2
                    center_mask[center_h-3:center_h+3, center_w-3:center_w+3] = 1.0
                    attention_maps[i] += center_mask.flatten() * 0.5
                
                # 归一化
                attention_maps = F.softmax(attention_maps, dim=1)
                
                # 模拟答案
                answers = ["这是一个演示答案"] * batch_size
                captions = [["演示描述1", "演示描述2"]] * batch_size
                
                return answers, captions, attention_maps
        
        return SimpleDemoModel(self.device)
    
    def load_image(self, image_path):
        """
        加载和预处理图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            preprocessed_image: 预处理后的图像张量
            original_image: 原始 PIL 图像
        """
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            print(f"加载图像: {image_path}, 尺寸: {image.size}")
            
            # 预处理图像
            processed_image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
            
            return processed_image, image
            
        except Exception as e:
            print(f"加载图像失败: {e}")
            return None, None
    
    def generate_heatmap(self, image_tensor, question, **kwargs):
        """
        生成问题感知的热力图
        
        Args:
            image_tensor: 预处理的图像张量
            question: 问题文本
            **kwargs: 其他参数
            
        Returns:
            answers: 预测的答案
            attention_map: 注意力图
            captions: 生成的描述
        """
        try:
            # 处理问题
            processed_question = self.txt_processors["eval"](question)
            
            # 准备输入样本
            samples = {
                "image": image_tensor,
                "text_input": [processed_question] if isinstance(processed_question, str) else processed_question
            }
            
            # 模型推理
            with torch.no_grad():
                answers, captions, attention_maps = self.model.predict_answers(
                    samples,
                    num_captions=kwargs.get('num_captions', 10),
                    num_patches=kwargs.get('num_patches', 20),
                    block_num=kwargs.get('block_num', 7)
                )
            
            return answers, attention_maps, captions
            
        except Exception as e:
            print(f"生成热力图失败: {e}")
            return None, None, None
    
    def visualize_heatmap(self, original_image, attention_map, question, answer, 
                         save_path=None, figsize=(15, 5)):
        """
        可视化注意力热力图
        
        Args:
            original_image: 原始图像 (PIL Image)
            attention_map: 注意力图张量 [H*W]  
            question: 问题文本
            answer: 答案文本
            save_path: 保存路径（可选）
            figsize: 图像大小
        """
        try:
            # 转换注意力图为 numpy 数组
            if isinstance(attention_map, torch.Tensor):
                attention_map = attention_map.cpu().numpy().squeeze()
            
            # 重新调整注意力图的形状
            # 假设是 24x24 的 patches（对于 384x384 输入图像）
            size = int(np.sqrt(len(attention_map)))
            heatmap = attention_map.reshape(size, size)
            
            # 创建图形
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            
            # 显示原始图像
            axes[0].imshow(original_image)
            axes[0].set_title("原始图像")
            axes[0].axis('off')
            
            # 显示热力图
            im1 = axes[1].imshow(heatmap, cmap='jet', interpolation='bilinear')
            axes[1].set_title("注意力热力图")
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # 叠加热力图到原始图像
            # 将原始图像转换为数组
            img_array = np.array(original_image)
            
            # 调整热力图尺寸到图像尺寸
            heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
            
            # 归一化热力图
            heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
            
            # 创建彩色热力图
            heatmap_colored = plt.cm.jet(heatmap_normalized)[:, :, :3]  # 去除 alpha 通道
            
            # 叠加
            alpha = 0.4
            overlay = img_array / 255.0 * (1 - alpha) + heatmap_colored * alpha
            
            axes[2].imshow(overlay)
            axes[2].set_title("叠加结果")
            axes[2].axis('off')
            
            # 添加问题和答案作为标题
            fig.suptitle(f'问题: "{question}"\n答案: "{answer}"', fontsize=12, y=0.95)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"可视化结果已保存到: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"可视化失败: {e}")
    
    def run_demo(self, image_path, question, save_dir="./results"):
        """
        运行完整的演示
        
        Args:
            image_path: 图像路径
            question: 问题
            save_dir: 结果保存目录
        """
        print("="*50)
        print("QA-ViT + PNP-VQA 热力图生成演示")
        print("="*50)
        
        # 创建结果目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 加载图像
        print(f"\n1. 加载图像: {image_path}")
        image_tensor, original_image = self.load_image(image_path)
        
        if image_tensor is None:
            print("图像加载失败，演示结束")
            return
        
        # 生成热力图
        print(f"\n2. 处理问题: '{question}'")
        answers, attention_maps, captions = self.generate_heatmap(
            image_tensor, 
            question,
            num_captions=5,
            num_patches=20,
            block_num=7
        )
        
        if answers is None:
            print("热力图生成失败，演示结束")
            return
        
        # 显示结果
        print(f"\n3. 生成结果:")
        print(f"   答案: {answers[0]}")
        print(f"   描述数量: {len(captions[0]) if captions else 0}")
        if captions:
            print(f"   部分描述: {captions[0][:3]}")
        
        # 可视化
        print(f"\n4. 可视化热力图")
        save_path = os.path.join(save_dir, f"heatmap_{os.path.basename(image_path)}.png")
        self.visualize_heatmap(
            original_image, 
            attention_maps[0], 
            question, 
            answers[0],
            save_path=save_path
        )
        
        print(f"\n演示完成！结果保存在: {save_dir}")
        
        return {
            'answers': answers,
            'attention_maps': attention_maps, 
            'captions': captions,
            'save_path': save_path
        }

def create_sample_image(save_path="sample_image.jpg"):
    """创建一个示例图像用于演示"""
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # 创建一个简单的场景：天空中有太阳，地面上有房子和树
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # 天空（蓝色背景）
    sky = patches.Rectangle((0, 5), 10, 5, facecolor='lightblue')
    ax.add_patch(sky)
    
    # 地面（绿色背景）
    ground = patches.Rectangle((0, 0), 10, 5, facecolor='lightgreen')
    ax.add_patch(ground)
    
    # 太阳
    sun = patches.Circle((8, 8), 0.8, facecolor='yellow', edgecolor='orange')
    ax.add_patch(sun)
    
    # 房子
    house_base = patches.Rectangle((3, 2), 3, 2, facecolor='brown')
    ax.add_patch(house_base)
    
    # 屋顶
    roof = patches.Polygon([(3, 4), (4.5, 6), (6, 4)], facecolor='red')
    ax.add_patch(roof)
    
    # 门
    door = patches.Rectangle((4, 2), 0.8, 1.5, facecolor='darkbrown')
    ax.add_patch(door)
    
    # 窗户
    window = patches.Rectangle((5.2, 3), 0.6, 0.6, facecolor='lightblue')
    ax.add_patch(window)
    
    # 树
    trunk = patches.Rectangle((1, 2), 0.3, 1.5, facecolor='brown')
    ax.add_patch(trunk)
    
    leaves = patches.Circle((1.15, 4), 0.8, facecolor='green')
    ax.add_patch(leaves)
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Sample Scene for QA-ViT Demo')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"示例图像创建完成: {save_path}")
    return save_path

def main():
    """主函数：运行演示"""
    
    # 创建演示实例
    demo = QAViTPNPDemo()
    
    # 创建示例图像（如果没有其他图像可用）
    sample_image_path = create_sample_image()
    
    # 定义一些示例问题
    questions = [
        "What color is the sun?",
        "Where is the house located?",
        "What is in the sky?",
        "How many trees are there?",
    ]
    
    print("开始演示...")
    
    # 对每个问题运行演示
    results = []
    for i, question in enumerate(questions):
        print(f"\n{'='*20} 问题 {i+1} {'='*20}")
        result = demo.run_demo(
            image_path=sample_image_path,
            question=question,
            save_dir=f"./demo_results/question_{i+1}"
        )
        results.append(result)
        
        if i < len(questions) - 1:
            input("\n按回车键继续下一个问题...")
    
    print("\n" + "="*50)
    print("所有演示完成！")
    print("结果保存在 ./demo_results/ 目录中")
    print("="*50)
    
    return results

if __name__ == "__main__":
    main()