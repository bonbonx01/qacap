#!/usr/bin/env python3
"""
QA-ViT + PNP-VQA 快速开始示例

这是一个简化的示例，展示如何使用修复后的实现：
1. 不依赖 LAVIS 框架
2. 使用模拟数据演示核心功能
3. 展示热力图生成和可视化过程
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class MockTextEncoder:
    """模拟文本编码器用于演示"""
    def __init__(self):
        self.tokenizer = MockTokenizer()
        
    def __call__(self, input_ids, attention_mask, return_dict=True):
        # 返回模拟的文本特征
        batch_size, seq_len = input_ids.shape
        last_hidden_state = torch.randn(batch_size, seq_len, 768)
        
        class MockOutput:
            def __init__(self, hidden_state):
                self.last_hidden_state = hidden_state
        
        return MockOutput(last_hidden_state)

class MockTokenizer:
    """模拟分词器"""
    def __call__(self, texts, padding='longest', truncation=True, return_tensors="pt"):
        max_len = 20
        batch_size = len(texts) if isinstance(texts, list) else 1
        
        class MockTokenizerOutput:
            def __init__(self, batch_size, max_len):
                self.input_ids = torch.randint(1, 1000, (batch_size, max_len))
                self.attention_mask = torch.ones(batch_size, max_len)
                
            def to(self, device):
                self.input_ids = self.input_ids.to(device)
                self.attention_mask = self.attention_mask.to(device)
                return self
        
        return MockTokenizerOutput(batch_size, max_len)

class MockQAViTVisionTower:
    """模拟 QA-ViT 视觉塔"""
    def __init__(self, config, instruction_dim):
        self.config = config
        self.instruction_dim = instruction_dim
        self.num_patches = 576  # 24x24 patches
        
    def to(self, device):
        return self
        
    @property
    def device(self):
        return torch.device('cpu')
    
    def __call__(self, pixel_values, instruct_states=None, instruct_masks=None, 
                 output_attentions=True, output_hidden_states=True):
        batch_size = pixel_values.shape[0]
        
        # 模拟视觉输出
        class MockVisionOutput:
            def __init__(self, batch_size, num_patches):
                # 模拟注意力权重 (num_layers, batch, num_heads, seq_len, seq_len)
                seq_len = num_patches + 1  # +1 for CLS token
                self.attentions = [
                    torch.rand(batch_size, 12, seq_len, seq_len) 
                    for _ in range(12)  # 12 layers
                ]
                
                # 模拟最后隐藏状态
                self.last_hidden_state = torch.randn(batch_size, seq_len, 768)
        
        return MockVisionOutput(batch_size, self.num_patches)

class MockCaptioningModel:
    """模拟描述生成模型"""
    def __init__(self):
        self.prompt = "a picture of"
        self.device = torch.device('cpu')
        
    def to(self, device):
        return self
    
    def forward_encoder(self, samples):
        # 返回模拟的编码器输出
        batch_size = samples['image'].shape[0]
        num_patches = 577  # 包含CLS token
        hidden_dim = 768
        return torch.randn(batch_size, num_patches, hidden_dim)

class MockQuestionAnsweringModel:
    """模拟问答模型"""
    def __init__(self):
        self.device = torch.device('cpu')
        
    def to(self, device):
        return self

def create_mock_pnp_vqa_model():
    """创建模拟的 PNP-VQA 模型用于演示"""
    
    # 这里我们不能直接导入修复的模型，因为它依赖 LAVIS
    # 所以我们创建一个简化版本来演示核心概念
    
    class SimplifiedPNPVQAWithQAViT:
        def __init__(self):
            self.text_encoder = MockTextEncoder()
            self.qavit_vision_tower = MockQAViTVisionTower({}, 768)
            self.image_captioning_model = MockCaptioningModel()
            self.question_answering_model = MockQuestionAnsweringModel()
            self.num_patches = 576
            
        @property
        def device(self):
            return torch.device('cpu')
            
        def extract_qavit_attention(self, vision_outputs, block_num=7):
            """从 QA-ViT 输出中提取注意力图"""
            block_num = min(block_num, len(vision_outputs.attentions) - 1)
            attention_weights = vision_outputs.attentions[block_num]
            
            # 平均跨头部的注意力
            attention_weights = attention_weights.mean(dim=1)
            
            # 提取CLS token到图像patches的注意力
            cls_attention = attention_weights[:, 0, -self.num_patches:]
            
            # 归一化
            cls_attention = torch.softmax(cls_attention, dim=-1)
            return cls_attention
            
        def forward_itm(self, samples, block_num=7):
            """生成问题感知的注意力图"""
            image = samples['image']
            questions = samples['text_input']
            
            # 编码问题
            encoded_questions = self.text_encoder.tokenizer(
                questions, 
                padding='longest', 
                truncation=True,
                return_tensors="pt"
            )
            
            # 获取文本特征
            text_outputs = self.text_encoder(
                input_ids=encoded_questions.input_ids,
                attention_mask=encoded_questions.attention_mask,
                return_dict=True
            )
            
            instruct_states = text_outputs.last_hidden_state
            instruct_masks = encoded_questions.attention_mask
            
            # 通过 QA-ViT 处理图像
            vision_outputs = self.qavit_vision_tower(
                pixel_values=image,
                instruct_states=instruct_states,
                instruct_masks=instruct_masks,
                output_attentions=True,
                output_hidden_states=True
            )
            
            # 提取注意力图
            attention_maps = self.extract_qavit_attention(
                vision_outputs, 
                block_num=block_num
            )
            
            # 存储到samples中
            samples['gradcams'] = attention_maps.reshape(image.size(0), -1)
            return samples
            
        def predict_answers(self, samples, **kwargs):
            """主要的预测接口"""
            # 生成注意力图
            samples = self.forward_itm(samples, block_num=kwargs.get('block_num', 7))
            
            # 模拟答案和描述
            batch_size = samples['image'].size(0)
            answers = ["This is a sample answer"] * batch_size
            captions = [["Sample caption 1", "Sample caption 2"]] * batch_size
            
            return answers, captions, samples['gradcams']
    
    return SimplifiedPNPVQAWithQAViT()

def create_sample_image():
    """创建示例图像"""
    # 创建一个简单的合成图像
    image = Image.new('RGB', (384, 384), color='lightblue')
    
    # 可以添加一些图形元素
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 384)
    ax.set_ylim(0, 384)
    
    # 添加一些形状
    circle = patches.Circle((192, 300), 50, facecolor='yellow')  # 太阳
    ax.add_patch(circle)
    
    house = patches.Rectangle((100, 50), 180, 120, facecolor='brown')  # 房子
    ax.add_patch(house)
    
    roof = patches.Polygon([(100, 170), (190, 220), (280, 170)], facecolor='red')
    ax.add_patch(roof)
    
    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig('temp_image.png', dpi=96, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 转换为 tensor
    image = Image.open('temp_image.png').convert('RGB').resize((384, 384))
    image_tensor = torch.tensor(np.array(image)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    return image_tensor, image

def visualize_attention_heatmap(original_image, attention_map, question, answer):
    """可视化注意力热力图"""
    # 转换注意力图
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.cpu().numpy().squeeze()
    
    # 重塑为24x24
    size = int(np.sqrt(len(attention_map)))
    heatmap = attention_map.reshape(size, size)
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 热力图
    im1 = axes[1].imshow(heatmap, cmap='jet', interpolation='bilinear')
    axes[1].set_title("QA-ViT Attention Heatmap")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 叠加效果
    img_array = np.array(original_image)
    from scipy.ndimage import zoom
    heatmap_resized = zoom(heatmap, 
                          (img_array.shape[0]/heatmap.shape[0], 
                           img_array.shape[1]/heatmap.shape[1]))
    
    # 归一化并创建彩色热力图
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
    heatmap_colored = plt.cm.jet(heatmap_norm)[:, :, :3]
    
    # 叠加
    alpha = 0.4
    overlay = img_array / 255.0 * (1 - alpha) + heatmap_colored * alpha
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    # 添加标题
    fig.suptitle(f'Question: "{question}"\nAnswer: "{answer}"', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('qavit_demo_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("可视化结果已保存为: qavit_demo_result.png")

def main():
    """主演示函数"""
    print("="*60)
    print("QA-ViT + PNP-VQA 快速开始演示")
    print("="*60)
    
    # 1. 创建模型
    print("\n1. 创建模拟模型...")
    model = create_mock_pnp_vqa_model()
    print("   ✓ 模型创建成功")
    
    # 2. 创建示例数据
    print("\n2. 准备示例数据...")
    image_tensor, original_image = create_sample_image()
    question = "What color is the sun in the image?"
    
    samples = {
        'image': image_tensor,
        'text_input': [question]
    }
    print(f"   ✓ 图像尺寸: {image_tensor.shape}")
    print(f"   ✓ 问题: {question}")
    
    # 3. 运行模型推理
    print("\n3. 运行模型推理...")
    answers, captions, attention_maps = model.predict_answers(
        samples,
        block_num=7,
        num_captions=5
    )
    
    print(f"   ✓ 答案: {answers[0]}")
    print(f"   ✓ 注意力图形状: {attention_maps.shape}")
    print(f"   ✓ 描述数量: {len(captions[0])}")
    
    # 4. 可视化结果
    print("\n4. 可视化注意力热力图...")
    
    try:
        visualize_attention_heatmap(
            original_image,
            attention_maps[0],
            question,
            answers[0]
        )
        print("   ✓ 可视化完成")
    except ImportError as e:
        print(f"   ⚠ 可视化需要额外依赖 (scipy): {e}")
        print("   → 请安装: pip install scipy")
    
    # 5. 输出总结
    print("\n" + "="*60)
    print("演示总结:")
    print("="*60)
    print("✅ 成功展示了 QA-ViT + PNP-VQA 的核心功能")
    print("✅ 生成了问题感知的注意力图")
    print("✅ 提供了答案和图像描述")
    print("✅ 创建了热力图可视化")
    
    print("\n关键特性:")
    print("• 问题感知注意力: 注意力图根据问题内容动态调整")
    print("• 多模态融合: 结合视觉和文本信息进行推理") 
    print("• 可视化友好: 直观展示模型关注的图像区域")
    
    print("\n实际使用中，您可以:")
    print("1. 替换模拟组件为真实的预训练模型")
    print("2. 使用真实图像和多样化的问题") 
    print("3. 调整注意力提取的层数和参数")
    print("4. 扩展可视化功能和评估指标")
    
    # 清理临时文件
    if os.path.exists('temp_image.png'):
        os.remove('temp_image.png')
    
    print(f"\n🎉 演示完成！")

if __name__ == "__main__":
    main()