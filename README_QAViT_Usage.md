# QA-ViT + PnP-VQA 热力图生成使用说明

这是一个使用 QA-ViT 替换 PnP-VQA 中 GradCAM 生成方式的实现，用于生成问题感知的注意力热力图。

## 修复的问题

### 原始代码中的问题：

1. **配置不匹配**: `from_config` 方法与 `__init__` 方法参数不一致
2. **设备管理**: `self.device` 未正确初始化
3. **架构假设**: 对 QA-ViT 注意力结构做了不准确的假设
4. **依赖缺失**: 缺少 `CLIPVisionTower` 的正确实现
5. **梯度计算**: 某些操作在 `no_grad()` 下进行但后续需要梯度

### 修复后的改进：

1. **统一接口**: 重新设计了 `__init__` 和 `from_config` 方法
2. **正确的设备管理**: 从子模型中获取设备信息
3. **使用真实的 QA-ViT**: 集成了现有的 `InstructCLIPVisionModel`
4. **改进的注意力提取**: 更准确的注意力图提取逻辑
5. **完整的预处理**: 添加了正确的图像预处理流程

## 文件说明

- `qacap_vqa_fixed.py`: 修复后的 QA-ViT + PnP-VQA 模型实现
- `qavit_heatmap_demo.py`: 完整的热力图生成演示代码
- `qavit_config_example.yaml`: 模型配置示例文件
- `README_QAViT_Usage.md`: 本使用说明文档

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision
pip install transformers
pip install matplotlib seaborn opencv-python
pip install pillow numpy
```

### 2. 运行演示

```python
from qavit_heatmap_demo import QAViTHeatmapDemo, create_simple_demo_model

# 创建演示模型（用于测试）
model = create_simple_demo_model()

# 初始化演示
demo = QAViTHeatmapDemo(model, device='cpu')

# 运行单张图片演示
result = demo.demo_single_image(
    image_path="your_image.jpg",
    question="What is in the image?",
    save_dir="./results/"
)

# 运行多问题演示
results = demo.demo_multiple_questions(
    image_path="your_image.jpg",
    questions=[
        "Where is the car?",
        "What color is the building?",
        "Is there a person in the image?"
    ],
    save_dir="./results/"
)
```

### 3. 使用真实模型

```python
from qacap_vqa_fixed import PNPVQAWithQAViT
from lavis.models import load_model
from lavis.common.config import Config

# 加载配置
config = Config.from_file("qavit_config_example.yaml")

# 创建模型
model = PNPVQAWithQAViT.from_config(config.model)

# 初始化演示
demo = QAViTHeatmapDemo(model, device='cuda')

# 使用模型
image_path = "path/to/your/image.jpg"
question = "What is the main object in the image?"

result = demo.demo_single_image(image_path, question)
```

## 核心功能

### 1. 问题感知注意力生成

模型通过以下步骤生成问题感知的注意力图：

```python
def forward_itm(self, samples, block_num=7):
    # 1. 编码问题为指令特征
    instruct_states = self.encode_questions(samples['text_input'])
    
    # 2. 通过 QA-ViT 处理图像和问题
    vision_outputs = self.qavit_vision_model(
        pixel_values=samples['image'],
        instruct_states=instruct_states,
        output_attentions=True
    )
    
    # 3. 提取注意力图
    attention_maps = self.extract_qavit_attention(vision_outputs, block_num)
    
    return attention_maps
```

### 2. 热力图可视化

演示代码提供了丰富的可视化功能：

- 原始图像显示
- 注意力热力图
- 热力图叠加显示
- 补丁边界显示
- 多问题对比显示

### 3. 注意力分析

```python
stats = demo.analyze_attention_statistics(attention_map)
print(f"注意力统计:")
print(f"  平均值: {stats['attention_mean']:.4f}")
print(f"  标准差: {stats['attention_std']:.4f}")
print(f"  最大值: {stats['attention_max']:.4f}")
print(f"  熵值: {stats['entropy']:.4f}")
print(f"  前5个关注区域: {stats['top_patches']}")
```

## 模型架构

### QA-ViT 集成

```
输入图像 → 图像补丁 → Vision Transformer
                          ↓
问题文本 → 文本编码器 → 指令特征 → QA-ViT 注意力层
                          ↓
                      问题感知的视觉特征
                          ↓
                      注意力图提取
```

### 关键组件

1. **InstructCLIPVisionModel**: 核心的 QA-ViT 视觉模型
2. **MMCLIPAttention**: 多模态注意力机制
3. **文本编码器**: 将问题编码为指令特征
4. **注意力提取器**: 从多层注意力中提取热力图

## 配置参数

### 重要参数说明

- `integration_point`: 指令集成时机
  - `"all"`: 所有层都集成指令
  - `"early"`: 前半部分层集成
  - `"late"`: 后半部分层集成

- `block_num`: 用于热力图的注意力层索引（0-11）
- `instruction_dim`: 指令特征维度，需与文本编码器匹配
- `image_size`: 输入图像尺寸
- `patch_size`: 视觉补丁大小

## 性能优化

### 内存管理

```python
# 模型卸载（适用于大型模型）
model = PNPVQAWithQAViT(..., offload_model=True)

# 梯度检查点（节省内存）
qavit_config.gradient_checkpointing = True
```

### 批处理支持

```python
# 批量处理多个图像-问题对
samples = {
    'image': torch.stack([img1, img2, img3]),  # [batch_size, 3, H, W]
    'text_input': ["Question 1", "Question 2", "Question 3"]
}

attention_maps = model.forward_itm(samples)  # [batch_size, num_patches]
```

## 常见问题

### Q: 热力图不够清晰怎么办？
A: 尝试调整 `block_num` 参数，使用不同层的注意力：
- 浅层（0-3）：关注低级特征
- 中层（4-7）：关注中级特征  
- 深层（8-11）：关注高级语义

### Q: 如何提升问题理解能力？
A: 可以尝试：
- 使用更强的文本编码器（如 RoBERTa）
- 调整 `integration_point` 参数
- 增加指令特征维度

### Q: 内存不足怎么办？
A: 可以：
- 启用 `offload_model=True`
- 减小 `batch_size`
- 使用梯度检查点
- 降低图像分辨率

## 扩展功能

### 自定义注意力提取

```python
def custom_attention_extractor(self, vision_outputs, block_num=7):
    """自定义的注意力提取逻辑"""
    attentions = vision_outputs.attentions[block_num]
    
    # 实现您的提取逻辑
    custom_attention = process_attention(attentions)
    
    return custom_attention
```

### 多尺度注意力

```python
def multi_scale_attention(self, vision_outputs):
    """提取多个尺度的注意力"""
    multi_scale_maps = []
    for layer_idx in [3, 7, 11]:  # 不同深度的层
        attention = self.extract_qavit_attention(vision_outputs, layer_idx)
        multi_scale_maps.append(attention)
    
    return torch.stack(multi_scale_maps)
```

## 引用

如果使用此代码，请引用相关论文：

```bibtex
@article{qavit2023,
  title={QA-ViT: Question-Aware Vision Transformer for Visual Question Answering},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2023}
}

@article{pnpvqa2022,
  title={PNP-VQA: Plug-and-Play VQA with Frozen Language Models},
  author={Anthony Meng Huat Tiong et al.},
  journal={arXiv preprint arXiv:2210.08773},
  year={2022}
}
```