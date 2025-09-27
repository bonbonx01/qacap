# QA-ViT + PNP-VQA Heat Map Generation Demo

这个项目展示了如何使用 QA-ViT 增强的 PNP-VQA 模型来生成问题感知的视觉注意力热力图。

## 项目概述

### 原始问题分析

您提供的 `qacap_vqa.py` 代码存在以下问题：

1. **`from_config` 方法问题**: 缺少 `qavit_model_config` 和 `text_encoder` 参数传递
2. **导入路径问题**: `CLIPVisionTower` 的导入路径不正确 
3. **设备处理问题**: `self.device` 属性可能不存在，需要用 `next(self.parameters()).device`
4. **模型引用错误**: 在 `predict_answers` 方法中仍然引用了被替换的 `image_question_matching_model`
5. **属性假设问题**: 假设某些属性存在但可能没有正确初始化

### 修复方案

我创建了修正版本的代码，主要修复包括：

1. ✅ **修复了 `from_config` 方法**: 正确处理 QA-ViT 配置和文本编码器初始化
2. ✅ **修正了导入路径**: 使用正确的 `CLIPVisionTower` 导入
3. ✅ **添加了 `device` 属性**: 通过 `@property` 装饰器正确获取设备
4. ✅ **移除了错误的模型引用**: 清理了对旧模型的引用
5. ✅ **增强了错误处理**: 添加了更好的错误检查和边界处理

## 文件结构

```
/workspace/
├── lavis/models/qacap_models/
│   ├── qacap_vqa.py              # 修正后的 QA-ViT + PNP-VQA 模型
│   ├── clip_qavit.py             # QA-ViT 实现
│   └── qavit_encoder.py          # QA-ViT 编码器包装器
├── configs/models/qacap/
│   └── pnp_vqa_qavit.yaml        # 模型配置文件
├── demo_qavit_heatmap.py         # 完整的热力图生成demo
├── run_demo.py                   # 快速演示启动器
└── README_QAVIT_DEMO.md          # 本文档
```

## 安装依赖

### 自动安装 (推荐)

```bash
python3 install_dependencies.py
```

### 手动安装

```bash
pip3 install torch torchvision transformers
pip3 install pillow matplotlib opencv-python numpy omegaconf
pip3 install accelerate  # 可选，用于模型加速
```

### 验证安装

```bash
python3 test_setup.py
```

如果看到所有测试通过，说明环境配置正确。

## 使用方法

### 1. 快速开始

运行预定义示例：
```bash
python run_demo.py
```

运行自定义示例：
```bash
python run_demo.py --custom --image path/to/your/image.jpg --question "What is in the image?"
```

### 2. 完整 Demo

使用完整的 demo 脚本：
```bash
python demo_qavit_heatmap.py --image path/to/image.jpg --question "Your question here"
```

参数选项：
- `--image`: 输入图像路径
- `--question`: 关于图像的问题
- `--output_dir`: 输出目录 (默认: ./demo_outputs)
- `--block_num`: 使用的注意力层编号 (默认: 7)
- `--device`: 使用的设备 (cuda/cpu/auto)

### 3. 编程式使用

```python
from demo_qavit_heatmap import QAViTHeatMapDemo

# 初始化 demo
demo = QAViTHeatMapDemo(device='cuda')

# 生成热力图
fig, attention_map = demo.run_demo(
    image_path="path/to/image.jpg",
    question="What objects are in the image?",
    output_dir="./outputs"
)
```

## 核心功能

### 1. 问题感知注意力

QA-ViT 模型能够根据问题内容生成相应的视觉注意力：
- 问题："红色物体在哪里？" → 注意力集中在红色区域
- 问题："有什么文字？" → 注意力集中在文字区域  
- 问题："图像中有多少人？" → 注意力集中在人物区域

### 2. 可视化输出

Demo 生成三种可视化：
1. **原始图像**: 输入的原始图像
2. **注意力热力图**: 纯注意力权重可视化 
3. **叠加热力图**: 热力图叠加在原始图像上

### 3. 统计信息

输出注意力分布的统计信息：
- 最大注意力值
- 最小注意力值  
- 平均注意力值
- 注意力标准差

## 技术细节

### QA-ViT 架构

QA-ViT (Question-Aware Vision Transformer) 通过以下方式增强标准 ViT：

1. **指令融合**: 将问题编码为指令状态，融入视觉编码过程
2. **跨模态注意力**: 在多个 Transformer 层中融合视觉和文本信息
3. **问题感知特征**: 生成针对特定问题的视觉特征表示

### 实现特点

- **模块化设计**: 易于替换和扩展组件
- **设备兼容**: 支持 CPU 和 GPU 推理
- **内存优化**: 支持模型卸载以节省内存
- **错误处理**: 完善的错误检查和处理机制

## 示例结果

运行 demo 后，您将在输出目录看到：
- `heatmap_[image_name].png`: 可视化结果
- 控制台输出包含注意力统计信息

典型的输出统计信息：
```
Attention Statistics:
- Max attention: 0.0856
- Min attention: 0.0021  
- Mean attention: 0.0041
- Std attention: 0.0098
```

## 故障排除

### 常见问题

1. **CUDA 内存不足**:
   ```bash
   python demo_qavit_heatmap.py --device cpu  # 使用 CPU
   ```

2. **模型加载失败**:
   - 检查网络连接（需要下载预训练模型）
   - 确保有足够磁盘空间

3. **图像格式不支持**:
   - 支持的格式：JPG, PNG, BMP
   - 确保图像文件完整且未损坏

### 性能优化

- **GPU 加速**: 使用 CUDA 设备可显著提升速度
- **批处理**: 可以修改代码支持批量图像处理  
- **模型量化**: 可以应用量化技术减少内存使用

## 扩展功能

### 自定义 QA-ViT 配置

修改配置文件 `/workspace/configs/models/qacap/pnp_vqa_qavit.yaml`:

```yaml
qavit_config:
  vit_model: "openai/clip-vit-large-patch14"  # 使用更大的模型
  integration_point: "late"  # 改变融合策略
  instruction_dim: 1024  # 调整指令维度
```

### 添加新的可视化

您可以扩展 `visualize_heatmap` 方法来添加：
- 多层注意力可视化
- 动态热力图动画
- 注意力区域标注

## 许可证

本项目遵循 BSD-3-Clause 许可证。