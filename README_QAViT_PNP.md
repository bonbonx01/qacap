# QA-ViT + PNP-VQA 实现说明和演示

本项目实现了使用 QA-ViT（问题感知视觉Transformer）替换 PNP-VQA 中传统 GradCAM 注意力机制的方案，并提供了完整的演示代码。

## 📋 项目概述

### 原始问题分析
原始的 `qacap_vqa.py` 实现存在以下问题：

1. **配置方法不匹配**: `from_config()` 方法仍然创建 `image_question_matching_model`，但 `__init__` 中已不使用
2. **缺失属性引用**: 代码中引用了不存在的 `self.image_question_matching_model` 属性  
3. **参数结构不一致**: `from_config` 和 `__init__` 的参数不匹配
4. **文本编码器假设**: 假设文本编码器具有特定结构，缺乏灵活性

### 解决方案
✅ **修复了所有已识别的问题**
✅ **添加了健壮的错误处理**
✅ **支持多种文本编码器类型**
✅ **保持了与原始API的兼容性**

## 📁 文件结构

```
/workspace/
├── qacap_vqa_fixed.py      # 修复后的主要实现
├── qavit_pnp_demo.py       # 完整的演示脚本
├── demo_config.yaml        # 配置文件
├── test_demo.py            # 完整功能测试
├── simple_test.py          # 简化结构测试
└── README_QAViT_PNP.md     # 本说明文档
```

## 🔧 主要改进

### 1. 架构改进
- **QA-ViT 集成**: 使用问题感知的视觉注意力替换传统 GradCAM
- **模块化设计**: 清晰的组件分离和接口定义
- **设备管理**: 自动处理GPU/CPU设备转换

### 2. 代码质量
- **错误处理**: 添加了完善的异常处理机制
- **类型兼容**: 支持多种文本编码器类型（BERT、T5等）
- **参数验证**: 添加了输入参数的有效性检查

### 3. 功能扩展
- **热力图可视化**: 提供直观的注意力图展示
- **批处理支持**: 支持批量图像处理
- **配置驱动**: 通过配置文件灵活调整参数

## 🚀 使用说明

### 环境要求
```bash
pip install torch torchvision transformers
pip install pillow matplotlib numpy opencv-python
pip install lavis  # 可选，用于完整功能
```

### 基本使用

1. **导入模型**:
```python
from qacap_vqa_fixed import PNPVQAWithQAViT

# 从配置创建模型
model = PNPVQAWithQAViT.from_config(config)
```

2. **运行推理**:
```python
samples = {
    "image": image_tensor,      # [batch, 3, H, W]
    "text_input": [question]    # 问题列表
}

answers, captions, attention_maps = model.predict_answers(samples)
```

### 演示脚本使用

```bash
# 运行完整演示
python qavit_pnp_demo.py

# 或者自定义演示
from qavit_pnp_demo import QAViTPNPDemo

demo = QAViTPNPDemo()
result = demo.run_demo(
    image_path="your_image.jpg",
    question="What is in the image?",
    save_dir="./results"
)
```

## 🎯 核心特性

### 1. 问题感知注意力
- 使用 QA-ViT 生成与问题相关的注意力图
- 相比传统 GradCAM 更准确地定位相关区域
- 支持多层注意力提取和分析

### 2. 灵活的文本处理
```python
# 自动适配不同的文本编码器
if hasattr(self.text_encoder, 'bert'):
    # BERT-like 编码器处理
elif hasattr(self.text_encoder, 'encoder'):
    # T5-like 编码器处理  
else:
    # 通用编码器处理
```

### 3. 热力图可视化
- 原始图像显示
- 注意力热力图
- 叠加可视化效果
- 支持自定义颜色映射

## 📊 示例输出

演示脚本将生成以下内容：
- **答案预测**: "The sun is yellow"
- **注意力图**: 高亮显示与问题相关的图像区域
- **描述生成**: ["A bright yellow sun in the sky", "Sunlight illuminates the scene"]
- **可视化图像**: 包含原图、热力图和叠加效果的组合图

## 🧪 测试验证

运行测试确保功能正常：

```bash
# 基本结构测试（无需外部依赖）
python simple_test.py

# 完整功能测试（需要安装依赖）  
python test_demo.py
```

### 测试覆盖
- ✅ 代码结构完整性
- ✅ 关键组件存在性
- ✅ 配置文件正确性
- ✅ 问题修复验证
- ✅ 功能改进确认

## 🔍 技术细节

### QA-ViT 注意力提取
```python
def extract_qavit_attention(self, vision_outputs, block_num=7):
    # 从指定层获取注意力权重
    attention_weights = vision_outputs.attentions[block_num]
    # 平均跨头部的注意力
    attention_weights = attention_weights.mean(dim=1)
    # 提取CLS token到图像块的注意力
    cls_attention = attention_weights[:, 0, -self.num_patches:]
    return torch.softmax(cls_attention, dim=-1)
```

### 设备管理
```python
@property 
def device(self):
    return next(self.parameters()).device

# 自动设备转换
instruct_states = instruct_states.to(self.device)
pixel_values = pixel_values.to(self.device)
```

## 🤝 贡献和扩展

### 可能的改进方向
1. **多尺度注意力**: 支持不同分辨率的注意力分析
2. **注意力融合**: 结合多层注意力信息
3. **实时处理**: 优化推理速度用于实时应用
4. **批量优化**: 进一步优化批处理性能

### 自定义扩展
- 修改 `extract_qavit_attention()` 方法自定义注意力提取逻辑
- 调整 `visualize_heatmap()` 方法改变可视化效果
- 扩展 `run_demo()` 方法添加新的演示功能

## 📝 更新日志

### v1.0 (当前版本)
- ✅ 修复原始实现中的所有问题
- ✅ 添加完整的 QA-ViT 支持
- ✅ 实现热力图生成和可视化
- ✅ 提供完整的演示和测试代码
- ✅ 支持多种配置和扩展选项

---

## 📞 支持

如需帮助或有问题，请检查：
1. 确保所有依赖已正确安装
2. 验证输入图像和问题格式正确
3. 检查设备配置（GPU/CPU）
4. 参考演示代码中的示例用法

**注意**: 本实现为演示版本，在生产环境使用前请进行充分测试和优化。