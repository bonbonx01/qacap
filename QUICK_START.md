# QA-ViT + PNP-VQA 快速开始指南

## 🚀 快速开始 (5分钟设置)

### 步骤 1: 安装依赖
```bash
python3 install_dependencies.py
```

### 步骤 2: 验证安装
```bash
python3 test_setup.py
```
如果看到 "✓ All tests passed!"，继续下一步。

### 步骤 3: 运行演示
```bash
# 运行默认示例
python3 run_demo.py

# 或使用自定义图像
python3 run_demo.py --custom --image your_image.jpg --question "What is in the image?"
```

## 📝 原始代码问题总结

您的 `qacap_vqa.py` 实现主要存在以下问题：

| 问题类型 | 原因 | 解决方案 |
|---------|------|---------|
| **from_config方法** | 缺少 qavit_model_config 和 text_encoder 参数 | ✅ 添加完整的参数处理逻辑 |
| **导入错误** | clip_vision_tower 路径不正确 | ✅ 修正为正确的 qavit_encoder 导入 |
| **设备处理** | 直接使用不存在的 self.device | ✅ 添加 @property device 方法 |
| **模型引用** | 引用已删除的 image_question_matching_model | ✅ 清理所有无效引用 |
| **边界检查** | 缺少注意力层数检查 | ✅ 添加完整的边界和错误处理 |

## 🎯 主要改进

### 1. 架构优化
- **问题感知注意力**: QA-ViT 根据问题内容生成相关的视觉注意力
- **动态融合**: 在多个层级融合视觉和文本信息
- **兼容性**: 保持与原 PNP-VQA 的完全兼容

### 2. 功能增强
- **三种可视化**: 原图、热力图、叠加图
- **统计分析**: 详细的注意力分布统计
- **批处理支持**: 可扩展支持多图像处理

### 3. 用户体验
- **一键安装**: 自动化依赖管理
- **完整测试**: 环境验证和功能测试
- **详细文档**: 完整的使用说明和API文档

## 📊 预期输出

运行demo后，您将看到：

```
============================================================
QA-ViT + PNP-VQA Heat Map Generation Demo
============================================================
Generating heat map for question: 'What colors are in the image?'
Model setup completed successfully!

Attention Statistics:
- Max attention: 0.0856
- Min attention: 0.0021  
- Mean attention: 0.0041
- Std attention: 0.0098

Heat map saved to: ./demo_outputs/heatmap_sample.png
```

以及包含三个子图的可视化图像：
1. 原始图像
2. QA-ViT 注意力热力图  
3. 注意力叠加图像

## 🔧 自定义使用

### 编程式API
```python
from demo_qavit_heatmap import QAViTHeatMapDemo

# 初始化
demo = QAViTHeatMapDemo(device='cuda')

# 生成热力图
fig, attention_map = demo.run_demo(
    image_path="path/to/image.jpg",
    question="What objects are visible?",
    output_dir="./my_outputs"
)
```

### 配置修改
编辑 `/workspace/configs/models/qacap/pnp_vqa_qavit.yaml`:
```yaml
qavit_config:
  vit_model: "openai/clip-vit-large-patch14"  # 更大模型
  integration_point: "late"  # 改变融合策略
```

## 🐛 故障排除

### 常见问题
1. **内存不足**: 使用 `--device cpu`
2. **依赖缺失**: 重新运行 `python3 install_dependencies.py`
3. **模型下载慢**: 检查网络连接，模型会自动缓存

### 获取帮助
```bash
python3 demo_qavit_heatmap.py --help
python3 run_demo.py --help
```

## 📈 后续扩展

- **批量处理**: 扩展支持多图像批处理
- **注意力分析**: 添加层级注意力比较
- **实时推理**: 集成webcam实时处理
- **模型微调**: 支持特定领域的模型微调

---

**Ready to go! 🎉** 您现在拥有一个完整的、可工作的 QA-ViT + PNP-VQA 热力图生成系统!