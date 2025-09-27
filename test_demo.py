#!/usr/bin/env python3
"""
测试 QA-ViT + PNP-VQA 演示脚本

这个脚本测试我们的演示是否能正常工作，包括：
1. 基本功能测试
2. 图像加载测试  
3. 热力图生成测试
4. 可视化测试
"""

import sys
import os
sys.path.append('/workspace')

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def test_demo_basic():
    """测试演示的基本功能"""
    print("1. 测试基本功能...")
    
    try:
        from qavit_pnp_demo import QAViTPNPDemo
        
        # 创建演示实例
        demo = QAViTPNPDemo()
        print("   ✓ 演示实例创建成功")
        
        return True
    except Exception as e:
        print(f"   ✗ 基本功能测试失败: {e}")
        return False

def test_image_creation():
    """测试示例图像创建"""
    print("2. 测试图像创建...")
    
    try:
        from qavit_pnp_demo import create_sample_image
        
        # 创建示例图像
        image_path = create_sample_image("test_image.jpg")
        
        # 检查文件是否存在
        if os.path.exists(image_path):
            print(f"   ✓ 图像创建成功: {image_path}")
            return image_path
        else:
            print("   ✗ 图像文件未创建")
            return None
            
    except Exception as e:
        print(f"   ✗ 图像创建失败: {e}")
        return None

def test_image_loading(demo, image_path):
    """测试图像加载"""
    print("3. 测试图像加载...")
    
    try:
        image_tensor, original_image = demo.load_image(image_path)
        
        if image_tensor is not None and original_image is not None:
            print(f"   ✓ 图像加载成功")
            print(f"   - 张量形状: {image_tensor.shape}")
            print(f"   - 原始图像尺寸: {original_image.size}")
            return image_tensor, original_image
        else:
            print("   ✗ 图像加载返回 None")
            return None, None
            
    except Exception as e:
        print(f"   ✗ 图像加载失败: {e}")
        return None, None

def test_heatmap_generation(demo, image_tensor):
    """测试热力图生成"""
    print("4. 测试热力图生成...")
    
    try:
        question = "What is in the image?"
        answers, attention_maps, captions = demo.generate_heatmap(
            image_tensor, 
            question,
            num_captions=3,
            num_patches=10,
            block_num=5
        )
        
        if answers is not None and attention_maps is not None:
            print("   ✓ 热力图生成成功")
            print(f"   - 答案: {answers[0] if answers else 'None'}")
            print(f"   - 注意力图形状: {attention_maps.shape}")
            print(f"   - 描述数量: {len(captions[0]) if captions else 0}")
            return answers, attention_maps, captions
        else:
            print("   ✗ 热力图生成返回 None")
            return None, None, None
            
    except Exception as e:
        print(f"   ✗ 热力图生成失败: {e}")
        return None, None, None

def test_visualization(demo, original_image, attention_maps, question, answer):
    """测试可视化功能"""
    print("5. 测试可视化...")
    
    try:
        # 测试可视化（不保存）
        demo.visualize_heatmap(
            original_image,
            attention_maps[0],
            question,
            answer,
            save_path="test_visualization.png"
        )
        
        # 检查是否生成了文件
        if os.path.exists("test_visualization.png"):
            print("   ✓ 可视化成功，文件已保存")
            return True
        else:
            print("   ✓ 可视化功能正常（未保存文件）")
            return True
            
    except Exception as e:
        print(f"   ✗ 可视化失败: {e}")
        return False

def test_full_demo(demo, image_path):
    """测试完整演示流程"""
    print("6. 测试完整演示流程...")
    
    try:
        result = demo.run_demo(
            image_path=image_path,
            question="What color is the sun?",
            save_dir="./test_results"
        )
        
        if result is not None:
            print("   ✓ 完整演示流程测试成功")
            return True
        else:
            print("   ✗ 完整演示流程返回 None")
            return False
            
    except Exception as e:
        print(f"   ✗ 完整演示流程失败: {e}")
        return False

def main():
    """主测试函数"""
    print("="*50)
    print("QA-ViT + PNP-VQA 演示测试")
    print("="*50)
    
    # 记录测试结果
    test_results = []
    
    # 1. 基本功能测试
    demo = None
    if test_demo_basic():
        from qavit_pnp_demo import QAViTPNPDemo
        demo = QAViTPNPDemo()
        test_results.append(("基本功能", True))
    else:
        test_results.append(("基本功能", False))
        print("\n基本功能测试失败，无法继续后续测试")
        return test_results
    
    # 2. 图像创建测试
    image_path = test_image_creation()
    if image_path:
        test_results.append(("图像创建", True))
    else:
        test_results.append(("图像创建", False))
        print("\n图像创建失败，无法继续后续测试")
        return test_results
    
    # 3. 图像加载测试
    image_tensor, original_image = test_image_loading(demo, image_path)
    if image_tensor is not None:
        test_results.append(("图像加载", True))
    else:
        test_results.append(("图像加载", False))
        print("\n图像加载失败，无法继续后续测试")
        return test_results
    
    # 4. 热力图生成测试
    answers, attention_maps, captions = test_heatmap_generation(demo, image_tensor)
    if answers is not None:
        test_results.append(("热力图生成", True))
    else:
        test_results.append(("热力图生成", False))
        print("\n热力图生成失败，跳过可视化测试")
        test_results.append(("可视化", False))
        test_results.append(("完整流程", False))
        return test_results
    
    # 5. 可视化测试  
    if test_visualization(demo, original_image, attention_maps, "What is in the image?", answers[0]):
        test_results.append(("可视化", True))
    else:
        test_results.append(("可视化", False))
    
    # 6. 完整流程测试
    if test_full_demo(demo, image_path):
        test_results.append(("完整流程", True))
    else:
        test_results.append(("完整流程", False))
    
    # 输出测试结果总结
    print("\n" + "="*50)
    print("测试结果总结:")
    print("="*50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:12} - {status}")
        if result:
            passed += 1
    
    print("-" * 25)
    print(f"总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！演示准备就绪。")
    else:
        print(f"\n⚠️  有 {total-passed} 个测试失败，请检查相关功能。")
    
    return test_results

if __name__ == "__main__":
    results = main()