#!/usr/bin/env python3
"""
QA-ViT + PNP-VQA 概念演示 (无外部依赖)

这个演示脚本不依赖任何外部库，纯粹用于展示：
1. QA-ViT + PNP-VQA 的核心概念
2. 注意力机制的工作原理
3. 模型架构和数据流程
"""

import random
import math

def simulate_attention_map(question, image_regions):
    """
    模拟 QA-ViT 根据问题生成的注意力图
    
    Args:
        question: 问题文本
        image_regions: 图像区域描述列表
        
    Returns:
        attention_weights: 各区域的注意力权重
    """
    print(f"🧠 QA-ViT 正在分析问题: '{question}'")
    
    # 简单的关键词匹配来模拟问题感知
    attention_weights = []
    question_lower = question.lower()
    
    for region in image_regions:
        region_lower = region.lower()
        # 计算相关性分数
        relevance = 0.1  # 基础注意力
        
        # 检查关键词匹配
        question_words = question_lower.split()
        region_words = region_lower.split()
        
        for q_word in question_words:
            for r_word in region_words:
                if q_word == r_word or q_word in r_word or r_word in q_word:
                    relevance += 0.3
        
        # 添加一些随机性来模拟复杂的语义理解
        relevance += random.uniform(0, 0.2)
        attention_weights.append(relevance)
    
    # 归一化注意力权重（softmax）
    max_weight = max(attention_weights)
    exp_weights = [math.exp(w - max_weight) for w in attention_weights]
    sum_exp = sum(exp_weights)
    normalized_weights = [w / sum_exp for w in exp_weights]
    
    return normalized_weights

def generate_answer_and_captions(question, image_regions, attention_weights):
    """
    基于注意力权重生成答案和描述
    
    Args:
        question: 问题文本
        image_regions: 图像区域描述
        attention_weights: 注意力权重
        
    Returns:
        answer: 预测答案
        captions: 生成的描述列表
    """
    print("💬 PNP-VQA 正在生成答案和描述...")
    
    # 找到最受关注的区域
    max_attention_idx = attention_weights.index(max(attention_weights))
    focused_region = image_regions[max_attention_idx]
    
    # 基于问题类型和焦点区域生成答案
    question_lower = question.lower()
    
    if "what color" in question_lower:
        if "sun" in focused_region.lower():
            answer = "yellow"
        elif "sky" in focused_region.lower():
            answer = "blue"
        elif "grass" in focused_region.lower():
            answer = "green"
        else:
            answer = "multiple colors"
    elif "where" in question_lower:
        answer = f"in the {focused_region.lower()}"
    elif "how many" in question_lower:
        answer = "one" if "sun" in question_lower else "several"
    else:
        answer = f"related to {focused_region.lower()}"
    
    # 生成基于注意力的描述
    captions = []
    for i, (region, weight) in enumerate(zip(image_regions, attention_weights)):
        if weight > 0.15:  # 只为高注意力区域生成描述
            captions.append(f"A {region.lower()} with attention weight {weight:.2f}")
    
    return answer, captions

def visualize_attention_text(image_regions, attention_weights):
    """
    以文本形式可视化注意力分布
    
    Args:
        image_regions: 图像区域描述
        attention_weights: 注意力权重
    """
    print("\n📊 注意力分布可视化:")
    print("-" * 50)
    
    max_width = 30
    for region, weight in zip(image_regions, attention_weights):
        bar_length = int(weight * max_width)
        bar = "█" * bar_length + "░" * (max_width - bar_length)
        print(f"{region:15} │{bar}│ {weight:.3f}")
    
    print("-" * 50)

def demonstrate_qavit_pnp():
    """主要的演示函数"""
    print("=" * 60)
    print("🚀 QA-ViT + PNP-VQA 概念演示")
    print("=" * 60)
    
    # 定义一个示例场景
    print("\n🖼️  模拟图像场景:")
    image_regions = [
        "bright yellow sun",
        "blue sky with clouds",
        "green grass field", 
        "brown wooden house",
        "red triangular roof",
        "tall green tree",
        "white fluffy clouds",
        "stone pathway"
    ]
    
    for i, region in enumerate(image_regions):
        print(f"   区域 {i+1}: {region}")
    
    # 定义示例问题
    questions = [
        "What color is the sun?",
        "Where is the house located?", 
        "How many trees are there?",
        "What is in the sky?"
    ]
    
    print(f"\n🔍 准备回答 {len(questions)} 个问题...")
    
    # 对每个问题进行演示
    for i, question in enumerate(questions):
        print(f"\n{'=' * 20} 问题 {i+1} {'=' * 20}")
        print(f"❓ 问题: {question}")
        
        # 1. 生成注意力图
        attention_weights = simulate_attention_map(question, image_regions)
        
        # 2. 可视化注意力
        visualize_attention_text(image_regions, attention_weights)
        
        # 3. 生成答案和描述  
        answer, captions = generate_answer_and_captions(
            question, image_regions, attention_weights
        )
        
        print(f"\n✅ 预测答案: {answer}")
        print(f"📝 生成的描述:")
        for j, caption in enumerate(captions):
            print(f"   {j+1}. {caption}")
        
        # 显示关键统计信息
        max_attention = max(attention_weights)
        focused_region_idx = attention_weights.index(max_attention)
        
        print(f"\n📈 关键统计:")
        print(f"   最关注区域: {image_regions[focused_region_idx]}")
        print(f"   最大注意力值: {max_attention:.3f}")
        print(f"   平均注意力值: {sum(attention_weights)/len(attention_weights):.3f}")
        
        if i < len(questions) - 1:
            print(f"\n{'⏳ 准备下一个问题...'}")
    
    # 总结演示
    print(f"\n{'=' * 60}")
    print("🎯 演示总结")
    print("=" * 60)
    print("✨ 核心特性展示:")
    print("  • 问题感知注意力: 不同问题产生不同的注意力分布")
    print("  • 多模态融合: 结合视觉区域信息和文本问题")
    print("  • 动态焦点: 根据问题内容自动关注相关区域")
    print("  • 描述生成: 基于注意力权重生成相关描述")
    
    print(f"\n🔧 技术实现要点:")
    print("  • QA-ViT 替换传统 GradCAM 提供更精准的注意力")
    print("  • 注意力权重指导后续的描述生成和问答")
    print("  • 支持多种问题类型和复杂场景理解")
    print("  • 提供可视化和可解释的推理过程")
    
    print(f"\n📁 相关文件:")
    print("  • qacap_vqa_fixed.py - 修复后的完整实现")
    print("  • qavit_pnp_demo.py - 完整功能演示脚本") 
    print("  • demo_config.yaml - 配置文件")
    print("  • README_QAViT_PNP.md - 详细说明文档")
    
    print(f"\n🎉 演示完成！")
    print("在实际使用中，这些模拟的组件会被真实的神经网络模型替换。")

if __name__ == "__main__":
    # 设置随机种子以获得可重现的结果
    random.seed(42)
    demonstrate_qavit_pnp()