#!/usr/bin/env python3
"""
简化的 QA-ViT + PnP-VQA 热力图演示
不依赖外部库，仅展示核心逻辑
"""

import math
import random

class SimpleAttentionMapDemo:
    """
    简化的注意力图演示类
    模拟 QA-ViT + PnP-VQA 的核心功能
    """
    
    def __init__(self):
        self.grid_size = 14  # 14x14 的注意力网格 (224/16 = 14)
        
    def simulate_qavit_attention(self, question):
        """
        模拟 QA-ViT 生成问题感知的注意力图
        
        Args:
            question: 问题文本
            
        Returns:
            注意力权重列表
        """
        print(f"生成问题感知注意力图: '{question}'")
        
        # 初始化随机注意力
        attention = [random.random() * 0.1 for _ in range(self.grid_size * self.grid_size)]
        
        # 根据问题类型调整注意力模式
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['car', 'vehicle', 'truck', 'bus']):
            # 车辆相关：关注中下部区域
            self._add_attention_region(attention, 7, 12, 3, 10, 0.8)
            print("  检测到车辆相关问题，关注道路区域")
            
        elif any(word in question_lower for word in ['person', 'man', 'woman', 'people']):
            # 人物相关：关注中央区域
            self._add_attention_region(attention, 4, 10, 4, 10, 0.7)
            print("  检测到人物相关问题，关注中央区域")
            
        elif any(word in question_lower for word in ['sky', 'cloud', 'sun', 'weather']):
            # 天空相关：关注上部区域
            self._add_attention_region(attention, 0, 5, 0, 13, 0.9)
            print("  检测到天空相关问题，关注上部区域")
            
        elif any(word in question_lower for word in ['building', 'house', 'window', 'door']):
            # 建筑相关：关注中上部区域
            self._add_attention_region(attention, 2, 8, 2, 11, 0.6)
            print("  检测到建筑相关问题，关注建筑区域")
            
        elif any(word in question_lower for word in ['tree', 'plant', 'flower', 'grass']):
            # 植物相关：关注边缘和中部
            self._add_attention_region(attention, 3, 11, 0, 3, 0.5)  # 左侧
            self._add_attention_region(attention, 3, 11, 10, 13, 0.5)  # 右侧
            print("  检测到植物相关问题，关注边缘区域")
            
        else:
            # 通用问题：关注中央区域
            self._add_attention_region(attention, 5, 9, 5, 9, 0.4)
            print("  通用问题，关注中央区域")
        
        # 标准化注意力权重
        total = sum(attention)
        attention = [w / total for w in attention]
        
        return attention
    
    def _add_attention_region(self, attention, row_start, row_end, col_start, col_end, boost):
        """
        在指定区域增加注意力权重
        """
        for i in range(row_start, min(row_end, self.grid_size)):
            for j in range(col_start, min(col_end, self.grid_size)):
                idx = i * self.grid_size + j
                if idx < len(attention):
                    attention[idx] += boost
    
    def visualize_attention_text(self, attention, question):
        """
        用文本形式可视化注意力图
        """
        print(f"\n问题: '{question}'")
        print("注意力热力图 (█ 表示高注意力, ░ 表示低注意力):")
        print("=" * (self.grid_size * 2 + 2))
        
        # 找到注意力的最大和最小值用于标准化
        max_attention = max(attention)
        min_attention = min(attention)
        attention_range = max_attention - min_attention
        
        for i in range(self.grid_size):
            row = "|"
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                # 标准化到 0-1 范围
                normalized = (attention[idx] - min_attention) / attention_range if attention_range > 0 else 0
                
                # 选择显示字符
                if normalized > 0.8:
                    char = "█"
                elif normalized > 0.6:
                    char = "▓"
                elif normalized > 0.4:
                    char = "▒"
                elif normalized > 0.2:
                    char = "░"
                else:
                    char = " "
                
                row += char + " "
            row += "|"
            print(row)
        
        print("=" * (self.grid_size * 2 + 2))
        
        # 显示统计信息
        print(f"注意力统计:")
        print(f"  最大值: {max_attention:.4f}")
        print(f"  最小值: {min_attention:.4f}")
        print(f"  平均值: {sum(attention) / len(attention):.4f}")
        print(f"  总和: {sum(attention):.4f}")
        
        # 找到前5个最高注意力的位置
        indexed_attention = [(attention[i], i) for i in range(len(attention))]
        indexed_attention.sort(reverse=True)
        
        print(f"  前5个关注区域 (行, 列):")
        for rank, (weight, idx) in enumerate(indexed_attention[:5]):
            row = idx // self.grid_size
            col = idx % self.grid_size
            print(f"    {rank+1}. 位置({row:2d}, {col:2d}): {weight:.4f}")
    
    def analyze_attention_pattern(self, attention, question):
        """
        分析注意力模式
        """
        print(f"\n注意力模式分析:")
        
        # 计算不同区域的注意力
        regions = {
            '上部': (0, 4, 0, 13),      # 天空区域
            '中上': (4, 7, 0, 13),      # 建筑区域
            '中部': (7, 10, 0, 13),     # 主要物体区域
            '下部': (10, 14, 0, 13),    # 地面区域
            '左侧': (0, 14, 0, 4),      # 左边缘
            '中央': (4, 10, 4, 10),     # 中央区域
            '右侧': (0, 14, 10, 14),    # 右边缘
        }
        
        region_attention = {}
        for region_name, (r1, r2, c1, c2) in regions.items():
            total_attention = 0
            pixel_count = 0
            
            for i in range(r1, min(r2, self.grid_size)):
                for j in range(c1, min(c2, self.grid_size)):
                    idx = i * self.grid_size + j
                    if idx < len(attention):
                        total_attention += attention[idx]
                        pixel_count += 1
            
            avg_attention = total_attention / pixel_count if pixel_count > 0 else 0
            region_attention[region_name] = avg_attention
        
        # 排序并显示
        sorted_regions = sorted(region_attention.items(), key=lambda x: x[1], reverse=True)
        
        print("  各区域平均注意力 (从高到低):")
        for region, attention_score in sorted_regions:
            bar_length = int(attention_score * 1000)  # 缩放用于显示
            bar = "█" * bar_length
            print(f"    {region:4s}: {attention_score:.4f} {bar}")
        
        # 分析问题匹配度
        question_lower = question.lower()
        expected_regions = []
        
        if any(word in question_lower for word in ['sky', 'cloud']):
            expected_regions.append('上部')
        if any(word in question_lower for word in ['building', 'house']):
            expected_regions.append('中上')
        if any(word in question_lower for word in ['car', 'person']):
            expected_regions.append('中部')
        if any(word in question_lower for word in ['ground', 'road']):
            expected_regions.append('下部')
        
        if expected_regions:
            print(f"  预期关注区域: {', '.join(expected_regions)}")
            actual_top_regions = [region for region, _ in sorted_regions[:2]]
            match_count = len(set(expected_regions) & set(actual_top_regions))
            print(f"  匹配程度: {match_count}/{len(expected_regions)} 个预期区域在前2位")
    
    def demo_single_question(self, question):
        """
        单个问题的完整演示
        """
        print("=" * 60)
        print(f"QA-ViT + PnP-VQA 注意力演示")
        print("=" * 60)
        
        # 生成注意力图
        attention = self.simulate_qavit_attention(question)
        
        # 可视化
        self.visualize_attention_text(attention, question)
        
        # 分析
        self.analyze_attention_pattern(attention, question)
        
        return attention
    
    def demo_multiple_questions(self, questions):
        """
        多问题对比演示
        """
        print("=" * 60)
        print("多问题注意力对比演示")
        print("=" * 60)
        
        all_results = []
        
        for i, question in enumerate(questions):
            print(f"\n{'='*20} 问题 {i+1} {'='*20}")
            attention = self.simulate_qavit_attention(question)
            all_results.append((question, attention))
        
        # 对比分析
        print(f"\n{'='*20} 对比分析 {'='*20}")
        for question, attention in all_results:
            max_attention = max(attention)
            max_idx = attention.index(max_attention)
            max_row = max_idx // self.grid_size
            max_col = max_idx % self.grid_size
            
            print(f"问题: '{question[:30]}{'...' if len(question) > 30 else ''}'")
            print(f"  最大注意力位置: ({max_row}, {max_col}), 权重: {max_attention:.4f}")
        
        return all_results


def main():
    """
    主演示函数
    """
    print("QA-ViT + PnP-VQA 简化演示")
    print("模拟问题感知的视觉注意力生成")
    print("")
    
    demo = SimpleAttentionMapDemo()
    
    # 单问题演示
    print("【单问题演示】")
    demo.demo_single_question("Where is the car in the image?")
    
    print("\n" + "="*60)
    input("按 Enter 键继续多问题演示...")
    
    # 多问题演示
    print("\n【多问题演示】")
    sample_questions = [
        "What is in the sky?",
        "Where is the person standing?", 
        "What color is the car?",
        "Are there any buildings?",
        "What plants can you see?"
    ]
    
    demo.demo_multiple_questions(sample_questions)
    
    print("\n" + "="*60)
    print("演示完成！")
    print("\n核心特点:")
    print("1. 问题感知：不同类型的问题会关注不同区域")
    print("2. 空间定位：注意力图显示模型关注的空间位置")
    print("3. 可解释性：可以分析模型的注意力模式")
    print("4. 对比分析：多个问题可以进行对比分析")
    print("\n在真实模型中，这些注意力图会用于:")
    print("- 引导图像描述生成")
    print("- 改善视觉问答准确性") 
    print("- 提供模型决策的可解释性")


if __name__ == "__main__":
    main()