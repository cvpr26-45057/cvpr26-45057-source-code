#!/usr/bin/env python3
"""
PID数据集图像分辨率调整脚本
将高分辨率图像调整到1024像素左右，同时调整GraphML标注中的坐标
"""

import os
import sys
import shutil
from PIL import Image
import networkx as nx
import numpy as np
from tqdm import tqdm

def resize_image_and_annotations(source_dir, target_dir, target_size=1024):
    """
    调整图像大小并同步调整标注坐标
    
    Args:
        source_dir: 原始数据目录
        target_dir: 输出目录
        target_size: 目标图像的长边尺寸
    """
    
    if not os.path.exists(source_dir):
        print(f"❌ 源目录不存在: {source_dir}")
        return False
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有图像文件
    files = os.listdir(source_dir)
    image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"📊 处理统计信息:")
    print(f"   - 源目录: {source_dir}")
    print(f"   - 目标目录: {target_dir}")
    print(f"   - 图像数量: {len(image_files)}")
    print(f"   - 目标尺寸: {target_size}px (长边)")
    
    processed_count = 0
    error_count = 0
    
    for image_file in tqdm(image_files, desc="处理图像"):
        try:
            base_name = os.path.splitext(image_file)[0]
            image_path = os.path.join(source_dir, image_file)
            graphml_path = os.path.join(source_dir, f"{base_name}.graphml")
            
            # 检查对应的GraphML文件是否存在
            if not os.path.exists(graphml_path):
                print(f"⚠️ 跳过 {image_file}: 缺少对应的GraphML文件")
                continue
            
            # 1. 加载和调整图像
            image = Image.open(image_path)
            original_width, original_height = image.size
            
            # 计算缩放比例（保持宽高比）
            scale_factor = min(target_size / original_width, target_size / original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            # 调整图像大小
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 保存调整后的图像
            target_image_path = os.path.join(target_dir, image_file)
            resized_image.save(target_image_path, quality=95)
            
            # 2. 调整GraphML标注坐标
            graph = nx.read_graphml(graphml_path)
            
            # 调整节点坐标
            for node_id, node_attrs in graph.nodes(data=True):
                # 检查并调整边界框坐标
                bbox_keys = ['xmin', 'ymin', 'xmax', 'ymax']
                if all(key in node_attrs for key in bbox_keys):
                    for key in bbox_keys:
                        original_coord = float(node_attrs[key])
                        if key.startswith('x'):  # x坐标
                            new_coord = original_coord * scale_factor
                        else:  # y坐标
                            new_coord = original_coord * scale_factor
                        graph.nodes[node_id][key] = str(new_coord)
            
            # 保存调整后的GraphML文件
            target_graphml_path = os.path.join(target_dir, f"{base_name}.graphml")
            nx.write_graphml(graph, target_graphml_path)
            
            processed_count += 1
            
        except Exception as e:
            print(f"❌ 处理 {image_file} 时出错: {e}")
            error_count += 1
            continue
    
    print(f"\n✅ 处理完成!")
    print(f"   - 成功处理: {processed_count} 个文件")
    print(f"   - 处理失败: {error_count} 个文件")
    
    # 验证一个样本
    if processed_count > 0:
        verify_sample(target_dir, target_size)
    
    return processed_count > 0

def verify_sample(target_dir, target_size):
    """验证调整后的样本"""
    print(f"\n🔍 验证调整结果...")
    
    files = os.listdir(target_dir)
    image_files = [f for f in files if f.endswith('.png')]
    
    if len(image_files) > 0:
        # 检查第一个图像
        sample_file = image_files[0]
        sample_path = os.path.join(target_dir, sample_file)
        sample_image = Image.open(sample_path)
        
        print(f"   - 样本文件: {sample_file}")
        print(f"   - 调整后尺寸: {sample_image.size}")
        print(f"   - 压缩比例: {sample_image.size[0]/7168:.2f}x")
        
        # 检查GraphML文件
        base_name = os.path.splitext(sample_file)[0]
        graphml_path = os.path.join(target_dir, f"{base_name}.graphml")
        
        if os.path.exists(graphml_path):
            graph = nx.read_graphml(graphml_path)
            node_count = len(graph.nodes())
            edge_count = len(graph.edges())
            print(f"   - 标注节点数: {node_count}")
            print(f"   - 标注边数: {edge_count}")
            
            # 检查坐标范围
            all_coords = []
            for node_id, node_attrs in graph.nodes(data=True):
                bbox_keys = ['xmin', 'ymin', 'xmax', 'ymax']
                if all(key in node_attrs for key in bbox_keys):
                    coords = [float(node_attrs[key]) for key in bbox_keys]
                    all_coords.extend(coords)
            
            if all_coords:
                print(f"   - 坐标范围: {min(all_coords):.1f} ~ {max(all_coords):.1f}")

def main():
    """主函数"""
    print("="*60)
    print("PID数据集图像分辨率调整工具")
    print("="*60)
    
    # 配置路径
    source_dir = "/mnt/ShareDB_6TB/baitianyou/RelTR-main/data/complete/Dataset PID"
    target_dir = "/mnt/ShareDB_6TB/baitianyou/RelTR-main/data/pid_resized"
    target_size = 1024  # 长边目标尺寸
    
    # 检查源目录
    if not os.path.exists(source_dir):
        print(f"❌ 源目录不存在: {source_dir}")
        return
    
    # 确认操作
    print(f"即将调整图像分辨率:")
    print(f"  源目录: {source_dir}")
    print(f"  目标目录: {target_dir}")
    print(f"  原始尺寸: 7168x4561")
    print(f"  目标尺寸: ~{target_size}x{int(target_size*4561/7168)}")
    print(f"  压缩比例: ~{target_size/7168:.2f}x")
    
    response = input("\n继续处理? (y/n): ")
    if response.lower() != 'y':
        print("操作已取消")
        return
    
    # 开始处理
    success = resize_image_and_annotations(source_dir, target_dir, target_size)
    
    if success:
        print(f"\n🎉 图像调整完成! 调整后的数据保存在: {target_dir}")
        print(f"现在可以使用调整后的数据进行训练了。")
    else:
        print(f"\n❌ 图像调整失败")

if __name__ == "__main__":
    main()
