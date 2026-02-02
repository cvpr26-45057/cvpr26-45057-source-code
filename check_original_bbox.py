#!/usr/bin/env python3
"""
检查原始边界框数据
"""

import os
import networkx as nx
from PIL import Image

def check_original_data():
    """检查原始数据中的边界框"""
    print("="*60)
    print("检查原始边界框数据")
    print("="*60)
    
    # 检查调整分辨率后的数据
    data_path = "/mnt/ShareDB_6TB/baitianyou/RelTR-main/data/pid_resized"
    
    # 检查第一个样本
    sample_graphml = os.path.join(data_path, "0.graphml")
    sample_image = os.path.join(data_path, "0.png")
    
    if os.path.exists(sample_graphml) and os.path.exists(sample_image):
        # 加载图像获取尺寸
        img = Image.open(sample_image)
        img_w, img_h = img.size
        print(f"图像尺寸: {img_w} x {img_h}")
        
        # 加载GraphML
        graph = nx.read_graphml(sample_graphml)
        
        print(f"节点数量: {len(graph.nodes())}")
        
        # 检查前5个节点的边界框
        print("\n原始边界框坐标:")
        count = 0
        for node_id, node_attrs in graph.nodes(data=True):
            bbox_keys = ['xmin', 'ymin', 'xmax', 'ymax']
            if all(key in node_attrs for key in bbox_keys):
                bbox = [float(node_attrs[key]) for key in bbox_keys]
                xmin, ymin, xmax, ymax = bbox
                
                print(f"  节点 {node_id}: [{xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f}]")
                print(f"    相对坐标: [{xmin/img_w:.6f}, {ymin/img_h:.6f}, {xmax/img_w:.6f}, {ymax/img_h:.6f}]")
                
                # 检查是否超出图像边界
                if xmin < 0 or ymin < 0 or xmax > img_w or ymax > img_h:
                    print(f"    ⚠️  坐标超出图像边界!")
                
                count += 1
                if count >= 5:
                    break
        
        # 统计所有边界框
        all_coords = []
        for node_id, node_attrs in graph.nodes(data=True):
            bbox_keys = ['xmin', 'ymin', 'xmax', 'ymax']
            if all(key in node_attrs for key in bbox_keys):
                bbox = [float(node_attrs[key]) for key in bbox_keys]
                all_coords.extend(bbox)
        
        if all_coords:
            print(f"\n所有坐标统计:")
            print(f"  范围: [{min(all_coords):.2f}, {max(all_coords):.2f}]")
            print(f"  相对范围: [{min(all_coords)/img_w:.6f}, {max(all_coords)/img_w:.6f}]")
            
            # 检查超出边界的坐标
            out_of_bounds = [c for c in all_coords if c < 0 or c > max(img_w, img_h)]
            if out_of_bounds:
                print(f"  ⚠️  {len(out_of_bounds)} 个坐标超出边界")
                print(f"  超出边界的坐标: {out_of_bounds[:10]}...")  # 只显示前10个
    else:
        print("❌ 找不到样本文件")

if __name__ == "__main__":
    check_original_data()
