import os
import sys
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import networkx as nx
import random
import tempfile
import json
from pycocotools.coco import COCO

class PIDGraphDatasetExtractor:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_info = []
        

    def extract(self):
        # pattern = os.path.join(self.base_dir, '**', '*.graphml')
        # files = set(glob.glob(pattern, recursive=True))
        # for graphml_path in files:
        #     self._process_graphml(graphml_path)
        # print(f"Found {len(files)} GraphML files in {self.base_dir}")

        # with open(os.path.join(self.output_dir, 'dataset_info.json'), 'w') as f:
        #     json.dump(self.data_info, f, indent=4)
        paired_files = []
        for root, dirs, files in os.walk(self.base_dir):
            # 按文件名分组
            file_groups = {}
            
            for file in files:
                base_name = os.path.splitext(file)[0]
                ext = os.path.splitext(file)[1].lower()
                
                if base_name not in file_groups:
                    file_groups[base_name] = {}
                
                if ext in ['.png', '.jpg', '.jpeg']:
                    file_groups[base_name]['image'] = os.path.join(root, file)
                elif ext == '.graphml':
                    file_groups[base_name]['annotation'] = os.path.join(root, file)
            
            # 找到完整配对的文件
            for base_name, files_dict in file_groups.items():
                if 'image' in files_dict and 'annotation' in files_dict:
                    paired_files.append({
                        'id': base_name,
                        'image_path': files_dict['image'],
                        'annotation_path': files_dict['annotation'],
                        'relative_path': os.path.relpath(root, self.base_dir)
                    })

        print(f"在 {self.base_dir} 中找到 {len(paired_files)} 对配对文件")
        return paired_files
    
    def extract_annotations_from_graphml(self, graphml_path):
        """从 GraphML 文件中提取标注信息"""
        try:
            graph = nx.read_graphml(graphml_path)
            
            # 提取节点（对象）信息
            objects = []
            for node_id, node_attrs in graph.nodes(data=True):
                # 检查是否有边界框信息
                bbox_keys = ['xmin', 'ymin', 'xmax', 'ymax']
                if all(key in node_attrs for key in bbox_keys):
                    try:
                        bbox = [float(node_attrs[key]) for key in bbox_keys]
                        
                        obj_info = {
                            'id': node_id,
                            'bbox': bbox,  # [xmin, ymin, xmax, ymax]
                            'label': node_attrs.get('label', 'unknown'),
                            'category': node_attrs.get('category', 'object'),
                            'attributes': {k: v for k, v in node_attrs.items() 
                                         if k not in bbox_keys + ['label', 'category']}
                        }
                        objects.append(obj_info)
                    except (ValueError, TypeError):
                        continue
            
            # 提取边（关系）信息
            relations = []
            for src, dst, edge_attrs in graph.edges(data=True):
                rel_info = {
                    'subject': src,
                    'object': dst,
                    'predicate': edge_attrs.get('edge_label'),
                    'attributes': dict(edge_attrs)
                }
                relations.append(rel_info)
            
            return {
                'objects': objects,
                'relations': relations,
                'num_objects': len(objects),
                'num_relations': len(relations)
            }
            
        except Exception as e:
            print(f"解析 GraphML 文件失败 {graphml_path}: {e}")
            return None


class PIDGraphDataset(Dataset):
    def __init__(self, complete_img=False, min_objects=1, transform=None):
        if complete_img==False:
            self.base_dir = 'PID2Graph/patched'
        else:
            self.base_dir = 'PID2Graph/Complete'
        self.transform = transform
        # 1. 加载关系类别配置
        self.rel_categories = ['__background__',      # 0 - 背景类别
                               'solid',            # 1 - 实线连接
                               'non-solid',       # 2 - 虚线连接
                               ]
        
        # 2. 创建关系类别映射
        self.relation_label_to_idx = {rel: idx for idx, rel in enumerate(self.rel_categories)}
        self.idx_to_relation_label = {idx: rel for idx, rel in enumerate(self.rel_categories)}
        
        # 3. 加载对象类别 (如果需要)
        self.obj_categories = [
            '__background__',     # 0 - 背景
            'pump',              # 1 - 泵
            'valve',             # 2 - 阀门
            'tank',              # 3 - 储罐
            'pipe',              # 4 - 管道
            'sensor',            # 5 - 传感器
            'motor',             # 6 - 电机
            'heat_exchanger',    # 7 - 换热器
            'compressor',        # 8 - 压缩机
            'filter',            # 9 - 过滤器
            'control_valve',     # 10 - 调节阀
            'pressure_vessel',   # 11 - 压力容器
            'instrument',        # 12 - 仪表
        ]
        
        self.label_to_idx = {
            obj: idx for idx, obj in enumerate(self.obj_categories)
        }
        
        self.idx_to_label = {
            idx: obj for idx, obj in enumerate(self.obj_categories)
        }
        # 提取数据集信息
        self.extractor = PIDGraphDatasetExtractor(self.base_dir)
        self.paired_files = self.extractor.extract()
        # 验证并过滤有效样本
        self.valid_samples = []
        print("验证样本...")
        
        for i, pair in enumerate(self.paired_files):
            annotations = self.extractor.extract_annotations_from_graphml(pair['annotation_path'])
            
            if annotations and annotations['num_objects'] >= min_objects:
                pair['annotations'] = annotations
                self.valid_samples.append(pair)
        
        print(f"\n数据集加载完成:")
        print(f"  原始配对: {len(self.paired_files)}")
        print(f"  有效样本: {len(self.valid_samples)}")
        self._create_label_mappings()

    def _create_label_mappings(self):
        """创建标签映射"""
        all_object_labels = set()
        all_relation_labels = set()
        
        for sample in self.valid_samples:
            for obj in sample['annotations']['objects']:
                all_object_labels.add(obj['label'])
            for rel in sample['annotations']['relations']:
                all_relation_labels.add(rel['predicate'])
        
        # 对象标签映射（添加背景类）
        self.object_classes = ['__background__'] + sorted(list(all_object_labels))
        self.object_label_to_idx = {label: idx for idx, label in enumerate(self.object_classes)}
        self.idx_to_object_label = {idx: label for label, idx in self.object_label_to_idx.items()}
        self.num_object_classes = len(self.object_classes)
        
        # 关系标签映射（添加背景类）
        self.relation_classes = ['__background__'] + sorted(list(all_relation_labels))
        self.relation_label_to_idx = {label: idx for idx, label in enumerate(self.relation_classes)}
        self.idx_to_relation_label = {idx: label for label, idx in self.relation_label_to_idx.items()}
        self.num_relation_classes = len(self.relation_classes)
        
        print(f"  对象类别: {self.num_object_classes} 个")
        print(f"  关系类别: {self.num_relation_classes} 个")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        """返回图像、目标标注和样本信息"""
        sample = self.valid_samples[idx]
        
        # 1. 加载图像
        image = Image.open(sample['image_path']).convert('RGB')
        orig_size = torch.tensor(image.size)  # (width, height)
        
        # 2. 处理标注
        annotations = sample['annotations']
        
        # 提取对象信息
        boxes = []
        labels = []
        object_ids = []
        
        for obj in annotations['objects']:
            boxes.append(obj['bbox'])  # [xmin, ymin, xmax, ymax]
            # 获取标签索引（跳过背景类，从1开始）
            label_idx = self.object_label_to_idx.get(obj['label'], 1)  # 默认为第一个真实类别
            labels.append(label_idx)
            object_ids.append(obj['id'])
        
        # 提取关系信息
        relations = []
        relation_labels = []
        rel_annotations = []
        for rel in annotations['relations']:
            try:
                # 找到主体和客体在对象列表中的索引
                sub_idx = object_ids.index(rel['subject'])
                obj_idx = object_ids.index(rel['object'])
                rel_label = self.relation_label_to_idx.get(rel['predicate'], 1)
                if len([sub_idx, obj_idx]) > 1:
                    # relations.append([sub_idx, obj_idx])  # [subject_idx, object_idx]
                    # relation_labels.append(rel_label)
                    rel_annotations.append([sub_idx, obj_idx, rel_label])
                else:
                    print(f"Warning: Relation {rel['predicate']} between {rel['subject']} and {rel['object']} not found in objects.")
                    rel_annotations.append([0, 0, 0])
            except ValueError:
                # 跳过找不到对应对象的关系
                continue
        # rel_annotations = torch.cat([relations, relation_labels.unsqueeze(1)], dim=1)
        
        # 3. 转换为张量
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # 处理没有对象的情况
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        if len(relations) > 0:
            relations = torch.as_tensor(relations, dtype=torch.int64)
            relation_labels = torch.as_tensor(relation_labels, dtype=torch.int64)
        else:
            relations = torch.zeros((0, 2), dtype=torch.int64)
            relation_labels = torch.zeros((0,), dtype=torch.int64)
        
        # 4. 构建目标字典
        target = {
            'boxes': boxes,                    # [N, 4] 边界框
            'labels': labels,                  # [N] 对象标签
            'image_id': torch.tensor([idx+ 1], dtype=torch.int64),   # [1] 图像ID
            'orig_size': orig_size,            # [2] 原始图像尺寸 [H, W] - 新增
            # 'relations': relations,            # [M, 2] 关系对索引
            # 'relation_labels': relation_labels, # [M] 关系标签
            'rel_annotations': torch.tensor(rel_annotations, dtype=torch.int64).view(-1, 3),  # [M, 3] - [subject_idx, object_idx, relation_label]
            'size': orig_size,              # [2] 图像尺寸 [H, W]
            # 'num_objects': len(boxes),         # 对象数量
            # 'num_relations': len(relations),   # 关系数量
            # 'sample_id': sample['id']            # 样本ID字符串
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }
        
        # 5. 应用图像变换
        if self.transform:
            image = self.transform(image)
        
        return image, target
    
    def get_dataset_info(self):
        """获取数据集信息"""
        return {
            'total_samples': len(self.valid_samples),
            'object_classes': self.object_classes,
            'relation_classes': self.relation_classes,
            'num_object_classes': self.num_object_classes,
            'num_relation_classes': self.num_relation_classes
        }
    
    def get_sample_by_id(self, sample_id):
        """根据样本ID获取样本"""
        for sample in self.valid_samples:
            if sample['id'] == sample_id:
                return sample
        return None


class CocoCompatibleDataset(torchvision.datasets.CocoDetection):
    """COCO 兼容的数据集包装类"""
    
    def __init__(self, original_dataset, indices=None):
        # 不调用父类的 __init__，因为我们要自定义行为
        self.original_dataset = original_dataset
        self.indices = indices if indices is not None else list(range(len(original_dataset)))
        
        # 🔑 创建 COCO API 对象
        self.coco = self._create_coco_api()
        
        # 复制原数据集的属性
        self.rel_categories = getattr(original_dataset, 'rel_categories', ['__background__'])
        self.obj_categories = getattr(original_dataset, 'obj_categories', ['__background__'])
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 获取真实的索引
        real_idx = self.indices[idx]
        return self.original_dataset[real_idx]
    
    def _create_coco_api(self):
        """创建 COCO API 对象"""
        try:
            # 构建 COCO 格式数据
            images = []
            annotations = []
            categories = []
            
            # 获取类别信息
            obj_categories = getattr(self.original_dataset, 'obj_categories', ['__background__', 'object'])
            
            # 创建类别（跳过背景类）
            for i, cat_name in enumerate(obj_categories):
                if i == 0:  # 跳过背景类
                    continue
                categories.append({
                    "id": i,
                    "name": cat_name,
                    "supercategory": "thing"
                })
            
            if not categories:
                categories.append({"id": 1, "name": "object", "supercategory": "thing"})
            
            # 处理样本（限制数量避免内存问题）
            num_samples = min(len(self.indices), 50)
            ann_id = 1
            
            for i in range(num_samples):
                try:
                    real_idx = self.indices[i]
                    image, target = self.original_dataset[real_idx]
                    
                    # 图像信息
                    if 'orig_size' in target:
                        h, w = target['orig_size'].tolist()
                    else:
                        h, w = 800, 800
                    
                    image_info = {
                        "id": i + 1,
                        "width": w,
                        "height": h,
                        "file_name": f"image_{real_idx}.jpg"
                    }
                    images.append(image_info)
                    
                    # 标注信息
                    if 'boxes' in target and 'labels' in target:
                        boxes = target['boxes']
                        labels = target['labels']
                        
                        for box, label in zip(boxes, labels):
                            if len(box) == 4 and label.item() > 0:
                                x1, y1, x2, y2 = box.tolist()
                                annotation = {
                                    "id": ann_id,
                                    "image_id": i + 1,
                                    "category_id": int(label.item()),
                                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                                    "area": (x2 - x1) * (y2 - y1),
                                    "iscrowd": 0
                                }
                                annotations.append(annotation)
                                ann_id += 1
                except Exception as e:
                    print(f"处理样本 {i} 时出错: {e}")
                    continue
            
            # 确保至少有一个图像和标注
            if not images:
                images.append({"id": 1, "width": 800, "height": 800, "file_name": "dummy.jpg"})
            
            if not annotations:
                annotations.append({
                    "id": 1, "image_id": 1, "category_id": 1,
                    "bbox": [100, 100, 100, 100], "area": 10000, "iscrowd": 0
                })
            
            # 创建 COCO 数据
            coco_data = {
                "images": images,
                "annotations": annotations,
                "categories": categories,
                "info": {"description": "PID Dataset in COCO format", "version": "1.0"}
            }
            # print('coco数据: ', coco_data.get('images', []))
            
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(coco_data, temp_file, indent=2)
            temp_file.close()
            
            try:
                coco_api = COCO(temp_file.name)
                os.unlink(temp_file.name)
                print(f"✅ 创建 COCO API: {len(images)} 图像, {len(annotations)} 标注")
                return coco_api
            except Exception as e:
                print(f"❌ COCO API 创建失败: {e}")
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                return self._create_minimal_coco_api()
                
        except Exception as e:
            print(f"❌ _create_coco_api 出错: {e}")
            return self._create_minimal_coco_api()
    
    def _create_minimal_coco_api(self):
        """创建最小的 COCO API"""
        dummy_data = {
            "images": [{"id": 1, "width": 800, "height": 800, "file_name": "dummy.jpg"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 100, 100], "area": 10000, "iscrowd": 0}],
            "categories": [{"id": 1, "name": "object", "supercategory": "thing"}]
        }
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(dummy_data, temp_file)
        temp_file.close()
        
        try:
            coco_api = COCO(temp_file.name)
            os.unlink(temp_file.name)
            return coco_api
        except:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            return None

def create_transforms(train=True):
    """创建数据变换"""
    if train:
        return T.Compose([
            T.Resize((800, 800)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((800, 800)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def collate_fn(batch):
    """自定义 collate 函数处理不同数量的对象和关系"""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # 将图像堆叠成批次
    images = torch.stack(images, dim=0)
    
    return images, targets


def create_datasets(complete_img=False, batch_size=4, shuffle=True, num_workers=2, train=True, split_ratio=0.8, seed=42):
    """创建 DataLoader"""
    transform = create_transforms(train=train)
    
    dataset = PIDGraphDataset(
        complete_img=complete_img,
        min_objects=1,
        transform=transform
    )
    # 划分训练集和验证集
    total_size = len(dataset)
    train_size = int(total_size * split_ratio)
    val_size = total_size - train_size
    print(f"\n=== 数据集划分 ===")
    print(f"总样本数: {total_size}")
    print(f"训练集大小: {train_size} ({split_ratio:.1%})")
    print(f"验证集大小: {val_size} ({1-split_ratio:.1%})")
    # 设置随机种子确保可重复性
    random.seed(seed)
    torch.manual_seed(seed)
    
    # 生成随机索引
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 创建子数据集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    # 为训练集和验证集设置不同的变换
    # 🔑 手动给 Subset 添加必要的属性
    train_dataset.rel_categories = dataset.rel_categories
    train_dataset.obj_categories = dataset.obj_categories
    train_dataset.relation_label_to_idx = dataset.relation_label_to_idx
    train_dataset.idx_to_relation_label = dataset.idx_to_relation_label
    train_dataset.label_to_idx = dataset.label_to_idx
    train_dataset.idx_to_label = dataset.idx_to_label
    train_dataset.object_classes = dataset.object_classes
    train_dataset.relation_classes = dataset.relation_classes
    train_dataset.num_object_classes = dataset.num_object_classes
    train_dataset.num_relation_classes = dataset.num_relation_classes
    
    val_dataset.rel_categories = dataset.rel_categories
    val_dataset.obj_categories = dataset.obj_categories
    val_dataset.relation_label_to_idx = dataset.relation_label_to_idx
    val_dataset.idx_to_relation_label = dataset.idx_to_relation_label
    val_dataset.label_to_idx = dataset.label_to_idx
    val_dataset.idx_to_label = dataset.idx_to_label
    val_dataset.object_classes = dataset.object_classes
    val_dataset.relation_classes = dataset.relation_classes
    val_dataset.num_object_classes = dataset.num_object_classes
    val_dataset.num_relation_classes = dataset.num_relation_classes

    train_dataset = CocoCompatibleDataset(dataset, train_indices)
    val_dataset = CocoCompatibleDataset(dataset, val_indices)
    print(f"✅ 创建 COCO 兼容数据集成功")
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    print(f"训练集类型检查: {isinstance(train_dataset, torchvision.datasets.CocoDetection)}")
    print(f"验证集类型检查: {isinstance(val_dataset, torchvision.datasets.CocoDetection)}")
    
    return train_dataset,  val_dataset


# 可视化函数
def visualize_sample(dataset, idx, save_path=None):
    """可视化样本"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # 获取原始数据（不经过变换）
    sample = dataset[idx]
    image_tensor, targets = dataset[idx]
            
    # 🔑 如果是张量，转换回 PIL 图像格式
    if isinstance(image_tensor, torch.Tensor):
        # 反归一化
        if image_tensor.shape[0] == 3:  # (C, H, W)
            # 假设使用了标准的 ImageNet 归一化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    
            # 反归一化
            image_tensor = image_tensor * std + mean
            image_tensor = torch.clamp(image_tensor, 0, 1)
                    
            # 转换为 (H, W, C) 格式
            image_np = image_tensor.permute(1, 2, 0).numpy()
                    
            # 转换为 PIL 图像
            image = Image.fromarray((image_np * 255).astype(np.uint8))
        else:
            print(f"❌ 未知的图像张量形状: {image_tensor.shape}")
            return
    else:
        image = image_tensor
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    # print(targets)
    # 绘制对象边界框
    colors = plt.cm.Set3(np.linspace(0, 1, max(targets['boxes'].shape[0], 1)))
    object_positions = {}

    for i, box in enumerate(targets['boxes']):
        bbox = box  # [xmin, ymin, xmax, ymax]
        print('box: ', box)
        print('bbox: ', bbox)
        # 绘制边界框
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), 
            bbox[2] - bbox[0], 
            bbox[3] - bbox[1],
            linewidth=2, 
            edgecolor=colors[i % len(colors)], 
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加标签
        ax.text(
            bbox[0], bbox[1] - 10, 
            f"{box['label']} ({box['id']})",
            fontsize=10, 
            color=colors[i % len(colors)],
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)
        )
        
        # 记录对象中心位置用于绘制关系
        object_positions[i] = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    # 绘制关系
    for rel in targets['relations']:
        if rel['subject'] in object_positions and rel['object'] in object_positions:
            sub_pos = object_positions[rel['subject']]
            obj_pos = object_positions[rel['object']]
            
            # 绘制箭头
            ax.annotate('', xy=obj_pos, xytext=sub_pos,
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
            
            # 添加关系标签
            mid_x = (sub_pos[0] + obj_pos[0]) / 2
            mid_y = (sub_pos[1] + obj_pos[1]) / 2
            ax.text(mid_x, mid_y, rel['predicate'], 
                   fontsize=8, color='red', 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
    
    ax.set_title(f"样本 {idx}: {sample['id']}\n对象: {annotations['num_objects']}, 关系: {annotations['num_relations']}")
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # 上级目录 (RelTR-main)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # 创建数据集和数据加载器
    dataset_train, dataset_val = create_datasets(
        complete_img=False,  # 使用 patched 数据
        batch_size=2,
        shuffle=True,
        train=True
    )
    from torch.utils.data import DataLoader
    import util.misc as utils
    def simple_collate_fn(batch):
        """简化的 collate 函数"""
        images = []
        targets = []
        
        for item in batch:
            if item is not None and len(item) == 2:
                image, target = item
                images.append(image)
                targets.append(target)
        
        if len(images) == 0:
            return None
        
        # 将图像堆叠成批次
        images = torch.stack(images, dim=0)
        
        return images, targets
    
    # 🔑 使用简单的 DataLoader，不使用分布式
    data_loader_train = DataLoader(
        dataset_train, 
        batch_size=2,
        shuffle=True,
        collate_fn=simple_collate_fn, 
        num_workers=0,  # 设置为 0 避免多进程问题
        drop_last=True
    )
    
    # 测试数据加载器
    print(f"\n=== 测试 DataLoader ===")
    for batch_idx, (images, targets) in enumerate(data_loader_train):
        print(f"Batch {batch_idx + 1}:")
        print(f"  图像形状: {images.shape}")
        print(f"  批次大小: {len(targets)}")
        
        for i, target in enumerate(targets):
            print(f"    样本 {i}: ID={target['image_id']}")
            print(f"      边界框形状: {target['boxes'].shape}")
            print(f"      标签: {target['labels'].tolist()}")
            print(f"      关系: {target['rel_annotations'].tolist()}")

        
        if batch_idx >= 1:  # 只显示前2个batch
            break
    
    # 可视化样本
    print(f"\n=== 可视化样本 ===")
    for i in range(min(2, len(dataset_train))):
        visualize_sample(dataset_train, i)