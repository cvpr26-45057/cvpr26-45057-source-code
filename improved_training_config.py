#!/usr/bin/env python3
"""
改进的训练配置 - 解决收敛慢的问题
基于收敛分析结果的优化方案
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, ReduceLROnPlateau

def get_improved_optimizer_and_scheduler(model, args):
    """
    获取改进的优化器和学习率调度器
    """
    
    # 1. 分层学习率设置
    backbone_params = []
    transformer_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        elif 'transformer' in name or 'encoder' in name or 'decoder' in name:
            transformer_params.append(param)
        else:
            head_params.append(param)
    
    # 2. 改进的优化器配置
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},  # backbone用更小学习率
        {'params': transformer_params, 'lr': args.lr},      # transformer正常学习率
        {'params': head_params, 'lr': args.lr * 2.0}       # 检测头用更大学习率
    ], 
    lr=args.lr, 
    weight_decay=args.weight_decay,
    betas=(0.9, 0.999),
    eps=1e-8
    )
    
    # 3. 改进的学习率调度策略
    if hasattr(args, 'lr_scheduler') and args.lr_scheduler == 'cosine':
        # 余弦退火调度
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs,
            eta_min=args.lr * 0.01  # 最小学习率为初始的1%
        )
    elif hasattr(args, 'lr_scheduler') and args.lr_scheduler == 'plateau':
        # 自适应调度（推荐）
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,        # 学习率减半
            patience=5,        # 5个epoch没改善就降低
            min_lr=args.lr * 0.001,  # 最小学习率
            verbose=True
        )
    else:
        # 多步调度（原方案改进）
        milestones = [args.lr_drop, args.lr_drop + 10, args.lr_drop + 20]
        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.5  # 每次降低50%
        )
    
    return optimizer, scheduler

def get_improved_training_config():
    """
    获取改进的训练配置
    """
    config = {
        # 学习率策略
        'initial_lr': 5e-5,  # 降低初始学习率
        'lr_scheduler': 'plateau',  # 使用自适应调度
        'warmup_epochs': 3,  # 添加warmup
        
        # 数据增强
        'enhanced_augmentation': True,
        'mixup_alpha': 0.2,  # 添加mixup
        'cutmix_alpha': 0.2,  # 添加cutmix
        
        # 正则化
        'weight_decay': 1e-4,
        'dropout': 0.1,
        'label_smoothing': 0.1,
        
        # 梯度优化
        'gradient_accumulation_steps': 4,  # 累积4步相当于batch_size=8
        'max_grad_norm': 0.1,  # 梯度裁剪
        
        # 早停策略
        'early_stopping_patience': 15,
        'save_best_only': True,
        
        # 查询数量动态调整
        'dynamic_queries': True,
        'start_queries': (400, 800),  # 开始时用较少查询
        'end_queries': (800, 1600),   # 后期增加到完整查询
        'query_ramp_epochs': 20,
    }
    
    return config

def apply_warmup_schedule(optimizer, epoch, warmup_epochs, base_lr):
    """
    应用学习率warmup
    """
    if epoch < warmup_epochs:
        warmup_factor = (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr * warmup_factor

def get_dynamic_queries(epoch, config):
    """
    动态调整查询数量
    """
    if not config.get('dynamic_queries', False):
        return config.get('end_queries', (800, 1600))
    
    start_obj, start_rel = config['start_queries']
    end_obj, end_rel = config['end_queries']
    ramp_epochs = config['query_ramp_epochs']
    
    if epoch < ramp_epochs:
        progress = epoch / ramp_epochs
        current_obj = int(start_obj + (end_obj - start_obj) * progress)
        current_rel = int(start_rel + (end_rel - start_rel) * progress)
    else:
        current_obj, current_rel = end_obj, end_rel
    
    return current_obj, current_rel

if __name__ == "__main__":
    print("🚀 改进的训练配置")
    print("="*50)
    
    config = get_improved_training_config()
    
    print("📋 主要改进点:")
    print(f"1. 降低初始学习率: {config['initial_lr']}")
    print(f"2. 使用自适应调度: {config['lr_scheduler']}")
    print(f"3. 梯度累积步数: {config['gradient_accumulation_steps']}")
    print(f"4. 早停耐心值: {config['early_stopping_patience']}")
    print(f"5. 动态查询调整: {config['dynamic_queries']}")
    
    print("\n🎯 预期效果:")
    print("- 更稳定的收敛过程")
    print("- 避免后期震荡")
    print("- 更好的泛化性能")
    print("- 自动学习率调整")
