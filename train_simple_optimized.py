#!/usr/bin/env python3
"""
简单优化的训练脚本 - 解决收敛慢的问题
基于原始train_resized.py，添加关键优化
"""

import argparse
import datetime
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('简单优化训练', add_help=False)
    
    # 优化的学习率参数
    parser.add_argument('--lr', default=5e-5, type=float, help='降低的初始学习率')
    parser.add_argument('--lr_backbone', default=5e-6, type=float, help='backbone更小学习率')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    
    # 优化参数
    parser.add_argument('--use_plateau_scheduler', action='store_true', 
                        help='使用plateau调度器')
    parser.add_argument('--patience', default=5, type=int,
                        help='plateau调度器的耐心值')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--return_interm_layers', action='store_true')

    # Transformer
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=800, type=int)
    parser.add_argument('--num_rel_queries', default=1600, type=int)
    parser.add_argument('--num_entities', default=800, type=int)
    parser.add_argument('--num_triplets', default=1600, type=int)
    parser.add_argument('--pre_norm', action='store_true')

    # Matcher parameters
    parser.add_argument('--set_iou_threshold', default=0.7, type=float)

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='pid')
    parser.add_argument('--pid_path', default='data/pid', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='simple_optimized_output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    
    return parser

def main(args):
    utils.init_distributed_mode(args)
    print("🚀 启动简单优化训练...")
    print(args)

    device = torch.device(args.device)

    # 设置随机种子
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('参数数量:', n_parameters)

    # 优化的优化器设置
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    # 优化的学习率调度器
    if args.use_plateau_scheduler:
        print("📉 使用ReduceLROnPlateau调度器")
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=args.patience, 
            min_lr=args.lr * 0.001, verbose=True)
    else:
        print("📉 使用StepLR调度器")
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats = evaluate(model, criterion, postprocessors,
                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(test_stats, output_dir / "eval.pth")
        return

    print("🚀 开始训练...")
    print(f"📋 配置: LR={args.lr}, 调度器={'plateau' if args.use_plateau_scheduler else 'step'}")
    
    training_log = []
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 15  # 早停的耐心值
    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
            
        val_stats = evaluate(model, criterion, postprocessors,
                            data_loader_val, base_ds, device, args.output_dir)
        
        # 学习率调度
        if args.use_plateau_scheduler:
            lr_scheduler.step(val_stats['loss'])
        else:
            lr_scheduler.step()

        # 记录训练日志
        log_entry = {
            'epoch': epoch,
            'train_loss': train_stats['loss'],
            'test_loss': val_stats['loss'],
            'train_class_error': train_stats.get('class_error', 0),
            'test_class_error': val_stats.get('class_error', 0),
            'train_rel_error': train_stats.get('rel_error', 0),
            'test_rel_error': val_stats.get('rel_error', 0),
            'lr': optimizer.param_groups[0]['lr']
        }
        training_log.append(log_entry)
        
        # 打印进度
        print(f"Epoch {epoch}: train_loss={train_stats['loss']:.4f}, "
              f"val_loss={val_stats['loss']:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")

        # 早停逻辑
        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']
            patience_counter = 0
            print(f"💾 新的最佳验证损失: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f"🛑 早停在epoch {epoch}，连续{max_patience}个epoch无改善")
            break

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if val_stats['loss'] <= best_val_loss:
                checkpoint_paths.append(output_dir / 'checkpoint_best.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # 保存训练日志
        with open(output_dir / 'training_log_simple_optimized.json', 'w') as f:
            for entry in training_log:
                f.write(json.dumps(entry) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('训练时间 {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('简单优化训练', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
