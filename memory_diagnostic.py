import torch
import psutil
import os
import gc
import sys
import subprocess

def check_system_memory():
    """检查系统内存使用情况"""
    print("=== 系统内存状态 ===")
    memory = psutil.virtual_memory()
    print(f"总内存: {memory.total / 1024**3:.2f} GB")
    print(f"可用内存: {memory.available / 1024**3:.2f} GB")
    print(f"已用内存: {memory.used / 1024**3:.2f} GB")
    print(f"内存使用率: {memory.percent:.1f}%")
    print()

def check_gpu_memory():
    """检查GPU内存使用情况"""
    print("=== GPU内存状态 ===")
    if torch.cuda.is_available():
        # 清理缓存
        torch.cuda.empty_cache()
        gc.collect()
        
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        total = props.total_memory
        
        print(f"GPU: {props.name}")
        print(f"总显存: {total / 1024**3:.2f} GB")
        print(f"已分配: {allocated / 1024**3:.2f} GB")
        print(f"已保留: {reserved / 1024**3:.2f} GB")
        print(f"空闲: {(total - reserved) / 1024**3:.2f} GB")
        print(f"碎片化: {(reserved - allocated) / 1024**3:.2f} GB")
        
        # 显示内存分配统计
        if hasattr(torch.cuda, 'memory_stats'):
            stats = torch.cuda.memory_stats(device)
            print(f"活跃内存块: {stats.get('active.all.current', 0)}")
            print(f"非活跃内存块: {stats.get('inactive_split.all.current', 0)}")
            
    else:
        print("CUDA不可用")
    print()

def check_cuda_processes():
    """检查CUDA进程"""
    print("=== CUDA进程状态 ===")
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("运行中的CUDA进程:")
            print("PID\t进程名\t\t显存使用(MB)")
            print("-" * 40)
            for line in result.stdout.strip().split('\n'):
                print(line.replace(',', '\t'))
        else:
            print("没有发现CUDA进程")
    except Exception as e:
        print(f"无法获取CUDA进程信息: {e}")
    print()

def check_pytorch_cache():
    """检查PyTorch缓存分配器状态"""
    print("=== PyTorch缓存分配器状态 ===")
    if torch.cuda.is_available():
        # 获取缓存分配器统计信息
        if hasattr(torch.cuda, 'memory_summary'):
            print(torch.cuda.memory_summary(device=0, abbreviated=True))
        
        print(f"缓存分配器设置:")
        print(f"  PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '未设置')}")
        
        # 检查内存碎片化
        total_reserved = torch.cuda.memory_reserved(0)
        total_allocated = torch.cuda.memory_allocated(0)
        fragmentation = total_reserved - total_allocated
        
        if total_reserved > 0:
            frag_percent = (fragmentation / total_reserved) * 100
            print(f"内存碎片化: {fragmentation / 1024**3:.2f} GB ({frag_percent:.1f}%)")
            
            if frag_percent > 50:
                print("警告: 内存碎片化严重！")
                print("建议: 重启Python进程或调整PYTORCH_CUDA_ALLOC_CONF")
    print()

def check_environment():
    """检查环境配置"""
    print("=== 环境配置 ===")
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")
    
    # 检查相关环境变量
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'PYTORCH_CUDA_ALLOC_CONF',
        'CUDA_LAUNCH_BLOCKING',
        'CUDA_CACHE_DISABLE'
    ]
    
    print("环境变量:")
    for var in env_vars:
        value = os.environ.get(var, '未设置')
        print(f"  {var}: {value}")
    print()

def memory_stress_test():
    """简单的内存压力测试"""
    print("=== 内存压力测试 ===")
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过测试")
        return
    
    device = torch.cuda.current_device()
    initial_allocated = torch.cuda.memory_allocated(device)
    
    try:
        # 尝试分配较大的张量
        print("尝试分配1GB张量...")
        tensor_1gb = torch.randn(1024, 1024, 256, device='cuda')  # 约1GB
        print(f"成功分配1GB，当前分配: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        
        print("尝试分配额外的2GB张量...")
        tensor_2gb = torch.randn(1024, 1024, 512, device='cuda')  # 约2GB
        print(f"成功分配2GB，当前分配: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        
        # 清理
        del tensor_1gb, tensor_2gb
        torch.cuda.empty_cache()
        
        final_allocated = torch.cuda.memory_allocated(device)
        print(f"清理后分配: {final_allocated / 1024**3:.2f} GB")
        
        if final_allocated > initial_allocated + 100 * 1024**2:  # 100MB tolerance
            print("警告: 内存可能未完全释放")
        else:
            print("内存清理正常")
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"内存不足: {e}")
            torch.cuda.empty_cache()
        else:
            print(f"其他错误: {e}")
    print()

def suggest_optimizations():
    """建议内存优化措施"""
    print("=== 内存优化建议 ===")
    
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print("1. 环境变量优化:")
        print("   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")
        print("   export CUDA_LAUNCH_BLOCKING=1")
        
        print("\n2. 代码优化:")
        print("   - 使用torch.cuda.empty_cache()定期清理缓存")
        print("   - 减少batch_size")
        print("   - 使用gradient_checkpointing")
        print("   - 使用混合精度训练(AMP)")
        
        print("\n3. 模型优化:")
        if total_memory < 20:
            print("   - 考虑减少查询数量")
            print("   - 使用较小的模型backbone")
        
        print("\n4. 数据优化:")
        print("   - 减少num_workers")
        print("   - 禁用pin_memory")
        print("   - 使用较小的图片分辨率")
        
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        if reserved > total_memory * 0.8:
            print(f"\n警告: 已保留内存({reserved:.1f}GB)接近总内存({total_memory:.1f}GB)!")
            print("建议重启Python进程")

def main():
    print("PyTorch GPU内存诊断工具")
    print("=" * 50)
    
    check_system_memory()
    check_gpu_memory()
    check_cuda_processes()
    check_pytorch_cache()
    check_environment()
    memory_stress_test()
    suggest_optimizations()
    
    print("诊断完成!")

if __name__ == "__main__":
    main()
