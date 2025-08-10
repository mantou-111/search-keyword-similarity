#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU模式配置文件 - 避免GPU内存问题
"""

import torch
from pathlib import Path

# CPU模式配置
class Config:
    MODEL_PATH = Path('D:/model/BAAI-bge-large-zh-v1.5')  # 模型路径
    INPUT_DIR = Path('./input')  # 输入目录
    OUTPUT_DIR = Path('./output')  # 输出目录
    HISTORY_DIR = OUTPUT_DIR / 'history'  # 历史文件目录
    CACHE_DIR = Path('./cache')  # 缓存目录
    MAX_LENGTH = 64  # 最大文本长度
    DEVICE = 'cpu'  # 强制使用CPU
    
    # CPU模式优化参数
    BATCH_SIZE = 16  # 向量化批处理大小（CPU模式较小）
    MEMORY_CACHE_SIZE = 20000  # 内存缓存大小
    QUEUE_LIMIT = 5000  # 异步写入队列大小
    MAX_CACHE_VECTORS = 100000  # SQLite缓存最大向量数
    TEXT_BATCH_SIZE = 2000  # 文本分批大小（CPU模式较小）
    USE_PARQUET = False  # 是否使用Parquet格式保存
    
    # 内存管理配置
    MEMORY_CLEANUP_INTERVAL = 1  # 每处理1个批次清理一次内存
    ENABLE_MEMORY_MONITORING = True  # 是否启用内存监控
    
    # 动态批次调整
    ENABLE_DYNAMIC_BATCHING = True  # 启用动态批次大小调整
    MIN_BATCH_SIZE = 4  # 最小批次大小
    MAX_BATCH_SIZE = 32  # 最大批次大小
    BATCH_ADJUSTMENT_FACTOR = 0.5  # 批次大小调整因子
    
    # 并行处理配置
    ENABLE_PARALLEL_PROCESSING = False  # 禁用并行处理
    PARALLEL_WORKERS = 1  # 并行工作线程数
    PARALLEL_CHUNK_SIZE = 2000  # 并行处理块大小
    
    # CPU模式特殊配置
    CPU_NUM_THREADS = 4  # CPU线程数
    ENABLE_CPU_OPTIMIZATION = True  # 启用CPU优化
    
    @classmethod
    def init_dirs(cls):
        # 初始化目录
        for d in [cls.INPUT_DIR, cls.OUTPUT_DIR, cls.HISTORY_DIR, cls.CACHE_DIR]:
            d.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def setup_cpu_optimization(cls):
        """设置CPU优化"""
        if cls.ENABLE_CPU_OPTIMIZATION:
            # 设置CPU线程数
            torch.set_num_threads(cls.CPU_NUM_THREADS)
            # 设置内存分配策略
            torch.set_num_interop_threads(1)
            print(f"CPU优化设置: 线程数={cls.CPU_NUM_THREADS}")

# 预设性能配置方案
def set_cpu_performance_profile(profile='balanced'):
    """设置CPU性能配置方案"""
    if profile == 'ultra_safe':
        # 超安全模式
        Config.BATCH_SIZE = 8
        Config.TEXT_BATCH_SIZE = 1000
        Config.MEMORY_CACHE_SIZE = 10000
        Config.MEMORY_CLEANUP_INTERVAL = 1
        Config.MIN_BATCH_SIZE = 2
        Config.MAX_BATCH_SIZE = 16
        print("已设置为CPU超安全模式")
        
    elif profile == 'memory_saving':
        # 内存节省模式
        Config.BATCH_SIZE = 16
        Config.TEXT_BATCH_SIZE = 2000
        Config.MEMORY_CACHE_SIZE = 20000
        Config.MEMORY_CLEANUP_INTERVAL = 1
        Config.MIN_BATCH_SIZE = 4
        Config.MAX_BATCH_SIZE = 32
        print("已设置为CPU内存节省模式")
        
    elif profile == 'balanced':
        # 平衡模式
        Config.BATCH_SIZE = 32
        Config.TEXT_BATCH_SIZE = 4000
        Config.MEMORY_CACHE_SIZE = 30000
        Config.MEMORY_CLEANUP_INTERVAL = 2
        Config.MIN_BATCH_SIZE = 8
        Config.MAX_BATCH_SIZE = 64
        print("已设置为CPU平衡模式")
        
    elif profile == 'speed_optimized':
        # 速度优化模式
        Config.BATCH_SIZE = 64
        Config.TEXT_BATCH_SIZE = 8000
        Config.MEMORY_CACHE_SIZE = 50000
        Config.MEMORY_CLEANUP_INTERVAL = 3
        Config.MIN_BATCH_SIZE = 16
        Config.MAX_BATCH_SIZE = 128
        print("已设置为CPU速度优化模式")

if __name__ == '__main__':
    # 测试CPU配置
    Config.setup_cpu_optimization()
    set_cpu_performance_profile('balanced')
    print(f"CPU模式配置完成: DEVICE={Config.DEVICE}, BATCH_SIZE={Config.BATCH_SIZE}") 