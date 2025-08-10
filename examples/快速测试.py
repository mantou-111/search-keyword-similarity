#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本
用于验证安装是否成功和基本功能是否正常
"""

import sys
import os
from pathlib import Path

def test_imports():
    """测试基本依赖包导入"""
    print("🔍 测试依赖包导入...")
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas导入失败: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy导入失败: {e}")
        return False
    
    try:
        import openpyxl
        print(f"✅ OpenPyXL: {openpyxl.__version__}")
    except ImportError as e:
        print(f"❌ OpenPyXL导入失败: {e}")
        return False
    
    return True

def test_config():
    """测试配置文件"""
    print("\n🔍 测试配置文件...")
    
    try:
        from config import Config
        print("✅ 配置文件导入成功")
        
        # 检查必要目录
        for dir_path in [Config.INPUT_DIR, Config.OUTPUT_DIR, Config.CACHE_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ 目录创建成功: {dir_path}")
            
    except ImportError as e:
        print(f"❌ 配置文件导入失败: {e}")
        return False
    
    return True

def test_model_download():
    """测试模型下载功能"""
    print("\n🔍 测试模型下载功能...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # 尝试加载一个小模型进行测试
        print("正在下载测试模型...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", cache_dir="./cache")
        model = AutoModel.from_pretrained("bert-base-chinese", cache_dir="./cache")
        
        print("✅ 模型下载测试成功")
        
        # 清理测试模型
        import shutil
        if os.path.exists("./cache"):
            shutil.rmtree("./cache")
            print("✅ 测试缓存清理完成")
            
    except Exception as e:
        print(f"⚠️ 模型下载测试失败: {e}")
        print("   这可能是网络问题，不影响主要功能")
        return True  # 不阻止继续
    
    return True

def test_basic_functionality():
    """测试基本功能"""
    print("\n🔍 测试基本功能...")
    
    try:
        # 测试文本向量化
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        # 使用小模型进行测试
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", cache_dir="./cache")
        model = AutoModel.from_pretrained("bert-base-chinese", cache_dir="./cache")
        
        # 测试文本
        text1 = "电影推荐"
        text2 = "好看的电影"
        
        # 编码
        inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
        inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
        
        # 获取向量
        with torch.no_grad():
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)
            
            # 计算相似度
            embeddings1 = outputs1.last_hidden_state.mean(dim=1)
            embeddings2 = outputs2.last_hidden_state.mean(dim=1)
            
            similarity = torch.cosine_similarity(embeddings1, embeddings2)
            print(f"✅ 相似度计算测试成功: {similarity.item():.4f}")
        
        # 清理
        shutil.rmtree("./cache")
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False
    
    return True

def create_sample_data():
    """创建示例数据文件"""
    print("\n🔍 创建示例数据...")
    
    try:
        import pandas as pd
        
        # 示例数据
        sample_data = {
            '关键词/营销要点/知识问答': [
                '电影推荐',
                '游戏攻略', 
                '美食制作',
                '旅游攻略',
                '健康养生'
            ],
            '搜索词': [
                '好看的电影',
                '游戏技巧',
                '家常菜做法', 
                '景点推荐',
                '养生方法'
            ],
            '账户添加状态': [
                '未添加',
                '已添加',
                '未添加',
                '未添加', 
                '已添加'
            ]
        }
        
        df = pd.DataFrame(sample_data)
        
        # 确保目录存在
        input_dir = Path('./input')
        input_dir.mkdir(exist_ok=True)
        
        # 保存示例数据
        sample_file = input_dir / '示例数据.xlsx'
        df.to_excel(sample_file, index=False)
        print(f"✅ 示例数据创建成功: {sample_file}")
        
    except Exception as e:
        print(f"❌ 示例数据创建失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 搜索词相似度分析工具 - 快速测试")
    print("=" * 50)
    
    # 测试依赖包
    if not test_imports():
        print("\n❌ 依赖包测试失败，请检查安装")
        return False
    
    # 测试配置文件
    if not test_config():
        print("\n❌ 配置文件测试失败")
        return False
    
    # 测试模型下载
    if not test_model_download():
        print("\n❌ 模型下载测试失败")
        return False
    
    # 测试基本功能
    if not test_basic_functionality():
        print("\n❌ 基本功能测试失败")
        return False
    
    # 创建示例数据
    if not create_sample_data():
        print("\n❌ 示例数据创建失败")
        return False
    
    print("\n🎉 所有测试通过！")
    print("\n📋 下一步操作:")
    print("1. 将你的数据文件放入 input/ 目录")
    print("2. 运行 python run_safe.py 开始分析")
    print("3. 查看 output/analysis/ 目录中的结果")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        sys.exit(1)
