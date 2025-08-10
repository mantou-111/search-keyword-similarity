import os
import torch
from pathlib import Path

class Config:
    # 基础路径配置
    BASE_DIR = Path(__file__).parent
    INPUT_DIR = BASE_DIR / 'input'
    OUTPUT_DIR = BASE_DIR / 'output'
    CACHE_DIR = BASE_DIR / 'cache'
    HISTORY_DIR = OUTPUT_DIR / 'history'
    
    # 模型配置
    MODEL_PATH = os.getenv('MODEL_PATH', 'D:/model/text2vec-bge-large-chinese')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 处理参数
    BATCH_SIZE = 32
    MAX_LENGTH = 128
    
    @classmethod
    def init_dirs(cls):
        """创建必要目录"""
        for d in [cls.INPUT_DIR, cls.OUTPUT_DIR, cls.CACHE_DIR, cls.HISTORY_DIR]:
            d.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def check(cls):
        """检查所有依赖版本"""
        missing = []
        for lib, ver in cls.REQUIREMENTS.items():
            try:
                mod = __import__(lib)
                if hasattr(mod, '__version__'):
                    current_ver = mod.__version__
                    if lib == 'torch':  # torch版本号可能包含+cu117后缀
                        assert current_ver.startswith(ver), f"{lib}需要{ver}，当前是{current_ver}"
                    else:
                        assert current_ver == ver, f"{lib}需要{ver}，当前是{current_ver}"
                
                # 特殊检查NumPy二进制兼容性
                if lib == 'numpy':
                    import numpy as np
                    assert np.dtype(np.void).itemsize == 96, "NumPy二进制不兼容"
                    
            except ImportError:
                missing.append(f"{lib}=={ver}")
            except AssertionError as e:
                print(f"版本错误: {str(e)}")
                missing.append(f"{lib}=={ver}")
        
        if missing:
            raise ImportError(
                f"缺少或不兼容的依赖:\n{', '.join(missing)}\n"
                "请执行以下命令修复:\n"
                "  conda install -c conda-forge numpy=1.21.2 pytables=3.6.1\n"
                "  pip install torch==1.13.1 transformers==4.30.0"
            )
