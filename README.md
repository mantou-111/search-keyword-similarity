# 搜索词与关键词相似度分析工具

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个基于深度学习的搜索词与关键词相似度分析工具，基于文娱行业验证，能够自动判断搜索词与关键词的相关性，并提供智能化的账户优化建议。
日后会持续优化代码，完善其判断精准度和效率，并考虑结合大语言模型（LLM）给出更深入的优化建议，帮助各位优化师提高优化效率。

## 🌟 主要功能

- **智能相似度计算**: 基于text2vec-bge-large-chinese模型，提供高精度的语义相似度分析，硬件配置较高的情况下建议使用BAAI-bge-large-zh-v1.5模型
- **多模式运行**: 支持GPU/CPU模式，自动适配不同硬件环境
- **批量处理**: 支持大规模数据的分批处理，避免内存溢出
- **智能建议**: 根据相似度自动生成账户优化建议
- **安全运行**: 多种安全模式，解决内存溢出和系统稳定性问题
- **结果分析**: 自动生成相关性判断和优化建议报告

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+ (GPU模式，可选)
- 至少8GB内存 (推荐16GB+)

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-username/search-keyword-similarity.git
cd search-keyword-similarity
```

2. **创建虚拟环境**
```bash
# 使用conda (推荐)
conda create -n similarity python=3.8
conda activate similarity

# 或使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

3. **安装依赖**
```bash
# 使用requirements.txt
pip install -r requirements.txt

# 或使用conda
conda env update -f environment.yml
```

4. **下载模型**
```bash
# 首次运行会自动下载text2vec-bge-large-chinese模型
# 或手动指定模型路径
export MODEL_PATH="path/to/your/model"
```

### 基本使用

1. **准备数据**
   - 将搜索词数据放入 `input/` 目录
   - 支持Excel格式 (.xlsx)

2. **运行分析**
```bash
# 推荐使用安全启动脚本
python run_safe.py

# 或直接运行主程序
python main_auto_batch.py
```

3. **查看结果**
   - 结果文件保存在 `output/analysis/` 目录
   - 包含相似度分数和优化建议

## 📁 项目结构

```
search-keyword-similarity/
├── README.md                 # 项目说明文档
├── LICENSE                   # 开源许可证
├── requirements.txt          # Python依赖
├── environment.yml           # Conda环境配置
├── config.py                # 主配置文件
├── config_cpu.py            # CPU模式配置
├── main_auto_batch.py       # 自动批处理主程序
├── main_cpu.py              # CPU模式主程序
├── main_ultra_safe.py       # 超安全模式主程序
├── analyze_similarity_result.py  # 相似度结果分析
├── analyze_unit_levels.py   # 单元级别分析
├── run_safe.py              # 安全启动脚本
├── run_ultra_safe.py        # 超安全启动脚本
├── run_ultra_safe_quiet.py  # 静默超安全启动脚本
├── utils/                   # 工具函数
├── docs/                    # 详细文档
└── examples/                # 使用示例
```

## 🔧 配置说明

### 模型配置

在 `config.py` 中可以配置：
- 模型路径
- 设备选择 (GPU/CPU)
- 批次大小
- 最大文本长度

### 运行模式

1. **安全模式** (`run_safe.py`)
   - 自动检查系统资源
   - 推荐首次使用

2. **超安全模式** (`run_ultra_safe.py`)
   - 解决内存溢出问题
   - 适合不稳定环境

3. **CPU模式** (`main_cpu.py`)
   - 完全避免GPU问题
   - 适合小内存设备

4. **批处理模式** (`main_auto_batch.py`)
   - 自动分批处理大数据
   - 支持断点续传

## 📊 输出格式

分析结果包含以下字段：
- `final_similarity`: 最终相似度分数 (0-1)
- `相关性判断（建议）`: 强相关/较相关/弱相关/不相关
- `建议优化动作`: 添加至账户/需人工判断/加入否词等

## 🚨 注意事项

1. **首次运行**: 建议使用安全启动脚本
2. **大数据量**: 建议分批处理或使用CPU模式
3. **系统资源**: 确保有足够的磁盘空间和内存
4. **定期清理**: 定期清理缓存文件以释放空间
5. **模型下载**: 首次运行会自动下载模型文件 (~1.5GB)

## 🐛 故障排除

### 常见问题

1. **内存不足**
   - 使用CPU模式或超安全模式
   - 减少批次大小
   - 关闭其他程序

2. **GPU问题**
   - 检查CUDA版本兼容性
   - 使用CPU模式
   - 更新显卡驱动

3. **模型下载失败**
   - 检查网络连接
   - 手动下载模型文件
   - 配置代理设置

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [BAAI-bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) - 中文语义向量模型
- [text2vec-bge-large-chinese](https://huggingface.co/shibing624/text2vec-bge-large-chinese) - 中文语义向量模型
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Transformers](https://huggingface.co/transformers/) - 预训练模型库

## 📞 联系方式

- 项目主页: [GitHub Repository](https://github.com/your-username/search-keyword-similarity)
- 问题反馈: [Issues](https://github.com/your-username/search-keyword-similarity/issues)
- 邮箱: 962349993@qq.com

---

如果这个项目对你有帮助，请给它一个 ⭐️ 星标！


