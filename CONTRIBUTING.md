# 贡献指南

感谢你考虑为搜索词相似度分析工具做出贡献！我们欢迎各种形式的贡献，包括但不限于：

- 🐛 报告Bug
- 💡 提出新功能建议
- 📝 改进文档
- 🔧 提交代码修复
- 🌟 分享使用经验

## 如何贡献

### 1. 报告Bug

如果你发现了Bug，请：

1. 在 [Issues](https://github.com/your-username/search-keyword-similarity/issues) 页面搜索是否已有相关报告
2. 如果没有，请创建新的Issue
3. 使用Bug报告模板，包含以下信息：
   - 操作系统和Python版本
   - 错误信息和堆栈跟踪
   - 重现步骤
   - 期望行为

### 2. 提出功能建议

如果你有新功能想法，请：

1. 在Issues页面搜索是否已有相关建议
2. 如果没有，请创建新的Issue
3. 使用功能请求模板，详细描述：
   - 功能用途
   - 使用场景
   - 预期效果

### 3. 提交代码

如果你想贡献代码，请：

1. Fork项目到你的GitHub账户
2. 创建功能分支：
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. 进行代码修改
4. 提交更改：
   ```bash
   git add .
   git commit -m "feat: 添加新功能描述"
   ```
5. 推送到你的分支：
   ```bash
   git push origin feature/your-feature-name
   ```
6. 创建Pull Request

## 开发环境设置

### 1. 克隆项目

```bash
git clone https://github.com/your-username/search-keyword-similarity.git
cd search-keyword-similarity
```

### 2. 创建开发环境

```bash
# 使用conda
conda env create -f environment.yml
conda activate similarity

# 或使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. 安装开发依赖

```bash
pip install pytest black flake8 mypy
```

## 代码规范

### 1. Python代码风格

我们遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 代码风格：

- 使用4个空格缩进
- 行长度不超过120字符
- 使用snake_case命名变量和函数
- 使用PascalCase命名类

### 2. 代码格式化

使用 [Black](https://black.readthedocs.io/) 自动格式化代码：

```bash
black .
```

### 3. 代码检查

使用 [Flake8](https://flake8.pycqa.org/) 检查代码质量：

```bash
flake8 .
```

### 4. 类型检查

使用 [MyPy](http://mypy-lang.org/) 进行类型检查：

```bash
mypy .
```

### 5. 运行测试

```bash
pytest
```

## 提交信息规范

我们使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### 类型说明

- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

### 示例

```
feat: 添加新的相似度计算方法

- 实现余弦相似度计算
- 添加相似度阈值配置
- 更新相关测试用例

Closes #123
```

## 文档贡献

### 1. 更新README

如果你改进了功能或添加了新特性，请相应更新README文档。

### 2. 添加文档

- 为新功能添加使用说明
- 更新安装指南
- 添加常见问题解答

### 3. 文档格式

- 使用Markdown格式
- 保持结构清晰
- 添加适当的代码示例

## 测试要求

### 1. 单元测试

- 为新功能添加测试用例
- 确保测试覆盖率不降低
- 测试应该独立且可重复

### 2. 集成测试

- 测试主要功能流程
- 验证不同配置下的行为
- 测试错误处理

### 3. 性能测试

- 对于性能敏感的代码，添加基准测试
- 确保修改不会显著影响性能

## 审查流程

### 1. Pull Request审查

- 所有代码更改都需要通过Pull Request
- 至少需要一名维护者审查
- 通过所有自动化检查

### 2. 审查要点

- 代码质量和风格
- 功能完整性
- 测试覆盖
- 文档更新
- 性能影响

### 3. 合并要求

- 通过所有测试
- 代码审查通过
- 解决所有反馈意见

## 发布流程

### 1. 版本号

我们使用 [语义化版本](https://semver.org/)：

- `MAJOR.MINOR.PATCH`
- 主版本号：不兼容的API修改
- 次版本号：向下兼容的功能性新增
- 修订号：向下兼容的问题修正

### 2. 发布步骤

1. 更新版本号
2. 更新CHANGELOG
3. 创建发布标签
4. 发布到PyPI（如果适用）

## 社区行为准则

### 1. 尊重他人

- 保持友好和专业的交流
- 尊重不同的观点和经验
- 避免个人攻击或不当言论

### 2. 建设性反馈

- 提供具体和有建设性的反馈
- 关注代码和想法，而不是个人
- 帮助他人学习和成长

### 3. 包容性

- 欢迎不同背景的贡献者
- 创造包容和友好的环境
- 支持新手的成长

## 获取帮助

如果你在贡献过程中遇到问题：

1. 查看 [Issues](https://github.com/your-username/search-keyword-similarity/issues) 页面
2. 搜索 [Discussions](https://github.com/your-username/search-keyword-similarity/discussions) 页面
3. 创建新的Issue或Discussion
4. 联系项目维护者

## 致谢

感谢所有为这个项目做出贡献的开发者！你的贡献让这个工具变得更好。

---

再次感谢你的贡献！🎉
