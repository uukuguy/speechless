# Improved Data Processing Pipeline

这是一个改进的数据预处理管道，解决了原始设计中的主要缺陷，提供了更好的模块化、可测试性和可维护性。

## 主要改进

### 1. 配置管理
- **YAML配置文件**: 使用YAML配置文件替代代码配置，更易维护
- **配置验证**: 内置配置验证机制，在运行前发现问题
- **配置模板**: 提供预设的数据集配置模板

### 2. 模块化架构
- **处理器工厂**: 使用工厂模式创建不同类型的处理器
- **职责分离**: 配置管理、数据处理、输出管理分离
- **可扩展性**: 易于添加新的数据集处理器

### 3. 错误处理
- **自定义异常**: 针对不同错误类型的专用异常类
- **详细日志**: 完整的处理日志和错误追踪
- **优雅降级**: 单个样例处理失败不影响整个流水线

### 4. 测试框架
- **单元测试**: 覆盖所有主要组件
- **集成测试**: 端到端测试保证整体功能
- **Mock支持**: 便于测试和调试

## 架构概览

```
improved_design/
├── config.py           # 配置管理系统
├── processors.py       # 数据处理器
├── output_manager.py   # 输出管理器
├── pipeline.py         # 主处理流水线
├── cli.py             # 命令行接口
├── test_framework.py  # 测试框架
├── configs/           # 配置文件目录
│   ├── gsm8k.yaml
│   └── math.yaml
└── README.md          # 此文件
```

## 使用方法

### 1. 初始化配置
```bash
python cli.py init-configs --config-dir configs
```

### 2. 处理单个数据集
```bash
python cli.py process --config gsm8k --output data/processed/gsm8k
```

### 3. 批量处理
```bash
python cli.py batch --configs gsm8k math --output data/processed
```

### 4. 验证配置
```bash
python cli.py validate --config gsm8k
```

### 5. 查看可用配置
```bash
python cli.py list-configs --detailed
```

## 配置文件格式

```yaml
name: gsm8k
data_source: data/openai/gsm8k
dataset_name: main
input_key: question
output_key: answer
ability: math
reward_style: rule
splits:
  - train
  - test
output_format: parquet
extract_answer: true
format_prompt: true
custom_params:
  answer_pattern: "#### ([\\-]?[0-9\\.\\,]+)"
  prompt_template: "{question} Let's think step by step."
```

## 添加新的数据集处理器

### 1. 创建处理器类
```python
class MyDatasetProcessor(DataProcessor):
    def extract_answer(self, raw_answer: str) -> str:
        # 实现答案提取逻辑
        pass
    
    def format_prompt(self, question: str) -> str:
        # 实现提示格式化逻辑
        pass
```

### 2. 注册处理器
```python
ProcessorFactory.register_processor("mydataset", MyDatasetProcessor)
```

### 3. 创建配置文件
```yaml
name: mydataset
data_source: path/to/my/dataset
# ... 其他配置
```

## 运行测试

```bash
python test_framework.py
```

## 与原始设计的比较

| 特性 | 原始设计 | 改进设计 |
|------|----------|----------|
| 配置管理 | 代码配置 | YAML文件配置 |
| 错误处理 | assert语句 | 自定义异常和日志 |
| 职责分离 | 混合责任 | 清晰的模块化 |
| 测试支持 | 困难 | 完整的测试框架 |
| 可扩展性 | 中等 | 高度可扩展 |
| 维护性 | 低 | 高 |

## 性能特性

- **内存效率**: 使用流式处理，避免加载整个数据集到内存
- **错误恢复**: 单个样例失败不会中断整个流程
- **并行支持**: 为未来添加并行处理做准备
- **缓存支持**: 可选的结果缓存机制

## 最佳实践

1. **配置验证**: 运行前总是验证配置
2. **干运行**: 使用 `--dry-run` 测试配置
3. **日志监控**: 关注处理日志中的错误信息
4. **批量处理**: 对相似数据集使用批量处理
5. **测试驱动**: 添加新功能时先写测试

这个改进的设计解决了原始架构的主要问题，提供了更好的可维护性、可测试性和可扩展性。