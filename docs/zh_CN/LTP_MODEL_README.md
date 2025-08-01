# LTP Model Integration

本项目已集成了哈工大社会计算与信息检索研究中心开发的 LTP (Language Technology Platform) 模型，用于中文自然语言处理任务。

## 模型信息

- **模型名称**: LTP/base
- **来源**: https://hf-mirror.com/LTP/base
- **支持任务**: 
  - 中文分词 (CWS)
  - 词性标注 (POS)
  - 命名实体识别 (NER)
  - 语义角色标注 (SRL)
  - 依存句法分析 (DEP)
  - 语义依存分析 (SDP)

## 使用方法

### 1. 基本使用

```python
from nn.models import LTPProcessor
from utils.logging_config import logger

# 初始化 LTP 处理器
ltp_processor = LTPProcessor()

# 测试文本
texts = ["他叫汤姆去拿外衣。", "我爱北京天安门。"]

# 分词
cws_result = ltp_processor.word_segmentation(texts)
logger.info(f"分词结果: {cws_result}")

# 词性标注
pos_result = ltp_processor.pos_tagging(texts)
logger.info(f"词性标注: {pos_result}")

# 命名实体识别
ner_result = ltp_processor.named_entity_recognition(texts)
logger.info(f"命名实体识别: {ner_result}")

# 完整分析
full_result = ltp_processor.full_analysis(texts)
logger.info(f"完整分析: {full_result}")
```

### 2. 自定义任务

```python
# 指定特定任务
custom_result = ltp_processor.process_text(
    texts, 
    tasks=["cws", "pos", "ner"]
)
```

### 3. 模型导出

```python
# 运行导出脚本
python nn/export_model.py
```

导出后会生成以下文件：
- `exported_model/ltp_model.onnx` - ONNX 格式模型
- `exported_model/ltp_model.pt` - TorchScript 格式模型
- `exported_model/ltp_tokenizer/` - LTP 分词器

## 依赖安装

首先需要安装 LTP 库：

```bash
pip install ltp
```

本项目推荐使用 [uv](https://github.com/astral-sh/uv) 进行 Python 依赖管理：

```bash
uv pip install -e .
```

## 常见问题

### 1. 模型加载失败

如果遇到以下错误：
```
Unrecognized model in LTP/base. Should have a `model_type` key in its config.json
```

这是因为 LTP/base 模型不被 transformers 库直接支持。解决方案：

1. 确保安装了 LTP 库：`pip install ltp`
2. 代码会自动使用 LTP 库的 Pipeline API
3. 如果 LTP 库不可用，会自动回退到 `bert-base-chinese` 作为备用模型

### 2. 网络连接问题

如果无法下载模型，可以：

1. 设置 Hugging Face 镜像：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

2. 或者使用本地模型路径

### 3. 功能限制

- 如果只安装了 transformers 而没有 LTP 库，某些高级 NLP 功能（如精确的中文分词、词性标注）可能不可用
- 备用模型主要提供基础的文本嵌入功能

## 测试

运行测试脚本验证 LTP 模型功能：

```bash
python test_ltp.py
```

## 注意事项

1. LTP 模型主要针对中文文本设计
2. 输入文本长度建议不超过 512 个 token
3. 如果安装了完整的 LTP 库，将使用官方的 Pipeline API
4. 如果只安装了 transformers，将回退到基础的模型推理

## 模型配置

在 `nn/models.py` 中的相关配置：

```python
LTP_MODEL_NAME = "LTP/base"
LTP_MAX_INPUT_LENGTH = 512
```

## 性能参考

根据官方文档，LTP Base 模型在各项任务上的性能：

| 任务 | F1 分数 |
|------|---------|
| 分词 (CWS) | 98.7 |
| 词性标注 (POS) | 98.5 |
| 命名实体识别 (NER) | 95.4 |
| 语义角色标注 (SRL) | 80.6 |
| 依存句法 (DEP) | 89.5 |
| 语义依存 (SDP) | 75.2 |

## 引用

如果在研究中使用了 LTP 模型，请引用：

```bibtex
@article{che2020n,
  title={N-LTP: A Open-source Neural Chinese Language Technology Platform with Pretrained Models},
  author={Che, Wanxiang and Feng, Yunlong and Qin, Libo and Liu, Ting},
  journal={arXiv preprint arXiv:2009.11616},
  year={2020}
}
```
