# 基于 BERT 的中文标点恢复

本项目使用预训练中文 BERT，在人民日报 1998 年带标点语料上进行 Token Classification 微调，实现中文文本的标点恢复。项目同时实现了面向长文本的重叠滑窗推理，并比较不同推理批次大小下的处理速度和显存占用。

## 环境要求

项目需要 Python、PyTorch、Transformers、scikit-learn、NumPy 和 Matplotlib。使用 GPU 训练时，应安装与本机 CUDA 驱动匹配的 PyTorch 版本。

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

如果需要从 Hugging Face 下载预训练模型，可设置镜像地址：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

## 完整运行流程

首先处理原始语料，生成训练集、验证集和测试集：

```powershell
python prepare_data.py
```

然后进行 BERT 微调：

```powershell
python train.py --epochs 1 --batch-size 16 --grad-accumulation 1 --max-length 256
```

接着在完整测试集上评估模型，并测试不同窗口批次大小的推理吞吐：

```powershell
python evaluate.py --batch-sizes 1 4 8 16 --repeats 2 --benchmark-articles 100 --window-chars 448 --stride 320
```

最后生成实验图表：

```powershell
python make_plots.py
```

## 脚本说明

- `punctuation.py`：定义标点类别，并负责语料行解析、目标标点去除、标签构造和标点恢复。
- `prepare_data.py`：以 GB18030 编码读取人民日报语料，清除拼音注释和词间空格，按照文章编号进行确定性数据集划分。
- `train.py`：使用 `google-bert/bert-base-chinese` 进行 Token Classification 微调，并保存验证集表现最好的模型。
- `inference.py`：使用固定起点的重叠滑窗和批量推理处理长文本，合并重叠窗口的预测结果。
- `evaluate.py`：计算测试集准确率、各类别 F1、混淆矩阵和不同 batch size 下的吞吐率。
- `make_plots.py`：生成训练曲线、各标点类别 F1、混淆矩阵和批量推理吞吐图。

## 主要输出

- `data/processed/`：预处理后的数据集和数据统计摘要。
- `outputs/bert-punctuation/`：微调后的模型、Tokenizer 和训练结果。
- `outputs/evaluation/`：测试集指标、预测结果和吞吐测试结果。
- `figures/`：训练、评估和吞吐测试生成的四张图表。

## 单独推理

训练完成后，可以对 UTF-8 编码的文本文件进行标点恢复。输入文件按行读取，每行视为一段文本：

```powershell
python inference.py --input input.txt --output predictions.txt --batch-size 8
```

长文本默认使用 448 个字符的窗口和 320 个字符的步长，相邻窗口重叠 128 个字符。`--batch-size` 可以根据显存大小调整。