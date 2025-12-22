本项目为 Coding with AI 课程作业。

# Molecule3D Benchmark：2D → 3D 构型预测
本仓库实现了 Molecule3D 数据集的基础数据处理、基线模型与评估流程，用于比较不同方法在 2D（SMILES）到 3D 构型预测任务上的表现。

## 功能概览
- 数据处理：从 Molecule3D parquet 或自定义 SDF 提取 `SMILES + SDF`，输出 `npz` 分片与 manifest。
- 模型基线：RDKit ETKDG、k-NN 模板检索、距离矩阵回归 + MDS、MPNN、EGNN。
- 评估指标：使用 RDKit `rdMolAlign.GetBestRMS()` 进行对称性识别的 RMSD 计算与统计。

## 目录结构
```
molecule3d-benchmark/
  scripts/
    prepare_data.py        # 数据提取与分片
    read_npz.py            # 查看分片内容
    eval_etkdg.py          # ETKDG 基线评估与绘图
  src/
    datasets/              # Molecule3D 读取与图构建
    models/                # ETKDG、k-NN、距离回归、MPNN、EGNN
    training/              # 训练与评估函数
    metrics/               # RMSD 计算与统计
```

## 环境依赖
基础依赖：
- `numpy`
- `rdkit`
- `pyarrow`

可选依赖（按需）：
- `huggingface_hub`：自动下载 Molecule3D 数据集
- `scikit-learn`：距离回归模型
- `xgboost`：XGBoost 距离回归
- `torch`：MPNN / EGNN
- `matplotlib`、`tqdm`：`eval_etkdg.py` 绘图与进度条

## 数据准备
默认优先使用本地 Molecule3D 缓存（HuggingFace 缓存目录），否则可自动下载。

示例：生成 CHON 子集（400000/50000/50000）
```bash
python scripts/prepare_data.py --split train --max-samples 400000 --allowed-elements C,H,O,N --prefix train_chon_smiles_sdf
python scripts/prepare_data.py --split validation --max-samples 50000 --allowed-elements C,H,O,N --prefix val_chon_smiles_sdf
python scripts/prepare_data.py --split test --max-samples 50000 --allowed-elements C,H,O,N --prefix test_chon_smiles_sdf
```

常用参数：
- `--data-dir`：指向 `Molecule3D_random_split` 目录
- `--split`：`train` / `validation` / `test`
- `--allowed-elements`：仅保留指定元素（如 `C,H,O,N`）
- `--atom-count`：仅保留指定原子数
- `--max-samples`：最多保留样本数
- `--shard-size`：分片大小
- `--sdf-path`：读取自定义 SDF（绕过 parquet）

输出：
- `<out-dir>/<prefix>_shardXXX.npz`
- `<out-dir>/<prefix>_manifest.json`

## 查看分片内容
```bash
python scripts/read_npz.py --npz data/processed/train_chon_smiles_sdf_shard000.npz --limit 3
```

## ETKDG 基线评估
```bash
python scripts/eval_etkdg.py --manifest data/processed/test_chon_smiles_sdf_manifest.json 
```

该脚本会输出 RMSD 统计并保存直方图与 CDF 图（如果有样本）。

## Python API（示例）
```python
from src.datasets.molecule3d import load_records
from src.models import FeatureConfig
from src.training.baselines import train_distance_regressor, train_knn, build_etkdg, evaluate_model

records = load_records(data_dir=None, split="train", max_samples=5000)
cfg = FeatureConfig(atom_count=20)
model = train_distance_regressor(records, cfg=cfg, model_name="rf")
summary = evaluate_model(model, records, max_samples=500)
print(summary)
```

## 评估方法说明
为确保评估的准确性并符合化学信息学的通用标准，本项目采用 RDKit `rdkit.Chem.rdMolAlign` 模块进行定量分析。RDKit 能够自动识别并处理分子的对称性（Symmetry Awareness），避免因原子编号相同但空间对称位置不同而导致的误差。具体流程如下：
1. **对象重构**：将模型输出的 3D 坐标矩阵映射回对应的 RDKit 分子对象（Mol Object），确保预测结构与真实结构拥有相同的原子拓扑顺序。
2. **最佳叠合与计算**：调用 `rdMolAlign.GetBestRMS()`，在旋转/平移对齐的同时考虑对称性操作，取最小 RMSD 作为最终误差。
