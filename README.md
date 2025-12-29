本项目为 Coding with AI 课程作业。

# Molecule3D Benchmark：2D → 3D 构型预测
本仓库实现了 Molecule3D 数据集的基础数据处理、基线模型与评估流程，用于比较不同方法在 2D（SMILES）到 3D 构型预测任务上的表现。

## 功能概览
- 数据处理：从 Molecule3D parquet 或自定义 SDF 提取 `SMILES + SDF`，输出 `npz` 分片与 manifest。
- 模型基线：RDKit ETKDG、k-NN 模板检索、距离矩阵回归 + MDS、MPNN、EGNN、EGNN+Transformer。
- 统一评估：各方法先输出结构，再用单一脚本计算 RDKit 对称性 RMSD 与绘图分析。

## 目录结构
```
molecule3d-benchmark/
  scripts/
    prepare_data.py        # 数据提取与分片
    read_npz.py            # 查看分片内容
    train_knn.py           # 训练 k-NN 模板检索
    train_distance_regressor.py # 训练距离回归模型
    train_mpnn.py          # 训练 MPNN
    train_egnn.py          # 训练 EGNN
    train_egnn_transformer.py # 训练 EGNN+Transformer
    predict_structures.py  # 统一预测结构输出
    eval_predictions.py    # 统一 RMSD 评估与绘图
  src/
    datasets/              # Molecule3D 读取与图构建
    models/                # ETKDG、k-NN、距离回归、MPNN、EGNN、EGNN+Transformer
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
- `torch`：MPNN / EGNN / EGNN+Transformer
- `matplotlib`：`eval_predictions.py` 绘图

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
可选：加 `--relative-paths` 让 manifest 中的 shard 路径可迁移到其他机器（相对路径）。

## 查看分片内容
```bash
python scripts/read_npz.py --npz data/processed/train_chon_smiles_sdf_shard000.npz --limit 3
```

## 统一训练/预测/评估流程
1) 使用 `prepare_data.py` 生成训练与测试 manifest（如 CHON）。
2) 训练需要训练的模型（默认读取全部数据，脚本输入统一使用 manifest）。
3) 用统一预测脚本输出结构（`.npz`）。
4) 用统一评估脚本计算 RMSD 并绘图。

### 训练（示例）
```bash
python scripts/train_knn.py --manifest data/processed/train_chon_smiles_sdf_manifest.json
python scripts/train_distance_regressor.py --manifest data/processed/train_chon_smiles_sdf_manifest.json --atom-count 20
python scripts/train_mpnn.py --manifest data/processed/train_chon_smiles_sdf_manifest.json
python scripts/train_egnn.py --manifest data/processed/train_chon_smiles_sdf_manifest.json
python scripts/train_egnn_transformer.py --manifest data/processed/train_chon_smiles_sdf_manifest.json
```

### 预测（示例）
```bash
python scripts/predict_structures.py --method etkdg --manifest data/processed/test_chon_smiles_sdf_manifest.json
python scripts/predict_structures.py --method knn --checkpoint checkpoints/knn.pkl --manifest data/processed/test_chon_smiles_sdf_manifest.json
python scripts/predict_structures.py --method distance_regressor --checkpoint checkpoints/distance_regressor.joblib --manifest data/processed/test_chon_smiles_sdf_manifest.json
python scripts/predict_structures.py --method mpnn --checkpoint checkpoints/mpnn.pt --manifest data/processed/test_chon_smiles_sdf_manifest.json
python scripts/predict_structures.py --method egnn --checkpoint checkpoints/egnn.pt --manifest data/processed/test_chon_smiles_sdf_manifest.json
python scripts/predict_structures.py --method egnn_transformer --checkpoint checkpoints/egnn_transformer.pt --manifest data/processed/test_chon_smiles_sdf_manifest.json
```

### 评估与绘图（示例）
```bash
python scripts/eval_predictions.py --predictions pred_etkdg_test_chon_smiles_sdf.npz --manifest data/processed/test_chon_smiles_sdf_manifest.json
```
默认会输出 RMSD 直方图与 CDF 图（`<predictions>_rmsd.png`）。
预测与评估请使用同一个 manifest，以保证样本顺序一致。
默认会保存指标日志（`<predictions>_metrics.json`），可用 `--output` 指定路径。
可选：传入 `--per-sample-output` 生成每个样本的 RMSD 记录（JSONL），便于绘制自定义图。

## 最小流程一键脚本
```bash
bash scripts/run_minitest.sh
```
可选：如果需要指定 parquet 目录，先设置 `DATA_DIR=/path/to/Molecule3D_random_split` 再运行。

## work/ 对比实验脚本
用于横向对比 MPNN / EGNN / EGNN+Transformer（训练规模 1000/5000/10000/20000）。
```bash
bash work/compare_mpnn/run.sh
bash work/compare_egnn/run.sh
bash work/compare_egnn_transformer/run.sh
```

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

## 方法简介
- RDKit ETKDG：使用 ETKDGv3 生成多构象，按能量（可选 MMFF）选择最优构象作为预测。
- k-NN 模板检索：用分子指纹检索最相似训练样本，将模板 3D 坐标映射到查询分子。
- 距离矩阵回归 + MDS：回归原子对距离向量，再用经典 MDS 还原 3D 坐标（可选 MMFF 微调）。
- MPNN：消息传递网络预测原子对距离向量，随后用 MDS 重建坐标。
- EGNN：等变 GNN 直接预测 3D 坐标，保持几何等变性。
- EGNN+Transformer：在 EGNN 坐标更新后加入图 Transformer 注意力层，增强全局建模能力。
