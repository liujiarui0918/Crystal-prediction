# 基于 Matbench 的晶体性质预测项目

这个仓库已经从原来的 Materials Project API + 传统机器学习流程，重构为基于 Matbench 晶体数据集的图神经网络流程，并且只保留了三个指定模型：CGCNN、ALIGNN、M3GNet。

## 数据来源

- 数据来源：Matbench v0.1，通过 `matminer.datasets` 加载。
- 使用的数据任务：
   - `matbench_mp_gap`
   - `matbench_mp_e_form`
   - `matbench_mp_is_metal`
- 每条样本都包含完整的周期性晶体结构，以及一个监督目标。

## 预测目标

- `matbench_mp_gap`：带隙回归，单位为 `eV`
- `matbench_mp_e_form`：每原子形成能回归，单位为 `eV/atom`
- `matbench_mp_is_metal`：金属 / 非金属二分类

## 使用了哪些特征

这个项目已经不再使用手工构造的表格特征，而是直接把晶体结构转成图表示，再交给图神经网络训练。

- CGCNN
   - 节点特征：元素种类对应的原子特征嵌入
   - 边特征：原子间距离的高斯展开
- ALIGNN
   - 晶体图上的原子特征
   - 周期性邻接关系形成的键特征
   - 线图中的键角三体相互作用特征
- M3GNet
   - 带周期性偏移信息的晶体图
   - 三体线图相互作用
   - MatGL 转图器提供的状态属性

## 使用模型

- CGCNN
- ALIGNN
- M3GNet

之前的 Random Forest、HistGradientBoosting、MLP 已经从主流程中移除，不再作为当前项目结果的一部分。

## 项目结构

```text
materials-project-ml/
├── data/
│   ├── processed/
│   └── raw/
├── models/
│   ├── matbench_mp_gap/
│   ├── matbench_mp_e_form/
│   └── matbench_mp_is_metal/
├── reports/
│   ├── figures/
│   ├── metrics_summary.json
│   └── summary.md
├── scripts/
└── src/mp_crystal_ml/
```

## 环境安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## 运行方式

直接运行脚本：

```bash
python scripts/run_pipeline.py --sample-size 64 --force-fetch
```

或者使用安装后的命令：

```bash
mp-crystal-ml --sample-size 64 --force-fetch
```

参数说明：

- `--sample-size`：每个 Matbench 任务抽样的结构数量
- `--random-state`：采样与数据划分随机种子
- `--force-fetch`：忽略缓存，重新生成当前抽样数据

## 输出内容

运行结束后会生成以下内容：

- `data/raw/`：抽样后的 Matbench 子集缓存
- `data/processed/`：训练 / 验证 / 测试划分清单
- `models/<任务>/<模型>/`：各模型训练后的权重与历史记录
- `reports/figures/`：对比图、训练曲线、parity 图、ROC 图、混淆矩阵
- `reports/metrics_summary.json`：机器可读的指标结果
- `reports/summary.md`：人工可读的实验总结

## 当前实验结果

下面是本环境中最新完成的一轮 64 样本对比结果。

### matbench_mp_gap

- 最优模型：ALIGNN
- RMSE：0.7795
- MAE：0.6822
- R²：0.6931

### matbench_mp_e_form

- 最优模型：ALIGNN
- RMSE：1.2797
- MAE：1.0211
- R²：0.0469

### matbench_mp_is_metal

- 最优模型：ALIGNN
- ROC-AUC：1.0000
- Accuracy：0.7143
- F1：0.5000

更完整的指标、表格和图表请查看：

- `reports/summary.md`
- `reports/metrics_summary.json`
- `reports/figures/`

## 当前状态

- 已完成从旧数据流程到 Matbench 数据流程的迁移
- 已接入 CGCNN、ALIGNN、M3GNet 三个模型
- 已完成端到端训练、评估、结果汇总与绘图
- 已清理旧的传统机器学习模型产物和旧版图表
