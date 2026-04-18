# Photovoltaic Forecasting Experiment

这个仓库提供我在智慧能源概论课程报告里的一个小实验，用于做光伏短时功率预测，并自动产出可用于展示的结果文件。

主要有两个python文件：

- `experiments/main.py`：负责读取数据、预处理、训练模型、评估指标，并保存实验结果数据。
- `experiments/vis.py`：负责读取实验结果，生成可视化图片和简短实验报告。

## 目标

- 对比 `Persistence`、`MLP`、`LSTM` 三种模型。
- 将“跑实验”和“出图/写报告”拆开，方便重复调参后单独重画图和重写摘要。
- 默认在 `artifacts/pv_experiment_light/` 输出如下文件：
  - `metrics.csv`
  - `predictions.csv`
  - `experiment_config.json`
  - `best_lstm.pt`
  - `prediction_curve.png`
  - `report_snippet.md`

其中：

- 前四个文件由 `main.py` 生成。
- 后两个文件由 `vis.py` 生成。

## 环境配置

我使用 `mamba` 管理环境（mamba是conda的加速版本），最小依赖可以按下面安装：

```bash
mamba install python pandas numpy pytorch pillow matplotlib -c pytorch -c conda-forge
```

说明：

- `matplotlib` 主要用于出图。
- 如果没有 `matplotlib`，报告脚本会自动退回到 `Pillow` 画图，仍然会输出 `prediction_curve.png`。
- 如果要使用显卡，确保 `pytorch` 与 CUDA 版本匹配即可。

## 输入数据格式

实验脚本默认需要一个 CSV 文件，至少包含两列：

- `timestamp`
- `power`

可选附加特征：

- `irradiance`
- `temperature`
- `humidity`

也就是数据。数据我自己下载下来，放在`data`文件夹中了。并且已经对其进行了预处理。

## 运行方式

仓库中已经放了一份公开真实小数据集：

- [light_pv_id00002_201801.csv](data\processed\light_pv_id00002_201801.csv)

可以直接运行 `start.sh`，也可以手动分两步执行。现在这两个脚本已经把推荐参数写成默认值，所以最简单的方式是：

```bash
python experiments/main.py
python experiments/vis.py
```

上面这两条命令默认等价于下面这组实验配置：

```bash
python experiments/main.py \
  --data-path data/processed/light_pv_id00002_201801.csv \
  --time-col timestamp \
  --target-col power \
  --resample-rule 45min \
  --add-time-features \
  --window-size 12 \
  --horizon 2 \
  --epochs 40 \
  --batch-size 32 \
  --hidden-size 128 \
  --output-dir artifacts/pv_experiment_light \
  --device cuda

python experiments/vis.py \
  --output-dir artifacts/pv_experiment_light
```

如果设备没有GPU，可以把 `--device cuda` 改成 `--device cpu`。当然，如果不换的话也可以，会自动退回到 CPU 模式。

数据来源与整理说明见 [README.md](data/README.md)。

这组参数的含义是：

- 先把 1 分钟数据重采样为 45 分钟数据。
- 使用过去 `12` 个时间步，也就是过去 `1` 小时的数据。
- 预测未来 `2` 个时间步，也就是未来 `90` 分钟的功率。
- 自动加入昼夜周期和周内周期的时间特征。
- 使用更大的隐藏层和更小的 batch size，让 LSTM 有更充足的拟合能力。

这样的设置既保留了全天连续功率曲线，也使数据同时呈现夜间低值、白天爬升、峰值波动和回落等不同阶段的特征，从而更贴近真实光伏功率序列在较长时间尺度下的时序结构。

## 说明

实验脚本会自动识别时间间隔断点，不会跨越长缺失段去拼接训练窗口；如果你显式开启 `--filter-night`，它也不会跨越被过滤掉的夜间区段去拼接窗口。这一点对 LSTM 这类序列模型尤其重要。

报告脚本生成的 `prediction_curve.png` 会展示整段测试集的预测结果，并补充完整时间轴标签，便于结果展示。

由于数据集较小，我把数据集和实验结果都放在了 GitHub 仓库里，方便直接运行和展示。你也可以替换成自己的数据集，或者调整参数来做更多实验。

这段代码是借助 AI 生成的，不过参数选择、模型选择和最终实验设置，都是我基于对数据的理解和一些简单调试后定下来的。
