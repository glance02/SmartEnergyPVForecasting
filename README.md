# Photovoltaic Forecasting Experiment

这个仓库提供我在智慧能源概论课程报告的小实验代码，用于做光伏短时功率预测，并输出相应的结果。

## 目标

- 对比 `Persistence`、`MLP`、`LSTM` 三种模型。
- 完成从数据读取、预处理、训练、评估到出图的完整流程。
- 自动生成以下文件到默认目录 `artifacts/pv_experiment/`：
  - `metrics.csv`
  - `predictions.csv`
  - `prediction_curve.png`
  - `report_snippet.md`
  - `best_lstm.pt`

## 建议依赖

我使用 `mamba` 管理环境，最小依赖可以按下面安装：

```bash
mamba install python pandas numpy pytorch pillow matplotlib -c pytorch -c conda-forge
```

说明：

- `matplotlib` 主要用于出图。
- 如果没有 `matplotlib`，脚本会自动退回到 `Pillow` 画图，仍然会输出 `prediction_curve.png`。
- 如果要使用显卡，确保 `pytorch` 与 CUDA 版本匹配即可。

## 输入数据格式

脚本默认需要一个 CSV 文件，至少包含两列：

- `timestamp`
- `power`

可选附加特征：

- `irradiance`
- `temperature`
- `humidity`

## 运行方式

### 直接使用仓库里已经下载好的真实数据

仓库中已经放了一份公开真实小数据集，路径是：

- [light_pv_id00002_201801.csv](data\processed\light_pv_id00002_201801.csv)

推荐直接这样跑：

```bash
python experiments/pv_forecast.py \
  --data-path data/processed/light_pv_id00002_201801.csv \
  --time-col timestamp \
  --target-col power \
  --resample-rule 15min \
  --add-time-features \
  --window-size 20 \
  --horizon 2 \
  --epochs 40 \
  --batch-size 32 \
  --hidden-size 128 \
  --output-dir artifacts/pv_experiment_light \
  --device cuda
```

如果使用 CPU，可以把 `--device cuda` 改成 `--device cpu`，或者直接不传这个参数，因为默认就是 `auto`，会自动选择可用的设备。

数据来源与整理说明见 [README.md](data/README.md)。

这组参数的含义是：

- 先把 1 分钟数据重采样为 15 分钟数据。
- 使用过去 `20` 个时间步，也就是过去 `5` 小时的数据。
- 预测未来 `2` 个时间步，也就是未来 `30` 分钟的功率。
- 自动加入昼夜周期和周内周期的时间特征。
- 使用更大的隐藏层和更小的 batch size，让 LSTM 有更充足的拟合能力。

这样做的原因是：对原始 1 分钟单变量数据而言，`Persistence` 基线会非常强，深度学习模型不容易体现优势；而在 `15min` 重采样后的 `30` 分钟 ahead 设置下，图形会比 `1` 小时 ahead 更平滑，也更适合课程报告展示。当前推荐参数是基于这份真实数据做过一轮简单调参后选出来的版本。

## 参数说明

- `--data-path`：真实数据 CSV 路径。
- `--time-col`：时间列名，默认 `timestamp`。
- `--target-col`：目标列名，默认 `power`。
- `--feature-cols`：额外特征列，多个特征用逗号分隔。
- `--window-size`：历史窗口长度，默认 `8`。
- `--horizon`：预测步长，默认 `1`。
- `--epochs`：训练轮数，默认 `40`。
- `--batch-size`：批大小，默认 `64`。
- `--hidden-size`：隐藏层宽度，默认 `64`。
- `--resample-rule`：重采样规则，例如 `15min`、`30min`、`1h`。
- `--add-time-features`：自动加入时间周期特征，如昼夜周期和周内周期。
- `--filter-night`：可选开启夜间低功率区段过滤；默认关闭，也就是保留完整昼夜序列。
- `--device`：`auto`、`cpu` 或 `cuda`。
- `--output-dir`：输出目录，默认 `artifacts/pv_experiment`。
- `--demo-synthetic`：启用内置 synthetic 数据。

脚本会自动识别时间间隔的断点，不会跨越长缺失段去拼接训练窗口；如果你显式开启 `--filter-night`，它也不会跨越被过滤掉的夜间区段去拼接窗口。这一点对 LSTM 这类序列模型尤其重要。

默认生成的 `prediction_curve.png` 会展示整段测试集的预测结果，并补充完整的时间轴标签，便于直接用于课程报告展示。

