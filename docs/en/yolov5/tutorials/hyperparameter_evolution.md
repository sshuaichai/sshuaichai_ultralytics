# 优化 YOLOv5 的超参数进化指南

📚 本指南解释了如何为 YOLOv5 🚀 进行 **超参数进化**。超参数进化是使用 [遗传算法 (GA)](https://en.wikipedia.org/wiki/Genetic_algorithm) 进行 [超参数优化](https://en.wikipedia.org/wiki/Hyperparameter_optimization) 的方法。

机器学习中的超参数控制训练的各个方面，找到它们的最佳值可能是一个挑战。传统方法如网格搜索在面对高维搜索空间、不明维度间的相关性以及每个点的昂贵评估时，会变得不可行，因此 GA 成为超参数搜索的合适选择。

## 在开始之前

在 [**Python>=3.8.0**](https://www.python.org/) 环境中克隆代码库并安装 [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)，包括 [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)。[模型](https://github.com/ultralytics/yolov5/tree/master/models) 和 [数据集](https://github.com/ultralytics/yolov5/tree/master/data) 会从最新的 YOLOv5 [发布](https://github.com/ultralytics/yolov5/releases) 中自动下载。

```bash
git clone https://github.com/ultralytics/yolov5  # 克隆仓库
cd yolov5
pip install -r requirements.txt  # 安装依赖
```

## 1. Initialize Hyperparameters
1. 初始化超参数
YOLOv5 有大约 30 个超参数用于各种训练设置。这些定义在 /data/hyps 目录下的 *.yaml 文件中。更好的初始猜测会产生更好的最终结果，因此在进化之前正确初始化这些值非常重要。如果不确定，请使用默认值，这些值是为从头开始训练 YOLOv5 COCO 优化的。
YOLOv5 has about 30 hyperparameters used for various training settings. These are defined in `*.yaml` files in the `/data/hyps` directory. Better initial guesses will produce better final results, so it is important to initialize these values properly before evolving. If in doubt, simply use the default values, which are optimized for YOLOv5 COCO training from scratch.

```yaml
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# 超参数用于从头开始低增强 COCO 训练
# python train.py --batch 64 --cfg yolov5n6.yaml --weights '' --data coco.yaml --img 640 --epochs 300 --linear
# 有关超参数进化的教程，请参见 https://github.com/ultralytics/yolov5#tutorials

lr0: 0.01  # 初始学习率 (SGD=1E-2, Adam=1E-3)
lrf: 0.01  # 最终 OneCycleLR 学习率 (lr0 * lrf)
momentum: 0.937  # SGD 动量/Adam beta1
weight_decay: 0.0005  # 优化器权重衰减 5e-4
warmup_epochs: 3.0  # 预热 epoch（分数也可以）
warmup_momentum: 0.8  # 预热初始动量
warmup_bias_lr: 0.1  # 预热初始偏置 lr
box: 0.05  # 边框损失增益
cls: 0.5  # 类别损失增益
cls_pw: 1.0  # 类别 BCELoss 正权重
obj: 1.0  # 对象损失增益（随像素缩放）
obj_pw: 1.0  # 对象 BCELoss 正权重
iou_t: 0.20  # IoU 训练阈值
anchor_t: 4.0  # 锚定多重阈值
# anchors: 3  # 每个输出层的锚点（0 表示忽略）
fl_gamma: 0.0  # 焦点损失 gamma (efficientDet 默认 gamma=1.5)
hsv_h: 0.015  # 图像 HSV-Hue 增强（分数）
hsv_s: 0.7  # 图像 HSV-Saturation 增强（分数）
hsv_v: 0.4  # 图像 HSV-Value 增强（分数）
degrees: 0.0  # 图像旋转（+/- 度）
translate: 0.1  # 图像平移（+/- 分数）
scale: 0.5  # 图像缩放（+/- 增益）
shear: 0.0  # 图像剪切（+/- 度）
perspective: 0.0  # 图像透视（+/- 分数），范围 0-0.001
flipud: 0.0  # 图像上下翻转（概率）
fliplr: 0.5  # 图像左右翻转（概率）
mosaic: 1.0  # 图像马赛克（概率）
mixup: 0.0  # 图像混合（概率）
copy_paste: 0.0  # 段落复制粘贴（概率）

```

## 2. Define Fitness
2. 定义适应度
适应度是我们要最大化的值。在 YOLOv5 中，我们将默认适应度函数定义为指标的加权组合：mAP@0.5 占权重的 10%，mAP@0.5:0.95 占剩余的 90%，不包括 精度 P 和召回率 R。你可以根据需要调整这些，或者使用 utils/metrics.py 中的默认适应度定义（推荐）。
Fitness is the value we seek to maximize. In YOLOv5 we define a default fitness function as a weighted combination of metrics: `mAP@0.5` contributes 10% of the weight and `mAP@0.5:0.95` contributes the remaining 90%, with [Precision `P` and Recall `R`](https://en.wikipedia.org/wiki/Precision_and_recall) absent. You may adjust these as you see fit or use the default fitness definition in utils/metrics.py (recommended).

```python
def fitness(x):
    """通过对加权指标 [P, R, mAP@0.5, mAP@0.5:0.95] 求和来评估模型的适应度，x 是形状为 (n, 4) 的 numpy 数组。"""
    w = [0.0, 0.0, 0.1, 0.9]  # [P, R, mAP@0.5, mAP@0.5:0.95] 的权重
    return (x[:, :4] * w).sum(1)
```

## 3. Evolve
3. 进化
进化是在我们想要改进的基本情景下进行的。在本例中，基本情景是使用预训练的 YOLOv5s 对 COCO128 进行 10 轮微调。基本情景的训练命令是：
Evolution is performed about a base scenario which we seek to improve upon. The base scenario in this example is finetuning COCO128 for 10 epochs using pretrained YOLOv5s. The base scenario training command is:

```bash
python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache
```
要 针对该情景 进行超参数进化，从 第 1 部分 中定义的初始值开始，并最大化 第 2 部分 中定义的适应度，请添加 --evolve：
To evolve hyperparameters **specific to this scenario**, starting from our initial values defined in **Section 1.**, and maximizing the fitness defined in **Section 2.**, append `--evolve`:

```bash
# 单 GPU
python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --evolve

# 多 GPU
for i in 0 1 2 3 4 5 6 7; do
  sleep $(expr 30 \* $i) &&  # 30 秒延迟（可选）
  echo 'Starting GPU '$i'...' &&
  nohup python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --device $i --evolve > evolve_gpu_$i.log &
done

# 多 GPU bash-while（不推荐）
for i in 0 1 2 3 4 5 6 7; do
  sleep $(expr 30 \* $i) &&  # 30 秒延迟（可选）
  echo 'Starting GPU '$i'...' &&
  "$(while true; do nohup python train.py... --device $i --evolve 1 > evolve_gpu_$i.log; done)" &
done

```
默认的进化设置将运行基本情景 300 次，即 300 代。你可以通过 --evolve 参数修改代数，即 python train.py --evolve 1000。
The default evolution settings will run the base scenario 300 times, i.e. for 300 generations. You can modify generations via the `--evolve` argument, i.e. `python train.py --evolve 1000`.

主要的遗传操作是 交叉 和 变异。在此工作中使用变异，以 80% 的概率和 0.04 的方差基于之前所有代中最佳父代的组合创建新后代。结果记录在 runs/evolve/exp/evolve.csv，并在每一代中保存最高适应度的后代为 runs/evolve/hyp_evolved.yaml：
The main genetic operators are **crossover** and **mutation**. In this work mutation is used, with an 80% probability and a 0.04 variance to create new offspring based on a combination of the best parents from all previous generations. Results are logged to `runs/evolve/exp/evolve.csv`, and the highest fitness offspring is saved every generation as `runs/evolve/hyp_evolved.yaml`:

```yaml
# YOLOv5 Hyperparameter Evolution Results
# Best generation: 287
# Last generation: 300
#    metrics/precision,       metrics/recall,      metrics/mAP_0.5, metrics/mAP_0.5:0.95,         val/box_loss,         val/obj_loss,         val/cls_loss
#              0.54634,              0.55625,              0.58201,              0.33665,             0.056451,             0.042892,             0.013441

# YOLOv5 超参数进化结果
# 最佳代数: 287
# 最后代数: 300
#    指标/精度,       指标/召回率,      指标/mAP_0.5, 指标/mAP_0.5:0.95,         验证/框损失,         验证/目标损失,         验证/分类损失
#              0.54634,              0.55625,              0.58201,              0.33665,             0.056451,             0.042892,             0.013441

lr0: 0.01  # 初始学习率 (SGD=1E-2, Adam=1E-3)
lrf: 0.2  # 最终OneCycleLR学习率 (lr0 * lrf)
momentum: 0.937  # SGD动量/Adam beta1
weight_decay: 0.0005  # 优化器权重衰减 5e-4
warmup_epochs: 3.0  # 预热世代（可以是小数）
warmup_momentum: 0.8  # 预热初始动量
warmup_bias_lr: 0.1  # 预热初始偏置学习率
box: 0.05  # 框损失增益
cls: 0.5  # 分类损失增益
cls_pw: 1.0  # 分类BCELoss正权重
obj: 1.0  # 目标损失增益（随像素缩放）
obj_pw: 1.0  # 目标BCELoss正权重
iou_t: 0.20  # 训练IoU阈值
anchor_t: 4.0  # 锚点多重阈值
# anchors: 3  # 每个输出层的锚点（0表示忽略）
fl_gamma: 0.0  # focal损失gamma（efficientDet默认gamma=1.5）
hsv_h: 0.015  # 图像HSV-色调增强（比例）
hsv_s: 0.7  # 图像HSV-饱和度增强（比例）
hsv_v: 0.4  # 图像HSV-明度增强（比例）
degrees: 0.0  # 图像旋转（正负度）
translate: 0.1  # 图像平移（正负比例）
scale: 0.5  # 图像缩放（正负增益）
shear: 0.0  # 图像剪切（正负度）
perspective: 0.0  # 图像透视（正负比例），范围0-0.001
flipud: 0.0  # 图像上下翻转（概率）
fliplr: 0.5  # 图像左右翻转（概率）
mosaic: 1.0  # 图像拼接（概率）
mixup: 0.0  # 图像混合（概率）
copy_paste: 0.0  # 分段复制粘贴（概率）

```
我们建议进行至少300代的进化以获得最佳结果。请注意，进化通常是昂贵且耗时的，因为基本场景需要训练数百次，可能需要数百甚至数千个GPU小时。
We recommend a minimum of 300 generations of evolution for best results. Note that **evolution is generally expensive and time-consuming**, as the base scenario is trained hundreds of times, possibly requiring hundreds or thousands of GPU hours.

## 4. Visualize
4. 可视化
evolve.csv在进化结束后由utils.plots.plot_evolve()绘制为evolve.png，每个超参数一个子图，显示适应度（y轴）与超参数值（x轴）的关系。黄色表示较高浓度。垂直分布表示一个参数已禁用且不变异。这在train.py中的meta字典中可由用户选择，适用于固定参数并防止它们进化。
`evolve.csv` is plotted as `evolve.png` by `utils.plots.plot_evolve()` after evolution finishes with one subplot per hyperparameter showing fitness (y-axis) vs hyperparameter values (x-axis). Yellow indicates higher concentrations. Vertical distributions indicate that a parameter has been disabled and does not mutate. This is user selectable in the `meta` dictionary in train.py, and is useful for fixing parameters and preventing them from evolving.

![evolve](https://user-images.githubusercontent.com/26833433/89130469-f43e8e00-d4b9-11ea-9e28-f8ae3622516d.png)

## Supported Environments

Ultralytics provides a range of ready-to-use environments, each pre-installed with essential dependencies such as [CUDA](https://developer.nvidia.com/cuda), [CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/), and [PyTorch](https://pytorch.org/), to kickstart your projects.

- **Free GPU Notebooks**: <a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
- **Google Cloud**: [GCP Quickstart Guide](../environments/google_cloud_quickstart_tutorial.md)
- **Amazon**: [AWS Quickstart Guide](../environments/aws_quickstart_tutorial.md)
- **Azure**: [AzureML Quickstart Guide](../environments/azureml_quickstart_tutorial.md)
- **Docker**: [Docker Quickstart Guide](../environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## Project Status

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>

This badge indicates that all [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) Continuous Integration (CI) tests are successfully passing. These CI tests rigorously check the functionality and performance of YOLOv5 across various key aspects: [training](https://github.com/ultralytics/yolov5/blob/master/train.py), [validation](https://github.com/ultralytics/yolov5/blob/master/val.py), [inference](https://github.com/ultralytics/yolov5/blob/master/detect.py), [export](https://github.com/ultralytics/yolov5/blob/master/export.py), and [benchmarks](https://github.com/ultralytics/yolov5/blob/master/benchmarks.py). They ensure consistent and reliable operation on macOS, Windows, and Ubuntu, with tests conducted every 24 hours and upon each new commit.
