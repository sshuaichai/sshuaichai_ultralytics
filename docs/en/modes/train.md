---
comments: true
description: 逐步指南教您如何使用Ultralytics YOLO训练YOLOv8模型，包括单GPU和多GPU训练的示例。
keywords: Ultralytics, YOLOv8, YOLO, 目标检测, 训练模式, 自定义数据集, GPU训练, 多GPU, 超参数, CLI示例, Python示例
---

# 使用Ultralytics YOLO进行模型训练

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO生态系统和集成">

## 介绍

训练深度学习模型涉及向模型提供数据并调整其参数，使其能够进行准确的预测。Ultralytics YOLOv8的训练模式专为高效和有效的目标检测模型训练而设计，充分利用现代硬件能力。本指南旨在涵盖使用YOLOv8丰富的功能进行自定义模型训练所需的所有细节。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="YouTube视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何在Google Colab上训练YOLOv8模型
</p>

## 为什么选择Ultralytics YOLO进行训练？

以下是选择YOLOv8训练模式的一些令人信服的理由：

- **效率：** 最大限度地利用硬件，无论是单GPU设置还是扩展到多GPU。
- **多功能性：** 除了现成的数据集如COCO、VOC和ImageNet，还可以在自定义数据集上进行训练。
- **用户友好：** 简单但功能强大的CLI和Python接口，提供简便的训练体验。
- **超参数灵活性：** 广泛的可自定义超参数，可优化模型性能。

### 训练模式的关键特性

以下是YOLOv8训练模式的一些显著特性：

- **自动数据集下载：** 像COCO、VOC和ImageNet这样的标准数据集在首次使用时会自动下载。
- **多GPU支持：** 无缝扩展训练工作到多个GPU，加快训练进程。
- **超参数配置：** 可以通过YAML配置文件或CLI参数修改超参数。
- **可视化和监控：** 实时跟踪训练指标和可视化学习过程，以获得更好的洞察。

!!! 提示 "提示"

    * YOLOv8数据集如COCO、VOC、ImageNet等在首次使用时会自动下载，即 `yolo train data=coco.yaml`

## 使用示例

在COCO8数据集上训练YOLOv8n，训练100个epoch，图像大小为640。可以使用`device`参数指定训练设备。如果未传递任何参数，则使用GPU `device=0`，否则使用`device='cpu'`。有关完整的训练参数列表，请参见下方的参数部分。

!!! 示例 "单GPU和CPU训练示例"

    设备会自动确定。如果有可用的GPU，则会使用GPU，否则会在CPU上启动训练。

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n.yaml')  # 从YAML构建新模型
        model = YOLO('yolov8n.pt')  # 加载预训练模型（推荐用于训练）
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # 从YAML构建并传递权重

        # 训练模型
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从YAML构建新模型并从头开始训练
        yolo detect train data=coco8.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # 从预训练*.pt模型开始训练
        yolo detect train data=coco8.yaml model=yolov8n.pt epochs=100 imgsz=640

        # 从YAML构建新模型，将预训练权重传递给它并开始训练
        yolo detect train data=coco8.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### 多GPU训练

多GPU训练通过在多个GPU上分配训练负载，更高效地利用可用硬件资源。此功能可通过Python API和命令行界面使用。要启用多GPU训练，请指定要使用的GPU设备ID。

!!! 示例 "多GPU训练示例"

    要使用2个GPU进行训练，CUDA设备0和1，使用以下命令。根据需要扩展到更多GPU。

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n.pt')  # 加载预训练模型（推荐用于训练）

        # 使用2个GPU训练模型
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640, device=[0, 1])
        ```

    === "CLI"

        ```bash
        # 使用GPU 0和1从预训练*.pt模型开始训练
        yolo detect train data=coco8.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0,1
        ```

### Apple M1和M2 MPS训练

随着Ultralytics YOLO模型对Apple M1和M2芯片的支持，现在可以在利用强大的Metal Performance Shaders (MPS)框架的设备上训练模型。MPS提供了一种在Apple定制硅上执行计算和图像处理任务的高性能方式。

要在Apple M1和M2芯片上启用训练，应在启动训练过程时指定`device='mps'`。以下是如何在Python和命令行中执行此操作的示例：

!!! 示例 "MPS训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n.pt')  # 加载预训练模型（推荐用于训练）

        # 使用MPS训练模型
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640, device='mps')
        ```

    === "CLI"

        ```bash
        # 使用MPS从预训练*.pt模型开始训练
        yolo detect train data=coco8.yaml model=yolov8n.pt epochs=100 imgsz=640 device=mps
        ```

通过利用M1/M2芯片的计算能力，这使得训练任务的处理更加高效。有关更详细的指导和高级配置选项，请参阅[PyTorch MPS文档](https://pytorch.org/docs/stable/notes/mps.html)。

### 恢复中断的训练

从先前保存的状态恢复训练是处理深度学习模型时的关键特性。在训练过程意外中断时或希望使用新数据或更多epoch继续训练模型时，这种功能非常有用。

当训练恢复时，Ultralytics YOLO会加载最后保存的模型权重，并恢复优化器状态、学习率调度器和epoch号。这使得您可以从上次中断的地方无缝继续训练过程。

您可以通过在调用`train`方法时将`resume`参数设置为`True`并指定包含部分训练模型权重的`.pt`文件的路径，轻松恢复Ultralytics YOLO中的训练。

以下是如何使用Python和命令行恢复中断训练的示例：

!!! 示例 "恢复训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('path/to/last.pt')  # 加载部分训练模型

        # 恢复训练
        results = model.train(resume=True)
        ```

    === "CLI"

        ```bash
        # 恢复中断的训练
        yolo train resume model=path/to/last.pt
        ```

通过设置`resume=True`，`train`函数将从上次中断的地方继续训练，使用存储在'path/to/last.pt'文件中的状态。如果省略或设置`resume`参数为`False`，`train`函数将开始新的训练会话。

请记住，检查点默认每个epoch结束时保存一次，或者使用`save_period`参数以固定间隔保存，因此您必须至少完成一个epoch才能恢复训练。

## 训练设置

YOLO模型的训练设置涵盖了训练过程中的各种超参数和配置。这些设置影响模型的性能、速度和准确性。关键训练设置包括批量大小、学习率、动量和权重衰减。此外，优化器的选择、损失函数和训练数据集的组成也会影响训练过程。仔细调整和实验这些设置对于优化性能至关重要。

| 参数               | 默认值  | 描述                                                                                                                                                                  |
|-------------------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model`           | `None`  | 指定用于训练的模型文件。接受路径到`.pt`预训练模型或`.yaml`配置文件。定义模型结构或初始化权重必不可少。                                                                |
| `data`            | `None`  | 数据集配置文件的路径（例如`coco8.yaml`）。该文件包含数据集特定参数，包括训练和验证数据的路径、类别名称和类别数量。                                                    |
| `epochs`          | `100`   | 总训练轮数。每个epoch表示整个数据集的完整遍历。调整此值可以影响训练持续时间和模型性能。                                                                              |
| `time`            | `None`  | 最大训练时间（小时）。如果设置，则覆盖`epochs`参数，允许在指定时间后自动停止训练。对于时间受限的训练场景很有用。                                                   |
| `patience`        | `100`   | 在验证指标没有改进的情况下等待的epoch数，之后将提前停止训练。通过在性能平稳时停止训练，防止过拟合。                                                                    |
| `batch`           | `16`    | 训练的批量大小，表示在模型的内部参数更新之前处理的图像数量。AutoBatch (`batch=-1`)根据GPU内存可用性动态调整批量大小。                                               |
| `imgsz`           | `640`   | 训练的目标图像大小。所有图像在输入模型之前都调整为此尺寸。影响模型的准确性和计算复杂性。                                                                              |
| `save`            | `True`  | 启用训练检查点和最终模型权重的保存。对于恢复训练或模型部署很有用。                                                                                                   |
| `save_period`     | `-1`    | 模型检查点保存的频率，以epoch为单位。-1表示禁用此功能。对于长时间的训练会话，保存中间模型很有用。                                                                      |
| `cache`           | `False` | 启用数据集图像在内存(`True`/`ram`)或磁盘(`disk`)上的缓存，或禁用(`False`)缓存。通过减少磁盘I/O来提高训练速度，但增加内存使用。                                         |
| `device`          | `None`  | 指定训练的计算设备：单个GPU (`device=0`)、多个GPU (`device=0,1`)、CPU (`device=cpu`)或Apple硅MPS (`device=mps`)。                                                        |
| `workers`         | `8`     | 数据加载的工作线程数量（每个`RANK`，如果是多GPU训练）。影响数据预处理和模型输入的速度，特别是在多GPU设置中。                                                            |
| `project`         | `None`  | 保存训练输出的项目目录名称。允许有条理地存储不同的实验。                                                                                                              |
| `name`            | `None`  | 训练运行的名称。用于在项目文件夹中创建子目录，保存训练日志和输出。                                                                                                     |
| `exist_ok`        | `False` | 如果为True，则允许覆盖现有的项目/名称目录。对于迭代实验很有用，不需要手动清理以前的输出。                                                                               |
| `pretrained`      | `True`  | 决定是否从预训练模型开始训练。可以是布尔值或特定模型的字符串路径，用于加载权重。提高训练效率和模型性能。                                                                |
| `optimizer`       | `'auto'`| 训练优化器的选择。选项包括`SGD`、`Adam`、`AdamW`、`NAdam`、`RAdam`、`RMSProp`等，或`auto`，基于模型配置自动选择。影响收敛速度和稳定性。                               |
| `verbose`         | `False` | 启用训练期间的详细输出，提供详细日志和进度更新。对于调试和密切监控训练过程很有用。                                                                                        |
| `seed`            | `0`     | 设置训练的随机种子，确保相同配置下结果的可重复性。                                                                                                                      |
| `deterministic`   | `True`  | 强制使用确定性算法，确保可重复性，但由于限制了非确定性算法，可能影响性能和速度。                                                                                         |
| `single_cls`      | `False` | 在训练期间将多类数据集中的所有类别视为单一类别。对于二分类任务或关注目标存在而不是分类时很有用。                                                                        |
| `rect`            | `False` | 启用矩形训练，优化批次组成以最小化填充。可以提高效率和速度，但可能影响模型准确性。                                                                                      |
| `cos_lr`          | `False` | 使用余弦学习率调度器，在epoch期间按照余弦曲线调整学习率。帮助管理学习率以获得更好的收敛。                                                                              |
| `close_mosaic`    | `10`    | 在最后N个epoch中禁用马赛克数据增强，以稳定训练。设置为0可禁用此功能。                                                                                                    |
| `resume`          | `False` | 从最后保存的检查点恢复训练。自动加载模型权重、优化器状态和epoch计数，无缝继续训练。                                                                                       |
| `amp`             | `True`  | 启用自动混合精度(AMP)训练，减少内存使用并可能加快训练速度，对准确性影响最小。                                                                                              |
| `fraction`        | `1.0`   | 指定用于训练的数据集的比例。允许在资源有限的情况下使用完整数据集的子集进行训练，适用于实验。                                                                              |
| `profile`         | `False` | 启用ONNX和TensorRT速度在训练期间的分析，有助于优化模型部署。                                                                                                              |
| `freeze`          | `None`  | 冻结模型的前N层或按索引指定的层，减少可训练参数数量。适用于微调或迁移学习。                                                                                            |
| `lr0`             | `0.01`  | 初始学习率（即`SGD=1E-2`，`Adam=1E-3`）。调整此值对于优化过程至关重要，影响模型权重的更新速度。                                                                         |
| `lrf`             | `0.01`  | 最终学习率作为初始率的比例 = (`lr0 * lrf`)，与调度器一起使用以随时间调整学习率。                                                                                        |
| `momentum`        | `0.937` | SGD的动量因子或Adam优化器的beta1，影响当前更新中包含过去梯度的程度。                                                                                                     |
| `weight_decay`    | `0.0005`| L2正则化项，惩罚较大的权重以防止过拟合。                                                                                                                                |
| `warmup_epochs`   | `3.0`   | 学习率预热的epoch数，从低值逐渐增加到初始学习率，以稳定训练。                                                                                                           |
| `warmup_momentum` | `0.8`   | 预热阶段的初始动量，在预热期内逐渐调整到设定的动量。                                                                                                                   |
| `warmup_bias_lr`  | `0.1`   | 预热阶段的偏置参数学习率，帮助在初始epoch稳定模型训练。                                                                                                                  |
| `box`             | `7.5`   | 损失函数中框损失组件的权重，影响准确预测边界框坐标的重视程度。                                                                                                            |
| `cls`             | `0.5`   | 总损失函数中分类损失的权重，影响正确分类预测相对于其他组件的重要性。                                                                                                     |
| `dfl`             | `1.5`   | 分布焦点损失的权重，用于某些YOLO版本的细粒度分类。                                                                                                                        |
| `pose`            | `12.0`  | 用于姿态估计训练的模型中的姿态损失权重，影响准确预测姿态关键点的重视程度。                                                                                              |
| `kobj`            | `2.0`   | 姿态估计模型中的关键点对象损失的权重，在检测置信度和姿态准确性之间平衡。                                                                                                |
| `label_smoothing` | `0.0`   | 应用标签平滑，将硬标签软化为目标标签和标签的均匀分布的混合，可以提高泛化能力。                                                                                          |
| `nbs`             | `64`    | 用于规范化损失的名义批量大小。                                                                                                                                              |
| `overlap_mask`    | `True`  | 确定在训练期间分割掩码是否应重叠，适用于实例分割任务。                                                                                                                    |
| `mask_ratio`      | `4`     | 分割掩码的下采样比例，影响训练期间使用的掩码分辨率。                                                                                                                      |
| `dropout`         | `0.0`   | 分类任务中的正则化丢弃率，通过在训练期间随机省略单元来防止过拟合。                                                                                                       |
| `val`             | `True`  | 启用训练期间的验证，允许定期评估模型在单独数据集上的性能。                                                                                                                |
| `plots`           | `False` | 生成并保存训练和验证指标的图表以及预测示例，提供对模型性能和学习进展的可视化洞察。                                                                                        |

## 增强设置和超参数

增强技术对于通过引入训练数据的变异性来提高YOLO模型的鲁棒性和性能至关重要，帮助模型更好地泛化到未见过的数据。下表概述了每个增强参数的目的和效果：

| 参数             | 类型     | 默认值       | 范围          | 描述                                                                                                                                                                 |
|------------------|----------|--------------|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `hsv_h`          | `float`  | `0.015`      | `0.0 - 1.0`   | 通过颜色轮的一部分调整图像的色调，引入颜色变异性。帮助模型在不同光照条件下泛化。                                                                                      |
| `hsv_s`          | `float`  | `0.7`        | `0.0 - 1.0`   | 通过一定比例改变图像的饱和度，影响颜色的强度。用于模拟不同的环境条件。                                                                                                |
| `hsv_v`          | `float`  | `0.4`        | `0.0 - 1.0`   | 通过一定比例调整图像的亮度，帮助模型在各种光照条件下表现良好。                                                                                                         |
| `degrees`        | `float`  | `0.0`        | `-180 - +180` | 在指定的角度范围内随机旋转图像，提高模型识别不同方向目标的能力。                                                                                                      |
| `translate`      | `float`  | `0.1`        | `0.0 - 1.0`   | 将图像水平和垂直平移一定比例的图像大小，有助于学习检测部分可见的目标。                                                                                               |
| `scale`          | `float`  | `0.5`        | `>=0.0`       | 按增益因子缩放图像，模拟相机不同距离的目标。                                                                                                                         |
| `shear`          | `float`  | `0.0`        | `-180 - +180` | 通过指定的度数剪切图像，模拟从不同角度观看的效果。                                                                                                                    |
| `perspective`    | `float`  | `0.0`        | `0.0 - 0.001` | 对图像应用随机透视变换，增强模型理解3D空间中的目标的能力。                                                                                                            |
| `flipud`         | `float`  | `0.0`        | `0.0 - 1.0`   | 以指定的概率将图像上下翻转，增加数据变异性而不影响目标的特性。                                                                                                         |
| `fliplr`         | `float`  | `0.5`        | `0.0 - 1.0`   | 以指定的概率将图像左右翻转，对于学习对称目标和增加数据集多样性很有用。                                                                                                 |
| `bgr`            | `float`  | `0.0`        | `0.0 - 1.0`   | 以指定的概率将图像通道从RGB翻转为BGR，有助于提高对错误通道排序的鲁棒性。                                                                                              |
| `mosaic`         | `float`  | `1.0`        | `0.0 - 1.0`   | 将四个训练图像组合成一个，模拟不同的场景组合和目标交互。对于复杂场景理解非常有效。                                                                                      |
| `mixup`          | `float`  | `0.0`        | `0.0 - 1.0`   | 混合两个图像及其标签，创建合成图像。通过引入标签噪音和视觉变异性增强模型的泛化能力。                                                                                 |
| `copy_paste`     | `float`  | `0.0`        | `0.0 - 1.0`   | 将目标从一个图像复制并粘贴到另一个图像中，增加目标实例和学习目标遮挡。                                                                                               |
| `auto_augment`   | `str`    | `randaugment`| -             | 自动应用预定义的增强策略（`randaugment`、`autoaugment`、`augmix`），通过多样化视觉特征优化分类任务。                                                                   |
| `erasing`        | `float`  | `0.4`        | `0.0 - 0.9`   | 在分类训练期间随机擦除图像的一部分，鼓励模型关注不太明显的特征进行识别。                                                                                             |
| `crop_fraction`  | `float`  | `1.0`        | `0.1 - 1.0`   | 将分类图像裁剪为其大小的一部分，强调中央特征并适应目标比例，减少背景干扰。                                                                                          |

这些设置可以根据数据集和任务的具体要求进行调整。通过实验不同的值可以找到最佳的增强策略，从而获得最佳的模型性能。

!!! 信息

    有关训练增强操作的更多信息，请参阅[参考部分](../reference/data/augment.md)。

## 日志记录

在训练YOLOv8模型时，您可能希望跟踪模型性能随时间的变化。这就是日志记录的作用所在。Ultralytics的YOLO提供对三种日志记录器的支持 - Comet、ClearML和TensorBoard。

要使用日志记录器，请从上面的代码片段中选择并运行它。所选的日志记录器将被安装并初始化。

### Comet

[Comet](../integrations/comet.md)是一个平台，允许数据科学家和开发人员跟踪、比较、解释和优化实验和模型。它提供实时指标、代码差异和超参数跟踪等功能。

要使用Comet：

!!! 示例

    === "Python"

        ```python
        # pip install comet_ml
        import comet_ml

        comet_ml.init()
        ```

请记住在Comet网站上登录您的帐户并获取您的API密钥。您需要将其添加到环境变量或脚本中以记录您的实验。

### ClearML

[ClearML](https://www.clear.ml/)是一个开源平台，自动化跟踪实验并帮助高效共享资源。旨在帮助团队更高效地管理、执行和重现他们的机器学习工作。

要使用ClearML：

!!! 示例

    === "Python"

        ```python
        # pip install clearml
        import clearml

        clearml.browser_login()
        ```

运行此脚本后，您需要在浏览器中登录ClearML帐户并认证您的会话。

### TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard)是一个用于TensorFlow的可视化工具包。它允许您可视化TensorFlow图，绘制图的执行定量指标，并显示通过图的附加数据，如图像。

要在[Google Colab](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)中使用TensorBoard：

!!! 示例

    === "CLI"

        ```bash
        load_ext tensorboard
        tensorboard --logdir ultralytics/runs  # 替换为 'runs' 目录
        ```

要在本地使用TensorBoard，请运行以下命令并在http://localhost:6006/查看结果。

!!! 示例

    === "CLI"

        ```bash
        tensorboard --logdir ultralytics/runs  # 替换为 'runs' 目录
        ```

这将加载TensorBoard并将其指向保存训练日志的目录。

设置日志记录器后，您可以继续进行模型训练。所有训练指标将自动记录在您选择的平台中，您可以访问这些日志以监控模型性能随时间的变化，比较不同的模型，并确定需要改进的地方。

