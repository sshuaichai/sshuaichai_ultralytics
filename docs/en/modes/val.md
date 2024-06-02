---

comments: true
description: 指南，用于验证YOLOv8模型。学习如何使用验证设置和指标评估YOLO模型的性能，包括Python和CLI示例。
keywords: Ultralytics, YOLO文档, YOLOv8, 验证, 模型评估, 超参数, 准确性, 指标, Python, CLI

---

# 使用Ultralytics YOLO进行模型验证

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO生态系统和集成">

## 简介

验证是机器学习管道中的一个关键步骤，它可以让您评估训练模型的质量。Ultralytics YOLOv8的验证模式提供了一套强大的工具和指标，用于评估目标检测模型的性能。本指南是一个完整的资源，旨在帮助您有效地使用验证模式，确保模型的准确性和可靠性。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?start=47"
    title="YouTube视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> Ultralytics模式教程：验证
</p>

## 为什么选择使用Ultralytics YOLO进行验证？

以下是使用YOLOv8验证模式的优势：

- **精确度：** 获得mAP50、mAP75和mAP50-95等精确指标，全面评估模型。
- **便利性：** 利用内置功能，记住训练设置，简化验证过程。
- **灵活性：** 使用相同或不同的数据集和图像大小验证模型。
- **超参数调优：** 使用验证指标微调模型，以提高性能。

### 验证模式的关键功能

以下是YOLOv8验证模式的一些显著功能：

- **自动设置：** 模型记住其训练配置，便于直接验证。
- **多指标支持：** 基于多种准确性指标评估模型。
- **CLI和Python API：** 可选择命令行界面或Python API进行验证，灵活便捷。
- **数据兼容性：** 与训练阶段使用的数据集以及自定义数据集无缝协作。

!!! 提示 "提示"

    * YOLOv8模型会自动记住其训练设置，因此您可以使用相同的图像大小和原始数据集轻松验证模型，只需`yolo val model=yolov8n.pt` 或 `model('yolov8n.pt').val()`

## 使用示例

验证已训练的YOLOv8n模型在COCO8数据集上的准确性。无需传递参数，因为`model`会将训练`data`和参数作为模型属性记住。请参阅下面的参数部分以获取完整的导出参数列表。

!!! 示例

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n.pt')  # 加载官方模型
        model = YOLO('path/to/best.pt')  # 加载自定义模型

        # 验证模型
        metrics = model.val()  # 无需参数，数据集和设置已记住
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # 包含每个类别的map50-95的列表
        ```

    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # 验证官方模型
        yolo detect val model=path/to/best.pt  # 验证自定义模型
        ```

## YOLO模型验证的参数

在验证YOLO模型时，可以微调多个参数以优化评估过程。这些参数控制输入图像大小、批处理和性能阈值等方面。下面是每个参数的详细说明，帮助您有效地自定义验证设置。

| 参数            | 类型     | 默认值  | 描述                                                                                                                              |
|-----------------|----------|---------|-----------------------------------------------------------------------------------------------------------------------------------|
| `data`          | `str`    | `None`  | 指定数据集配置文件的路径（例如，`coco8.yaml`）。该文件包括验证数据、类别名称和类别数量的路径。                                    |
| `imgsz`         | `int`    | `640`   | 定义输入图像的大小。所有图像在处理前都会调整到该尺寸。                                                                            |
| `batch`         | `int`    | `16`    | 设置每批次的图像数量。使用`-1`进行自动批次调整，基于GPU内存自动调整。                                                             |
| `save_json`     | `bool`   | `False` | 如果为`True`，将结果保存为JSON文件，以便进一步分析或与其他工具集成。                                                               |
| `save_hybrid`   | `bool`   | `False` | 如果为`True`，保存混合版本的标签，将原始标注与额外的模型预测相结合。                                                               |
| `conf`          | `float`  | `0.001` | 设置检测的最低置信度阈值。低于此阈值的检测将被丢弃。                                                                              |
| `iou`           | `float`  | `0.6`   | 设置非最大抑制（NMS）的交并比（IoU）阈值。帮助减少重复检测。                                                                       |
| `max_det`       | `int`    | `300`   | 限制每张图像的最大检测数量。在密集场景中使用，可防止过多的检测。                                                                   |
| `half`          | `bool`   | `True`  | 启用半精度（FP16）计算，减少内存使用，可能在不影响准确性的情况下提高速度。                                                         |
| `device`        | `str`    | `None`  | 指定验证设备（例如`cpu`，`cuda:0`等）。允许灵活使用CPU或GPU资源。                                                                   |
| `dnn`           | `bool`   | `False` | 如果为`True`，使用OpenCV DNN模块进行ONNX模型推理，提供PyTorch推理方法的替代方案。                                                   |
| `plots`         | `bool`   | `False` | 设置为`True`时，生成并保存预测与真实值的对比图，用于直观评估模型的性能。                                                           |
| `rect`          | `bool`   | `False` | 如果为`True`，使用矩形推理进行批处理，减少填充，可能提高速度和效率。                                                               |
| `split`         | `str`    | `val`   | 确定用于验证的数据集拆分（`val`，`test`或`train`）。允许灵活选择用于性能评估的数据段。                                               |

这些设置在验证过程中发挥着重要作用，允许定制和高效地评估YOLO模型。根据具体需求和资源调整这些参数，可以实现准确性和性能的最佳平衡。

### 带有参数的验证示例

以下示例展示了在Python和CLI中使用自定义参数进行YOLO模型验证。

!!! 示例

    === "Python"

        ```python
        from ultralytics import YOLO
        
        # 加载模型
        model = YOLO('yolov8n.pt')
        
        # 自定义验证设置
        validation_results = model.val(data='coco8.yaml',
                                       imgsz=640,
                                       batch=16,
                                       conf=0.25,
                                       iou=0.6,
                                       device='0')
        ```

    === "CLI"

        ```bash
        yolo val model=yolov8n.pt data=coco8.yaml imgsz=640 batch=16 conf=0.25 iou=0.6 device=0
        ```
