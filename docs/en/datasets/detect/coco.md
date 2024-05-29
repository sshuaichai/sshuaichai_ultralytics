---
comments: true
description: 了解COCO数据集，这是一种用于目标检测和分割的领先数据集，与Ultralytics集成。探索如何使用它来训练YOLO模型。
keywords: Ultralytics, COCO数据集, 目标检测, YOLO, YOLO模型训练, 图像分割, 计算机视觉, 深度学习模型
---

# COCO数据集

[COCO](https://cocodataset.org/#home)（Common Objects in Context）数据集是一个大规模的目标检测、分割和标注数据集。它旨在鼓励对各种对象类别的研究，通常用于计算机视觉模型的基准测试。对于从事目标检测、分割和姿势估计任务的研究人员和开发人员来说，这是一个必不可少的数据集。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/uDrn9QZJ2lk"
    title="YouTube视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics COCO数据集概述
</p>

## COCO预训练模型

| 模型                                                                                   | 尺寸<br><sup>(像素) | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>(毫秒) | 速度<br><sup>A100 TensorRT<br>(毫秒) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
|----------------------------------------------------------------------------------------|---------------------|----------------------|--------------------------------|-------------------------------------|------------------|-------------------|
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt)  | 640                 | 37.3                 | 80.4                           | 0.99                                | 3.2              | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt)  | 640                 | 44.9                 | 128.4                          | 1.20                                | 11.2             | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt)  | 640                 | 50.2                 | 234.7                          | 1.83                                | 25.9             | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt)  | 640                 | 52.9                 | 375.2                          | 2.39                                | 43.7             | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt)  | 640                 | 53.9                 | 479.1                          | 3.53                                | 68.2             | 257.8             |

## 主要特点

- COCO包含33万张图像，其中20万张图像具有目标检测、分割和标注任务的注释。
- 数据集包含80个对象类别，包括常见对象如汽车、自行车和动物，以及更具体的类别如雨伞、手提包和体育设备。
- 注释包括对象边界框、分割掩码和每张图像的标注。
- COCO提供标准化的评估指标，如用于目标检测的平均精度（mAP）和用于分割任务的平均召回率（mAR），适合比较模型性能。

## 数据集结构

COCO数据集分为三个子集：

1. **Train2017**：该子集包含118K张用于训练目标检测、分割和标注模型的图像。
2. **Val2017**：该子集包含5K张图像，用于模型训练期间的验证。
3. **Test2017**：该子集包含20K张图像，用于测试和基准测试训练后的模型。该子集的真实注释不公开，结果需提交到[COCO评估服务器](https://codalab.lisn.upsaclay.fr/competitions/7384)进行性能评估。

## 应用

COCO数据集广泛用于训练和评估目标检测（如YOLO、Faster R-CNN和SSD）、实例分割（如Mask R-CNN）和关键点检测（如OpenPose）的深度学习模型。其多样的对象类别、大量注释图像和标准化评估指标使其成为计算机视觉研究人员和从业者的重要资源。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含有关数据集路径、类别和其他相关信息。对于COCO数据集，`coco.yaml`文件位于[https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)。

!!! Example "ultralytics/cfg/datasets/coco.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco.yaml"
    ```

## 使用方法

要在COCO数据集上训练一个YOLOv8n模型100个epochs，图像尺寸为640，可以使用以下代码样例。有关可用参数的详细列表，请参阅模型的[训练](../../modes/train.md)页面。

!!! Example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n.pt')  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data='coco.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=coco.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## 示例图像和注释

COCO数据集包含各种对象类别和复杂场景的多样化图像。以下是一些数据集中的图像示例及其对应的注释：

![数据集示例图像](https://user-images.githubusercontent.com/26833433/236811818-5b566576-1e92-42fa-9462-4b6a848abe89.jpg)

- **拼接图像**：这张图像展示了由拼接数据集图像组成的训练批次。拼接是一种在训练过程中使用的技术，将多张图像组合成一张图像，以增加每个训练批次中对象和场景的多样性。这有助于提高模型在不同对象大小、纵横比和上下文中的泛化能力。

此示例展示了COCO数据集中图像的多样性和复杂性，以及在训练过程中使用拼接的好处。

## 引用和致谢

如果您在研究或开发工作中使用了COCO数据集，请引用以下论文：

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{lin2015microsoft,
              title={Microsoft COCO: Common Objects in Context},
              author={Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Dollár},
              year={2015},
              eprint={1405.0312},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

我们要感谢COCO联盟为计算机视觉社区创建并维护这一宝贵资源。有关COCO数据集及其创建者的更多信息，请访问[COCO数据集网站](https://cocodataset.org/#home)。
