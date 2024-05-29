---
comments: true
description: 了解 Caltech-101 数据集及其在机器学习中的结构和用途。包括使用该数据集训练 YOLO 模型的说明。
keywords: Caltech-101, 数据集, YOLO 训练, 机器学习, 目标识别, ultralytics
---

# Caltech-101 数据集

[Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) 数据集是一个广泛使用的对象识别任务数据集，包含约9000张来自101个对象类别的图像。这些类别选择反映了各种现实世界中的物体，图像本身经过精心选择和标注，提供了一个具有挑战性的对象识别算法基准。

## 主要特征

- Caltech-101 数据集包含约9000张彩色图像，分为101个类别。
- 类别涵盖了各种各样的物体，包括动物、车辆、家居用品和人。
- 每个类别的图像数量不同，约为40到800张图像。
- 图像尺寸各异，大多数图像为中等分辨率。
- Caltech-101 广泛用于机器学习领域的训练和测试，特别是对象识别任务。

## 数据集结构

与许多其他数据集不同，Caltech-101 数据集没有正式划分为训练和测试集。用户通常根据特定需求创建自己的划分。然而，常见的做法是使用一部分随机子集图像进行训练（例如，每个类别30张图像），其余图像用于测试。

## 应用

Caltech-101 数据集广泛用于训练和评估对象识别任务中的深度学习模型，如卷积神经网络（CNNs）、支持向量机（SVMs）和各种其他机器学习算法。其种类繁多的类别和高质量的图像使其成为机器学习和计算机视觉领域研究和开发的优秀数据集。

## 使用方法

要在 Caltech-101 数据集上训练一个 YOLO 模型100个epoch，可以使用以下代码片段。有关可用参数的详细列表，请参阅模型[训练](../../modes/train.md)页面。

!!! Example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n-cls.pt')  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data='caltech101', epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=caltech101 model=yolov8n-cls.pt epochs=100 imgsz=416
        ```

## 示例图像和注释

Caltech-101 数据集包含各种物体的高质量彩色图像，提供了一个结构良好的对象识别任务数据集。以下是一些数据集中的图像示例：

![数据集示例图像](https://user-images.githubusercontent.com/26833433/239366386-44171121-b745-4206-9b59-a3be41e16089.png)

该示例展示了 Caltech-101 数据集中物体的多样性和复杂性，强调了多样化数据集对于训练健壮对象识别模型的重要性。

## 引用和致谢

如果你在研究或开发工作中使用了 Caltech-101 数据集，请引用以下论文：

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{fei2007learning,
          title={Learning generative visual models from few training examples: An incremental Bayesian approach tested on 101 object categories},
          author={Fei-Fei, Li and Fergus, Rob and Perona, Pietro},
          journal={Computer vision and Image understanding},
          volume={106},
          number={1},
          pages={59--70},
          year={2007},
          publisher={Elsevier}
        }
        ```

我们要感谢 Li Fei-Fei、Rob Fergus 和 Pietro Perona 创建并维护 Caltech-101 数据集，使其成为机器学习和计算机视觉研究社区的宝贵资源。有关 Caltech-101 数据集及其创建者的更多信息，请访问 [Caltech-101 数据集网站](https://data.caltech.edu/records/mzrjq-6wc02)。
