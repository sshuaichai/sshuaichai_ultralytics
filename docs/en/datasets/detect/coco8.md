---
comments: true
description: 了解使用实用且多样化的COCO8数据集进行目标检测模型测试的好处。学习通过Ultralytics HUB和YOLOv8配置和使用该数据集的方法。
keywords: Ultralytics, COCO8数据集, 目标检测, 模型测试, 数据集配置, 检测方法, 健全性检查, 训练流水线, YOLOv8
---

# COCO8数据集

## 简介

[Ultralytics](https://ultralytics.com) COCO8 是一个小型但功能多样的目标检测数据集，由COCO 2017训练集的前8张图像组成，其中4张用于训练，4张用于验证。这个数据集非常适合用于测试和调试目标检测模型，或用于实验新的检测方法。由于只有8张图像，它足够小，可以轻松管理，但又足够多样化，可以测试训练流水线中的错误，并在训练更大数据集之前进行健全性检查。

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

此数据集旨在与Ultralytics [HUB](https://hub.ultralytics.com)和[YOLOv8](https://github.com/ultralytics/ultralytics)一起使用。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含有关数据集路径、类别和其他相关信息。对于COCO8数据集，`coco8.yaml`文件位于[https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml)。

!!! Example "ultralytics/cfg/datasets/coco8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8.yaml"
    ```

## 使用方法

要在COCO8数据集上训练一个YOLOv8n模型100个epochs，图像尺寸为640，可以使用以下代码样例。有关可用参数的详细列表，请参阅模型的[训练](../../modes/train.md)页面。

!!! Example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n.pt')  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=coco8.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

具体来说，这个命令的含义如下：

yolo detect train：表示使用 YOLO 模型进行目标检测任务，并且进入训练模式。
data=coco8.yaml：指定数据集的配置文件，这里使用的是 coco8.yaml。该文件通常包含数据集路径、类别等信息。
model=yolov8n.pt：指定预训练模型的路径，这里使用的是 yolov8n.pt。
epochs=100：指定训练的轮数，这里设置为 100 轮。
imgsz=640：指定输入图像的大小，这里设置为 640x640 像素。

YOLOv8 的训练配置
YOLOv8 的训练配置通常通过命令行参数和配置文件来设置。配置文件通常是一个 YAML 文件，用于指定数据集路径、类别等信息。

配置文件示例 (coco8.yaml)
```yaml
train: /path/to/train/images
val: /path/to/val/images

nc: 80
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
```

## 示例图像和注释

以下是COCO8数据集中的一些图像示例及其对应的注释：

<img src="https://user-images.githubusercontent.com/26833433/236818348-e6260a3d-0454-436b-83a9-de366ba07235.jpg" alt="数据集示例图像" width="800">

- **拼接图像**：这张图像展示了由拼接数据集图像组成的训练批次。拼接是一种在训练过程中使用的技术，将多张图像组合成一张图像，以增加每个训练批次中对象和场景的多样性。这有助于提高模型在不同对象大小、纵横比和上下文中的泛化能力。

此示例展示了COCO8数据集中图像的多样性和复杂性，以及在训练过程中使用拼接的好处。

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
