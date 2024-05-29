---
comments: true
description: African Wildlife objects detection, a leading dataset for object detection in forests, integrates with Ultralytics. Discover ways to use it for training YOLO models.
keywords: Ultralytics, African Wildlife dataset, object detection, YOLO, YOLO model training, object tracking, computer vision, deep learning models, forest research, animals tracking
---

# African Wildlife Dataset

该数据集展示了通常在南非自然保护区发现的四种常见动物类，包括非洲野生动物如水牛、大象、犀牛和斑马，提供了关于它们特征的宝贵见解。对于训练计算机视觉算法来说，这个数据集有助于在各种栖息地（从动物园到森林）中识别动物，并支持野生动物研究。

## 数据集结构

非洲野生动物目标检测数据集分为三个子集：

- **训练集**：包含1052张图像，每张图像都有相应的注释。
- **验证集**：包括225张图像，每张图像都有对应的注释。
- **测试集**：包含227张图像，每张图像都有相应的注释。

## 应用

该数据集可以应用于各种计算机视觉任务，如目标检测、目标跟踪和研究。具体来说，它可以用于训练和评估模型，以识别图像中的非洲野生动物对象，这可以在野生动物保护、生态研究和自然保护区及保护区的监测工作中发挥作用。此外，它还可以作为教育资源，使学生和研究人员能够研究和了解不同动物物种的特征和行为。

## 数据集 YAML

YAML（Yet Another Markup Language）文件定义了数据集配置，包括路径、类别和其他相关细节。对于非洲野生动物数据集，`african-wildlife.yaml` 文件位于 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/african-wildlife.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/african-wildlife.yaml)。

!!! Example "ultralytics/cfg/datasets/african-wildlife.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/african-wildlife.yaml"
    ```

## 使用方法

要在非洲野生动物数据集上训练一个 YOLOv8n 模型100个epoch，图像尺寸为640，可以使用以下代码样例。有关可用参数的详细列表，请参阅模型的[训练](../../modes/train.md)页面。

!!! Example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n.pt')  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data='african-wildlife.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=african-wildlife.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

!!! Example "推理示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('path/to/best.pt')  # 加载微调后的模型

        # 使用模型进行推理
        results = model.predict("https://ultralytics.com/assets/african-wildlife-sample.jpg")
        ```

    === "CLI"

        ```bash
        # 使用微调后的 *.pt 模型进行预测
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/african-wildlife-sample.jpg"
        ```

## 示例图像和注释

非洲野生动物数据集包含各种展示不同动物物种及其自然栖息地的图像。以下是一些数据集中的图像示例，每张图像都有相应的注释。

![African wildlife dataset sample image](https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/919f8190-ccf3-4a96-a5f1-55d9eebc77ec)

- **拼接图像**：这里展示的是一个由拼接数据集图像组成的训练批次。拼接是一种训练技术，将多张图像合并为一张图像，丰富了批次的多样性。这种方法有助于增强模型在不同对象尺寸、纵横比和上下文中的泛化能力。

此示例展示了非洲野生动物数据集中图像的多样性和复杂性，强调了在训练过程中包含拼接的好处。

## 引用和致谢

该数据集已根据 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 发布。
