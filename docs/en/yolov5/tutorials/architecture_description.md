---
comments: true
description: 探索Ultralytics的目标检测算法YOLOv5的架构。了解模型结构、数据增强方法、训练策略和损失计算技术。
keywords: Ultralytics, YOLOv5, 目标检测, 架构, 模型结构, 数据增强, 训练策略, 损失计算
---

# Ultralytics YOLOv5 架构

YOLOv5（v6.0/6.1）是由Ultralytics开发的强大目标检测算法。本文深入探讨YOLOv5的架构、数据增强策略、训练方法和损失计算技术。通过全面了解这些内容，将有助于提升在监控、自动驾驶和图像识别等各个领域的目标检测实际应用能力。

## 1. 模型结构

YOLOv5的架构由三个主要部分组成：

- **Backbone**：这是网络的主要部分。对于YOLOv5，Backbone采用的是`New CSP-Darknet53`结构，这是对以前版本中使用的Darknet架构的修改。
- **Neck**：连接Backbone和Head的部分。在YOLOv5中，使用了`SPPF`和`New CSP-PAN`结构。
- **Head**：负责生成最终输出的部分。YOLOv5使用`YOLOv3 Head`来完成这个任务。

模型的结构如下面的图片所示。模型结构的详细信息可以在`yolov5l.yaml`中找到。

![yolov5](https://user-images.githubusercontent.com/31005897/172404576-c260dcf9-76bb-4bc8-b6a9-f2d987792583.png)

YOLOv5相比前代版本引入了一些细微的变化：

1. 早期版本中的`Focus`结构被`6x6 Conv2d`结构取代。这个变化提高了效率[#4825](https://github.com/ultralytics/yolov5/issues/4825)。
2. `SPP`结构被`SPPF`取代。这一改变使处理速度提高了一倍以上。

要测试`SPP`和`SPPF`的速度，可以使用以下代码：

<details>
<summary>SPP vs SPPF 速度分析示例（点击展开）</summary>

```python
import time
import torch
import torch.nn as nn


class SPP(nn.Module):
    def __init__(self):
        """初始化具有三种不同大小最大池化层的SPP模块。"""
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(5, 1, padding=2)
        self.maxpool2 = nn.MaxPool2d(9, 1, padding=4)
        self.maxpool3 = nn.MaxPool2d(13, 1, padding=6)

    def forward(self, x):
        """对输入`x`应用三个最大池化层，并沿通道维度连接结果。"""
        o1 = self.maxpool1(x)
        o2 = self.maxpool2(x)
        o3 = self.maxpool3(x)
        return torch.cat([x, o1, o2, o3], dim=1)


class SPPF(nn.Module):
    def __init__(self):
        """初始化具有特定配置的MaxPool2d层的SPPF模块。"""
        super().__init__()
        self.maxpool = nn.MaxPool2d(5, 1, padding=2)

    def forward(self, x):
        """应用顺序最大池化，并与输入张量连接结果；期望输入张量x的任何形状。"""
        o1 = self.maxpool(x)
        o2 = self.maxpool(o1)
        o3 = self.maxpool(o2)
        return torch.cat([x, o1, o2, o3], dim=1)


def main():
    """比较SPP和SPPF在随机张量(8, 32, 16, 16)上的输出和性能。"""
    input_tensor = torch.rand(8, 32, 16, 16)
    spp = SPP()
    sppf = SPPF()
    output1 = spp(input_tensor)
    output2 = sppf(input_tensor)

    print(torch.equal(output1, output2))

    t_start = time.time()
    for _ in range(100):
        spp(input_tensor)
    print(f"SPP time: {time.time() - t_start}")

    t_start = time.time()
    for _ in range(100):
        sppf(input_tensor)
    print(f"SPPF time: {time.time() - t_start}")


if __name__ == '__main__':
    main()

```

result:

```
True
SPP time: 0.5373051166534424
SPPF time: 0.20780706405639648
```

</details>

## 2. Data Augmentation Techniques
2. 数据增强技术
YOLOv5采用了多种数据增强技术，以提高模型的泛化能力并减少过拟合。这些技术包括：
YOLOv5 employs various data augmentation techniques to improve the model's ability to generalize and reduce overfitting. These techniques include:

- **Mosaic Augmentation**: An image processing technique that combines four training images into one in ways that encourage object detection models to better handle various object scales and translations.
Mosaic 增强：一种图像处理技术，将四张训练图像组合成一张，鼓励目标检测模型更好地处理不同尺度和位移的目标。

  ![mosaic](https://user-images.githubusercontent.com/31005897/159109235-c7aad8f2-1d4f-41f9-8d5f-b2fde6f2885e.png)

- **Copy-Paste Augmentation**: An innovative data augmentation method that copies random patches from an image and pastes them onto another randomly chosen image, effectively generating a new training sample.
Copy-Paste 增强：一种创新的数据增强方法，将图像中的随机补丁复制并粘贴到另一张随机选择的图像上，从而有效地生成新的训练样本。

  ![copy-paste](https://user-images.githubusercontent.com/31005897/159116277-91b45033-6bec-4f82-afc4-41138866628e.png)

- **Random Affine Transformations**: This includes random rotation, scaling, translation, and shearing of the images.
随机仿射变换：包括图像的随机旋转、缩放、平移和剪切。

  ![random-affine](https://user-images.githubusercontent.com/31005897/159109326-45cd5acb-14fa-43e7-9235-0f21b0021c7d.png)

- **MixUp Augmentation**: A method that creates composite images by taking a linear combination of two images and their associated labels.
MixUp 增强：通过对两张图像及其关联标签进行线性组合，生成复合图像的方法。

  ![mixup](https://user-images.githubusercontent.com/31005897/159109361-3b24333b-f481-478b-ae00-df7838f0b5cd.png)

- **Albumentations**: A powerful library for image augmenting that supports a wide variety of augmentation techniques.
Albumentations：一个强大的图像增强库，支持多种增强技术。

- **HSV Augmentation**: Random changes to the Hue, Saturation, and Value of the images.
HSV 增强：对图像的色调、饱和度和明度进行随机变化。

  ![hsv](https://user-images.githubusercontent.com/31005897/159109407-83d100ba-1aba-4f4b-aa03-4f048f815981.png)

- **Random Horizontal Flip**: An augmentation method that randomly flips images horizontally.
随机水平翻转：一种随机水平翻转图像的增强方法。

  ![horizontal-flip](https://user-images.githubusercontent.com/31005897/159109429-0d44619a-a76a-49eb-bfc0-6709860c043e.png)

## 3. Training Strategies
3. 训练策略
YOLOv5采用了多种复杂的训练策略来提高模型的性能。这些策略包括：

多尺度训练：在训练过程中，输入图像会在原始大小的0.5到1.5倍范围内随机缩放。
AutoAnchor：这种策略优化了先验锚框，以匹配自定义数据中真实框的统计特性。
预热和余弦学习率调度：一种调整学习率的方法，以提高模型性能。
指数移动平均（EMA）：使用过去步骤的参数平均值来稳定训练过程并减少泛化误差的策略。
混合精度训练：一种在半精度格式下执行操作的方法，减少内存使用并提高计算速度。
超参数进化：自动调节超参数以实现最佳性能的策略。
YOLOv5 applies several sophisticated training strategies to enhance the model's performance. They include:

- **Multiscale Training**: The input images are randomly rescaled within a range of 0.5 to 1.5 times their original size during the training process.
- **AutoAnchor**: This strategy optimizes the prior anchor boxes to match the statistical characteristics of the ground truth boxes in your custom data.
- **Warmup and Cosine LR Scheduler**: A method to adjust the learning rate to enhance model performance.
- **Exponential Moving Average (EMA)**: A strategy that uses the average of parameters over past steps to stabilize the training process and reduce generalization error.
- **Mixed Precision Training**: A method to perform operations in half-precision format, reducing memory usage and enhancing computational speed.
- **Hyperparameter Evolution**: A strategy to automatically tune hyperparameters to achieve optimal performance.

## 4. Additional Features

### 4.1 Compute Losses
4. 其他特性
4.1 损失计算
YOLOv5的损失是三个独立损失组件的组合：

类别损失（BCE Loss）：二元交叉熵损失，衡量分类任务的误差。
目标性损失（BCE Loss）：另一个二元交叉熵损失，计算检测某个网格单元中是否存在目标的误差。
位置损失（CIoU Loss）：完全IoU损失，衡量在网格单元内定位目标的误差。
The loss in YOLOv5 is computed as a combination of three individual loss components:

- **Classes Loss (BCE Loss)**: Binary Cross-Entropy loss, measures the error for the classification task.
- **Objectness Loss (BCE Loss)**: Another Binary Cross-Entropy loss, calculates the error in detecting whether an object is present in a particular grid cell or not.
- **Location Loss (CIoU Loss)**: Complete IoU loss, measures the error in localizing the object within the grid cell.

The overall loss function is depicted by:
总体损失函数表示为：
![loss](https://latex.codecogs.com/svg.image?Loss=\lambda_1L_{cls}+\lambda_2L_{obj}+\lambda_3L_{loc})

### 4.2 Balance Losses
4.2 损失平衡
三个预测层（P3，P4，P5）的目标性损失权重不同。平衡权重分别为[4.0, 1.0, 0.4]。这种方法确保不同尺度的预测对总损失的贡献适当。
The objectness losses of the three prediction layers (`P3`, `P4`, `P5`) are weighted differently. The balance weights are `[4.0, 1.0, 0.4]` respectively. This approach ensures that the predictions at different scales contribute appropriately to the total loss.

![obj_loss](https://latex.codecogs.com/svg.image?L_{obj}=4.0\cdot&space;L_{obj}^{small}+1.0\cdot&space;L_{obj}^{medium}+0.4\cdot&space;L_{obj}^{large})

### 4.3 Eliminate Grid Sensitivity
4.3 消除网格敏感性
YOLOv5架构对比早期版本的YOLO，在框预测策略上做了一些重要改动。在YOLOv2和YOLOv3中，框坐标是通过最后一层的激活值直接预测的。
The YOLOv5 architecture makes some important changes to the box prediction strategy compared to earlier versions of YOLO. In YOLOv2 and YOLOv3, the box coordinates were directly predicted using the activation of the last layer.

![b_x](<https://latex.codecogs.com/svg.image?b_x=\sigma(t_x)+c_x>)
![b_y](<https://latex.codecogs.com/svg.image?b_y=\sigma(t_y)+c_y>)
![b_w](https://latex.codecogs.com/svg.image?b_w=p_w\cdot&space;e^{t_w})
![b_h](https://latex.codecogs.com/svg.image?b_h=p_h\cdot&space;e^{t_h})

<img src="https://user-images.githubusercontent.com/31005897/158508027-8bf63c28-8290-467b-8a3e-4ad09235001a.png#pic_center" width=40% alt="YOLOv5 grid computation">

However, in YOLOv5, the formula for predicting the box coordinates has been updated to reduce grid sensitivity and prevent the model from predicting unbounded box dimensions.
然而，在YOLOv5中，预测框坐标的公式进行了更新，以减少网格敏感性并防止模型预测无限制的框尺寸。

The revised formulas for calculating the predicted bounding box are as follows:
更新后的预测边界框公式如下：

![bx](<https://latex.codecogs.com/svg.image?b_x=(2\cdot\sigma(t_x)-0.5)+c_x>)
![by](<https://latex.codecogs.com/svg.image?b_y=(2\cdot\sigma(t_y)-0.5)+c_y>)
![bw](<https://latex.codecogs.com/svg.image?b_w=p_w\cdot(2\cdot\sigma(t_w))^2>)
![bh](<https://latex.codecogs.com/svg.image?b_h=p_h\cdot(2\cdot\sigma(t_h))^2>)

对比缩放前后的中心点偏移。中心点偏移范围从(0, 1)调整到(-0.5, 1.5)。因此，偏移可以很容易地得到0或1。
Compare the center point offset before and after scaling. The center point offset range is adjusted from (0, 1) to (-0.5, 1.5). Therefore, offset can easily get 0 or 1.

<img src="https://user-images.githubusercontent.com/31005897/158508052-c24bc5e8-05c1-4154-ac97-2e1ec71f582e.png#pic_center" width=40% alt="YOLOv5 grid scaling">
对比调整前后相对于锚框的高度和宽度缩放比。原始yolo/darknet框方程存在严重缺陷。宽度和高度完全不受限制，因为它们只是out=exp(in)，这很危险，可能导致梯度失控、不稳定性、NaN损失以及最终的训练。
Compare the height and width scaling ratio(relative to anchor) before and after adjustment. The original yolo/darknet box equations have a serious flaw. Width and Height are completely unbounded as they are simply out=exp(in), which is dangerous, as it can lead to runaway gradients, instabilities, NaN losses and ultimately a complete loss of training. [refer this issue](https://github.com/ultralytics/yolov5/issues/471#issuecomment-662009779)

<img src="https://user-images.githubusercontent.com/31005897/158508089-5ac0c7a3-6358-44b7-863e-a6e45babb842.png#pic_center" width=40% alt="YOLOv5 unbounded scaling">

### 4.4 构建目标

在YOLOv5中，构建目标过程对训练效率和模型准确性至关重要。它涉及将真实框分配到输出图中的适当网格单元，并与适当的锚框匹配。

此过程包括以下步骤：

- 计算真实框尺寸与每个锚模板尺寸的比率。

![rw](https://latex.codecogs.com/svg.image?r_w=w_{gt}/w_{at})

![rh](https://latex.codecogs.com/svg.image?r_h=h_{gt}/h_{at})

![rwmax](<https://latex.codecogs.com/svg.image?r_w^{max}=max(r_w,1/r_w)>)

![rhmax](<https://latex.codecogs.com/svg.image?r_h^{max}=max(r_h,1/r_h)>)

![rmax](<https://latex.codecogs.com/svg.image?r^{max}=max(r_w^{max},r_h^{max})>)

![match](https://latex.codecogs.com/svg.image?r^{max}<{\rm&space;anchor_t})

<img src="https://user-images.githubusercontent.com/31005897/158508119-fbb2e483-7b8c-4975-8e1f-f510d367f8ff.png#pic_center" width=70% alt="YOLOv5 IoU computation">

- 如果计算的比率在阈值范围内，则将真实框与相应的锚框匹配。

<img src="https://user-images.githubusercontent.com/31005897/158508771-b6e7cab4-8de6-47f9-9abf-cdf14c275dfe.png#pic_center" width=70% alt="YOLOv5 grid overlap">

- 将匹配的锚框分配到适当的单元中，考虑到修订后的中心点偏移，一个真实框可以分配给多个锚框。由于中心点偏移范围从(0, 1)调整到(-0.5, 1.5)，真实框可以分配给更多的锚框。

<img src="https://user-images.githubusercontent.com/31005897/158508139-9db4e8c2-cf96-47e0-bc80-35d11512f296.png#pic_center" width=70% alt="YOLOv5 anchor selection">

通过这种方式，构建目标过程确保每个真实对象在训练过程中正确分配和匹配，从而使YOLOv5更有效地学习目标检测任务。

## 结论

总之，YOLOv5在实时目标检测模型的发展中迈出了重要一步。通过引入各种新特性、增强功能和训练策略，它在性能和效率方面超越了YOLO家族的以前版本。

YOLOv5的主要增强包括使用动态架构、广泛的数据增强技术、创新的训练策略，以及计算损失和构建目标过程中的重要调整。所有这些创新显著提高了目标检测的准确性和效率，同时保持了YOLO模型的高速特性。

