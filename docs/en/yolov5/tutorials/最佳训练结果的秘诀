---
comments: true
description: 本综合指南提供了如何训练您的YOLOv5系统以获得最佳mAP的见解。掌握数据集准备、模型选择、训练设置等内容。
keywords: Ultralytics, YOLOv5, 训练指南, 数据集准备, 模型选择, 训练设置, mAP结果, 机器学习, 目标检测
---

📚 本指南解释了如何使用YOLOv5 🚀 产生最佳的mAP和训练结果。

大多数时候，如果您的数据集足够大且标注良好，即使不更改模型或训练设置，也可以获得良好的结果。**如果一开始没有得到好的结果，可以采取一些步骤进行改进，但我们总是建议用户**首先使用所有默认设置进行训练**，然后再考虑任何更改。这有助于建立性能基准并发现改进的地方。

如果对您的训练结果有疑问，**我们建议您提供尽可能多的信息**，以期得到有帮助的回复，包括结果图（训练损失、验证损失、P、R、mAP）、PR曲线、混淆矩阵、训练马赛克、测试结果和数据集统计图像如labels.png。这些都位于您的 `project/name` 目录下，通常是 `yolov5/runs/train/exp`。

我们为希望在YOLOv5训练中获得最佳结果的用户整理了一份完整指南。

## 数据集

- **每个类别的图像数。** 推荐每个类别≥1500张图像
- **每个类别的实例数。** 推荐每个类别≥10000个实例（标注对象）
- **图像多样性。** 必须代表实际部署环境。对于现实世界的用例，我们建议来自不同时间、不同季节、不同天气、不同光照、不同角度、不同来源（在线抓取、本地收集、不同相机）的图像等。
- **标签一致性。** 所有图像中所有类别的所有实例必须标注。部分标注将不起作用。
- **标签准确性。** 标签必须紧密包围每个对象。对象与边界框之间不应有空隙。所有对象都应有标签。
- **标签验证。** 在训练开始时查看 `train_batch*.jpg` 以验证您的标签是否正确显示，即查看[示例](./train_custom_data.md#local-logging)马赛克。
- **背景图像。** 背景图像是没有对象的图像，添加到数据集中以减少误报（FP）。我们建议大约0-10％的背景图像以帮助减少FP（COCO有1000张背景图像作为参考，占总量的1％）。背景图像不需要标签。

<a href="https://arxiv.org/abs/1405.0312"><img width="800" src="https://user-images.githubusercontent.com/26833433/109398377-82b0ac00-78f1-11eb-9c76-cc7820669d0d.png" alt="COCO分析"></a>

## 模型选择

较大的模型如YOLOv5x和[YOLOv5x6](https://github.com/ultralytics/yolov5/releases/tag/v5.0)在几乎所有情况下都会产生更好的结果，但它们有更多的参数，训练时需要更多的CUDA内存，并且运行速度较慢。对于**移动端**部署，我们推荐YOLOv5s/m，对于**云端**部署，我们推荐YOLOv5l/x。请参阅我们的README[表格](https://github.com/ultralytics/yolov5#pretrained-checkpoints)以获得所有模型的完整比较。

<p align="center"><img width="700" alt="YOLOv5模型" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/model_comparison.png"></p>

- **从预训练权重开始。** 推荐用于中小型数据集（即 [VOC](https://github.com/ultralytics/yolov5/blob/master/data/VOC.yaml)、[VisDrone](https://github.com/ultralytics/yolov5/blob/master/data/VisDrone.yaml)、[GlobalWheat](https://github.com/ultralytics/yolov5/blob/master/data/GlobalWheat2020.yaml)）。将模型名称传递给 `--weights` 参数。模型会自动从[最新YOLOv5发布](https://github.com/ultralytics/yolov5/releases)中下载。

```shell
python train.py --data custom.yaml --weights yolov5s.pt
                                             yolov5m.pt
                                             yolov5l.pt
                                             yolov5x.pt
                                             custom_pretrained.pt
```
从头开始训练。 推荐用于大型数据集（即 COCO、Objects365、OIv6）。传递您感兴趣的模型架构YAML以及一个空的 --weights '' 参数：
```bash
python train.py --data custom.yaml --weights '' --cfg yolov5s.yaml
                                                      yolov5m.yaml
                                                      yolov5l.yaml
                                                      yolov5x.yaml
```
训练设置
在修改任何内容之前，首先使用默认设置进行训练以建立性能基准。train.py设置的完整列表可以在train.py的argparser中找到。

# Epochs（训练轮数）。 
从300轮开始。如果过早过拟合，可以减少轮数。如果300轮后不过拟合，可以训练更长时间，即600、1200等轮数。
图像大小。 COCO以原始分辨率--img 640训练，由于数据集中小对象较多，它可以从更高分辨率如--img 1280中受益。如果有很多小对象，自定义数据集将在原生或更高分辨率训练中受益。最佳推理结果是在与训练相同的--img下获得的，即如果您在--img 1280训练，您也应该在--img 1280测试和检测。
# 批量大小。 
使用您的硬件允许的最大--batch-size。小批量大小会产生较差的批规范统计数据，应避免。
# 超参数。
默认超参数在hyp.scratch-low.yaml中。我们建议您首先使用默认超参数进行训练，然后再考虑修改任何参数。一般来说，增加数据增强超参数会减少和延迟过拟合，允许更长时间的训练和更高的最终mAP。减少某些损失组件增益超参数如 hyp['obj'] 将有助于减少这些特定损失组件的过拟合。要自动优化这些超参数，请参阅我们的超参数演化教程。
进一步阅读
如果您想了解更多，一个好的起点是Karpathy的“神经网络训练配方”，其中包含了适用于所有机器学习领域的优秀训练理念：https://karpathy.github.io/2019/04/25/recipe/

祝你好运 🍀 如果有其他问题，请告诉我们！






