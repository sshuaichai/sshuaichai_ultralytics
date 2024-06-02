---
comments: true
description: Learn how to profile speed and accuracy of YOLOv8 across various export formats; get insights on mAP50-95, accuracy_top5 metrics, and more.
keywords: Ultralytics, YOLOv8, benchmarking, speed profiling, accuracy profiling, mAP50-95, accuracy_top5, ONNX, OpenVINO, TensorRT, YOLO export formats
---

# 使用 Ultralytics YOLO 进行模型基准测试

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO ecosystem and integrations">

## 简介

一旦您的模型经过训练和验证，接下来的逻辑步骤是评估其在各种真实场景中的表现。Ultralytics YOLOv8 的基准模式提供了一个强大的框架，用于评估模型在各种导出格式下的速度和准确性。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?start=105"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> Ultralytics 模式教程：基准测试
</p>

## 为什么基准测试很重要？

- **明智的决策**：了解速度和准确性之间的权衡。
- **资源分配**：了解不同导出格式在不同硬件上的表现。
- **优化**：了解哪种导出格式为您的特定用例提供最佳性能。
- **成本效益**：根据基准测试结果更有效地利用硬件资源。

### 基准模式中的关键指标

- **mAP50-95**：用于目标检测、分割和姿态估计。
- **accuracy_top5**：用于图像分类。
- **推理时间**：每张图像的处理时间（以毫秒为单位）。

### 支持的导出格式

- **ONNX**：用于优化 CPU 性能
- **TensorRT**：用于最大化 GPU 效率
- **OpenVINO**：用于 Intel 硬件优化
- **CoreML、TensorFlow SavedModel 及更多**：满足多样化的部署需求。

!!! 提示 "提示"

    * 导出到 ONNX 或 OpenVINO 可使 CPU 性能提高最多 3 倍。
    * 导出到 TensorRT 可使 GPU 性能提高最多 5 倍。

## 使用示例

在所有支持的导出格式（包括 ONNX、TensorRT 等）上运行 YOLOv8n 基准测试。请参阅下面的 Arguments 部分，了解完整的导出参数列表。

!!! 示例

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # 在 GPU 上进行基准测试
        benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device=0)
        ```

    === "CLI"

        ```bash
        yolo benchmark model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

## 参数

参数如 `model`、`data`、`imgsz`、`half`、`device` 和 `verbose` 为用户提供了灵活性，使其能够根据具体需求微调基准测试，并轻松比较不同导出格式的性能。

| 键        | 默认值       | 描述                                                                                                                                                    |
|-----------|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model`   | `None`        | 指定模型文件的路径。接受 `.pt` 和 `.yaml` 格式，例如，预训练模型 `"yolov8n.pt"` 或配置文件。                                                           |
| `data`    | `None`        | 定义数据集的 YAML 文件路径，通常包括验证数据的路径和设置。例如： `"coco8.yaml"`。                                                                        |
| `imgsz`   | `640`         | 模型的输入图像大小。可以是方形图像的单个整数，也可以是非方形图像的元组 `(width, height)`，例如 `(640, 480)`。                                           |
| `half`    | `False`       | 启用 FP16（半精度）推理，减少内存使用并可能在兼容硬件上提高速度。使用 `half=True` 启用。                                                                |
| `int8`    | `False`       | 激活 INT8 量化，以在支持的设备上进一步优化性能，特别适用于边缘设备。设置 `int8=True` 使用。                                                              |
| `device`  | `None`        | 定义用于基准测试的计算设备，例如 `"cpu"`、`"cuda:0"`，或多 GPU 设置的设备列表，如 `"cuda:0,1"`。                                                       |
| `verbose` | `False`       | 控制日志输出的详细程度。布尔值；设置 `verbose=True` 可获取详细日志，或使用浮点数阈值错误。                                                                |

## 导出格式

基准测试将尝试在下面所有可能的导出格式上自动运行。

| 格式                                            | `format` 参数  | 模型                      | 元数据  | 参数                                                                            |
|-------------------------------------------------|----------------|---------------------------|---------|---------------------------------------------------------------------------------|
| [PyTorch](https://pytorch.org/)                 | -              | `yolov8n.pt`              | ✅       | -                                                                               |
| [TorchScript](../integrations/torchscript.md)   | `torchscript`  | `yolov8n.torchscript`     | ✅       | `imgsz`、`optimize`、`batch`                                                    |
| [ONNX](../integrations/onnx.md)                 | `onnx`         | `yolov8n.onnx`            | ✅       | `imgsz`、`half`、`dynamic`、`simplify`、`opset`、`batch`                         |
| [OpenVINO](../integrations/openvino.md)         | `openvino`     | `yolov8n_openvino_model/` | ✅       | `imgsz`、`half`、`int8`、`batch`                                                |
| [TensorRT](../integrations/tensorrt.md)         | `engine`       | `yolov8n.engine`          | ✅       | `imgsz`、`half`、`dynamic`、`simplify`、`workspace`、`int8`、`batch`             |
| [CoreML](../integrations/coreml.md)             | `coreml`       | `yolov8n.mlpackage`       | ✅       | `imgsz`、`half`、`int8`、`nms`、`batch`                                          |
| [TF SavedModel](../integrations/tf-savedmodel.md)| `saved_model`  | `yolov8n_saved_model/`    | ✅       | `imgsz`、`keras`、`int8`、`batch`                                               |
| [TF GraphDef](../integrations/tf-graphdef.md)   | `pb`           | `yolov8n.pb`              | ❌       | `imgsz`、`batch`                                                                 |
| [TF Lite](../integrations/tflite.md)            | `tflite`       | `yolov8n.tflite`          | ✅       | `imgsz`、`half`、`int8`、`batch`                                                 |
| [TF Edge TPU](../integrations/edge-tpu.md)      | `edgetpu`      | `yolov8n_edgetpu.tflite`  | ✅       | `imgsz`、`batch`                                                                 |
| [TF.js](../integrations/tfjs.md)                | `tfjs`         | `yolov8n_web_model/`      | ✅       | `imgsz`、`half`、`int8`、`batch`                                                 |
| [PaddlePaddle](../integrations/paddlepaddle.md) | `paddle`       | `yolov8n_paddle_model/`   | ✅       | `imgsz`、`batch`                                                                 |
| [NCNN](../integrations/ncnn.md)                 | `ncnn`         | `yolov8n_ncnn_model/`     | ✅       | `imgsz`、`half`、`batch`                                                         |

查看完整的 `export` 详情请参阅 [Export](../modes/export.md) 页面。
