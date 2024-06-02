---
comments: true
description: 了解如何使用 YOLOv8 的预测模式执行各种任务。学习如何处理图像、视频和数据格式等不同的推理源。
keywords: Ultralytics, YOLOv8, 预测模式, 推理源, 预测任务, 流模式, 图像处理, 视频处理, 机器学习, 人工智能
---

# 使用 Ultralytics YOLO 进行模型预测

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO 生态系统和集成">

## 简介

在机器学习和计算机视觉领域，对视觉数据进行推理或预测的过程称为“推理”。Ultralytics YOLOv8 提供了一项强大的功能，即**预测模式**，该模式专为广泛的数据源进行高性能、实时推理而设计。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/QtsI0TnwDZs?si=ljesw75cMO2Eas14"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何提取 Ultralytics YOLOv8 模型输出用于自定义项目。
</p>

## 实际应用

|                   制造业                   |                        体育                        |                   安全                    |
|:-----------------------------------------:|:-------------------------------------------------:|:-----------------------------------------:|
| ![车辆零件检测][car spare parts] | ![足球运动员检测][football player detect] | ![人跌倒检测][human fall detect] |
|           车辆零件检测           |              足球运动员检测               |            人跌倒检测            |

## 为什么选择使用 Ultralytics YOLO 进行推理？

以下是您应该考虑使用 YOLOv8 预测模式进行各种推理需求的原因：

- **多功能性：** 能够对图像、视频甚至实时流进行推理。
- **性能：** 专为实时、高速处理而设计，不牺牲准确性。
- **易用性：** 直观的 Python 和 CLI 接口，便于快速部署和测试。
- **高度可定制：** 多种设置和参数可根据您的具体要求调节模型的推理行为。

### 预测模式的主要功能

YOLOv8 的预测模式设计得非常健壮和多功能，具有以下特点：

- **兼容多种数据源：** 无论您的数据是单个图像、图像集合、视频文件还是实时视频流，预测模式都能处理。
- **流模式：** 使用流功能生成内存高效的 `Results` 对象生成器。通过在预测器的调用方法中设置 `stream=True` 来启用此功能。
- **批处理：** 能够在单个批处理中处理多个图像或视频帧，从而进一步加快推理时间。
- **易于集成：** 由于其灵活的 API，可以轻松集成到现有数据管道和其他软件组件中。

Ultralytics YOLO 模型返回的是一个 `Results` 对象列表，或者在推理时传递 `stream=True` 时返回一个内存高效的 `Results` 对象生成器：

!!! 示例 "Predict"

    === "返回一个列表，`stream=False`"

        ```python
        from ultralytics import YOLO

        # 加载一个模型
        model = YOLO('yolov8n.pt')  # 预训练的 YOLOv8n 模型

        # 对图像列表运行批量推理
        results = model(['im1.jpg', 'im2.jpg'])  # 返回一个 Results 对象列表

        # 处理结果列表
        for result in results:
            boxes = result.boxes  # 包含边界框输出的 Boxes 对象
            masks = result.masks  # 包含分割掩码输出的 Masks 对象
            keypoints = result.keypoints  # 包含姿态输出的 Keypoints 对象
            probs = result.probs  # 包含分类输出的 Probs 对象
            obb = result.obb  # 包含 OBB 输出的 Oriented Boxes 对象
            result.show()  # 显示在屏幕上
            result.save(filename='result.jpg')  # 保存到磁盘
        ```

    === "返回一个生成器，`stream=True`"

        ```python
        from ultralytics import YOLO

        # 加载一个模型
        model = YOLO('yolov8n.pt')  # 预训练的 YOLOv8n 模型

        # 对图像列表运行批量推理
        results = model(['im1.jpg', 'im2.jpg'], stream=True)  # 返回一个 Results 对象生成器

        # 处理结果生成器
        for result in results:
            boxes = result.boxes  # 包含边界框输出的 Boxes 对象
            masks = result.masks  # 包含分割掩码输出的 Masks 对象
            keypoints = result.keypoints  # 包含姿态输出的 Keypoints 对象
            probs = result.probs  # 包含分类输出的 Probs 对象
            obb = result.obb  # 包含 OBB 输出的 Oriented Boxes 对象
            result.show()  # 显示在屏幕上
            result.save(filename='result.jpg')  # 保存到磁盘
        ```

## 推理源

YOLOv8 可以处理不同类型的推理输入源，如下表所示。这些源包括静态图像、视频流和各种数据格式。表中还指示了每种源是否可以在流模式下使用（使用参数 `stream=True` ✅）。流模式在处理视频或实时流时特别有用，因为它会创建一个结果生成器，而不是将所有帧加载到内存中。

!!! 提示 "提示"

    使用 `stream=True` 处理长视频或大数据集以高效管理内存。当 `stream=False` 时，所有帧或数据点的结果将存储在内存中，对于大输入来说，这会迅速累积并导致内存不足错误。相反，`stream=True` 使用生成器，只保持当前帧或数据点的结果在内存中，从而显著减少内存消耗并防止内存不足问题。

| 源         | 参数                                   | 类型            | 备注                                                                                       |
|----------------|--------------------------------------------|-----------------|---------------------------------------------------------------------------------------------|
| 图像          | `'image.jpg'`                              | `str` 或 `Path` | 单个图像文件。                                                                          |
| URL            | `'https://ultralytics.com/images/bus.jpg'` | `str`           | 图像的 URL。                                                                            |
| 截屏     | `'screen'`                                 | `str`           | 捕获屏幕截图。                                                                       |
| PIL            | `Image.open('im.jpg')`                     | `PIL.Image`     | HWC 格式，RGB 通道。                                                               |
| OpenCV         | `cv2.imread('im.jpg')`                     | `np.ndarray`    | HWC 格式，BGR 通道，`uint8 (0-255)`。                                               |
| numpy          | `np.zeros((640,1280,3))`                   | `np.ndarray`    | HWC 格式，BGR 通道，`uint8 (0-255)`。                                               |
| torch          | `torch.zeros(16,3,320,640)`                | `torch.Tensor`  | BCHW 格式，RGB 通道，`float32 (0.0-1.0)`。                                          |
| CSV            | `'sources.csv'`                            | `str` 或 `Path` | 包含图像、视频或目录路径的 CSV 文件。                                |
| 视频 ✅        | `'video.mp4'`                              | `str` 或 `Path` | MP4、AVI 等格式的视频文件。                                                   |
| 目录 ✅    | `'path/'`                                  | `str` 或 `Path` | 包含图像或视频的目录路径。                                            |
| glob ✅         | `'path/*.jpg'`                             | `str`           | 匹配多个文件的 Glob 模式。使用 `*` 字符作为通配符。                  |
| YouTube ✅      | `'https://youtu.be/LNwODJXcvt4'`           | `str`           | YouTube 视频的 URL。                                                                     |
| 流 ✅       | `'rtsp://example.com/media.mp4'`           | `str`           | 流媒体协议 URL，如 RTSP、RTMP、TCP 或 IP 地址。                      |
| 多流 ✅ | `'list.streams'`                           | `str` 或 `Path` | `*.streams` 文本文件，每行一个流 URL，例如 8 个流将以批量大小 8 运行。 |

以下是使用每种源类型的代码示例：

!!! 示例 "预测源"

    === "图像"

        对图像文件运行推理。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 定义图像文件的路径
        source = 'path/to/image.jpg'

        # 对源运行推理
        results = model(source)  # Results 对象列表
        ```

    === "截屏"

        对当前屏幕内容（截屏）运行推理。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 定义当前截屏为源
        source = 'screen'

        # 对源运行推理
        results = model(source)  # Results 对象列表
        ```

    === "URL"

        对通过 URL 远程托管的图像或视频运行推理。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 定义远程图像或视频 URL
        source = 'https://ultralytics.com/images/bus.jpg'

        # 对源运行推理
        results = model(source)  # Results 对象列表
        ```

    === "PIL"

        对使用 Python Imaging Library (PIL) 打开的图像运行推理。
        ```python
        from PIL import Image
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 使用 PIL 打开图像
        source = Image.open('path/to/image.jpg')

        # 对源运行推理
        results = model(source)  # Results 对象列表
        ```

    === "OpenCV"

        对使用 OpenCV 读取的图像运行推理。
        ```python
        import cv2
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 使用 OpenCV 读取图像
        source = cv2.imread('path/to/image.jpg')

        # 对源运行推理
        results = model(source)  # Results 对象列表
        ```

    === "numpy"

        对表示为 numpy 数组的图像运行推理。
        ```python
        import numpy as np
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 创建一个随机 numpy 数组，形状为 (640, 640, 3)，值范围 [0, 255]，类型为 uint8
        source = np.random.randint(low=0, high=255, size=(640, 640, 3), dtype='uint8')

        # 对源运行推理
        results = model(source)  # Results 对象列表
        ```

    === "torch"

        对表示为 PyTorch 张量的图像运行推理。
        ```python
        import torch
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 创建一个随机 torch 张量，形状为 (1, 3, 640, 640)，值范围 [0, 1]，类型为 float32
        source = torch.rand(1, 3, 640, 640, dtype=torch.float32)

        # 对源运行推理
        results = model(source)  # Results 对象列表
        ```

    === "CSV"

        对 CSV 文件中列出的图像、URL、视频和目录集合运行推理。
        ```python
        import torch
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 定义包含图像、URL、视频和目录的 CSV 文件路径
        source = 'path/to/file.csv'

        # 对源运行推理
        results = model(source)  # Results 对象列表
        ```

    === "视频"

        对视频文件运行推理。通过使用 `stream=True`，您可以创建一个 Results 对象生成器以减少内存使用。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 定义视频文件路径
        source = 'path/to/video.mp4'

        # 对源运行推理
        results = model(source, stream=True)  # Results 对象生成器
        ```

    === "目录"

        对目录中的所有图像和视频运行推理。要同时捕获子目录中的图像和视频，请使用 glob 模式，例如 `path/to/dir/**/*`。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 定义包含推理图像和视频的目录路径
        source = 'path/to/dir'

        # 对源运行推理
        results = model(source, stream=True)  # Results 对象生成器
        ```

    === "glob"

        对所有匹配 glob 表达式（带有 `*` 字符）的图像和视频运行推理。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 定义一个搜索所有 JPG 文件的 glob 搜索模式
        source = 'path/to/dir/*.jpg'

        # 或定义一个包含子目录的递归 glob 搜索模式
        source = 'path/to/dir/**/*.jpg'

        # 对源运行推理
        results = model(source, stream=True)  # Results 对象生成器
        ```

    === "YouTube"

        对 YouTube 视频运行推理。通过使用 `stream=True`，您可以创建一个 Results 对象生成器，以减少长视频的内存使用。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 定义 YouTube 视频 URL 作为源
        source = 'https://youtu.be/LNwODJXcvt4'

        # 对源运行推理
        results = model(source, stream=True)  # Results 对象生成器
        ```

    === "流"

        使用 RTSP、RTMP、TCP 和 IP 地址协议对远程流媒体源运行推理。如果在 `*.streams` 文本文件中提供了多个流，则将运行批量推理，例如 8 个流将在批量大小 8 下运行，否则单个流将在批量大小 1 下运行。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 单个流，批量大小 1 推理
        source = 'rtsp://example.com/media.mp4'  # RTSP、RTMP、TCP 或 IP 流地址

        # 多个流，批量推理（例如，8 个流的批量大小为 8）
        source = 'path/to/list.streams'  # 每行一个流地址的 *.streams 文本文件

        # 对源运行推理
        results = model(source, stream=True)  # Results 对象生成器
        ```

## 推理参数

`model.predict()` 接受多个参数，可以在推理时传递以覆盖默认值：

!!! 示例

    ```python
    from ultralytics import YOLO

    # 加载预训练的 YOLOv8n 模型
    model = YOLO('yolov8n.pt')

    # 对 'bus.jpg' 运行带参数的推理
    model.predict('bus.jpg', save=True, imgsz=320, conf=0.5)
    ```

推理参数：

| 参数        | 类型           | 默认值                | 描述                                                                                                                                                                                                                          |
|----------------|----------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `source`        | `str`          | `'ultralytics/assets'` | 指定推理的数据源。可以是图像路径、视频文件、目录、URL 或实时流的设备 ID。支持多种格式和源，能够灵活应用于不同类型的输入。                                                                                                                                                         |
| `conf`          | `float`        | `0.25`                 | 设置检测的最小置信度阈值。低于此阈值的检测对象将被忽略。调整此值可以帮助减少误报。                                                                                                                                                                                |
| `iou`           | `float`        | `0.7`                  | 非极大值抑制（NMS）的交并比（IoU）阈值。较低的值通过消除重叠框来减少检测对象，有助于减少重复检测。                                                                                                                                                                      |
| `imgsz`         | `int 或 tuple` | `640`                  | 定义推理的图像大小。可以是单个整数 `640` 进行方形缩放，也可以是（高度、宽度）元组。适当的大小可以提高检测准确性和处理速度。                                                                                                                                                       |
| `half`          | `bool`         | `False`                | 启用半精度（FP16）推理，这可以在支持的 GPU 上加快模型推理速度，且对准确性的影响最小。                                                                                                                                                                              |
| `device`        | `str`          | `None`                 | 指定推理设备（例如，`cpu`、`cuda:0` 或 `0`）。允许用户在 CPU、特定 GPU 或其他计算设备之间选择进行模型执行。                                                                                                                                                                  |
| `max_det`       | `int`          | `300`                  | 每张图像允许的最大检测数。限制单个推理中模型可以检测的对象总数，防止在密集场景中产生过多输出。                                                                                                                                                                                |
| `vid_stride`    | `int`          | `1`                    | 视频输入的帧间隔。允许跳过视频中的帧，以加快处理速度，但会降低时间分辨率。值为 1 时处理每一帧，较高值会跳过帧。                                                                                                                                                         |
| `stream_buffer` | `bool`         | `False`                | 确定在处理视频流时是否应缓冲所有帧（`True`），或模型是否应返回最新帧（`False`）。对于实时应用程序很有用。                                                                                                                                                                              |
| `visualize`     | `bool`         | `False`                | 在推理期间激活模型特征的可视化，提供模型“看到”的信息。对于调试和模型解释非常有用。                                                                                                                                                                                           |
| `augment`       | `bool`         | `False`                | 启用测试时增强（TTA）进行预测，在推理速度方面可能会提高检测的鲁棒性。                                                                                                                                                                     |
| `agnostic_nms`  | `bool`         | `False`                | 启用类无关的非极大值抑制（NMS），将不同类别的重叠框合并。在类别重叠常见的多类别检测场景中很有用。                                                                                                                                                                               |
| `classes`       | `list[int]`    | `None`                 | 过滤预测到的一组类别 ID。只有属于指定类别的检测对象将被返回。对于多类别检测任务中专注于相关对象很有用。                                                                                                                                                                     |
| `retina_masks`  | `bool`         | `False`                | 使用模型中可用的高分辨率分割掩码。这可以增强分割任务的掩码质量，提供更精细的细节。                                                                                                                                                                                        |
| `embed`         | `list[int]`    | `None`                 | 指定从哪些层提取特征向量或嵌入。对于聚类或相似性搜索等下游任务很有用。                                                                                                                                                                                              |

可视化参数：

| 参数     | 类型            | 默认值 | 描述                                                                                                                                                                   |
|--------------|---------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `show`        | `bool`        | `False` | 如果 `True`，在窗口中显示带注释的图像或视频。对于开发或测试期间的即时视觉反馈很有用。                                           |
| `save`        | `bool`        | `False` | 启用将带注释的图像或视频保存到文件。对于文档、进一步分析或共享结果很有用。                                                     |
| `save_frames` | `bool`        | `False` | 处理视频时，将各个帧保存为图像。对于提取特定帧或进行详细的逐帧分析很有用。                                     |
| `save_txt`    | `bool`        | `False` | 将检测结果保存为文本文件，格式为 `[class] [x_center] [y_center] [width] [height] [confidence]`。对于与其他分析工具集成很有用。 |
| `save_conf`   | `bool`        | `False` | 在保存的文本文件中包含置信度分数。增强了用于后处理和分析的详细信息。                                                           |
| `save_crop`   | `bool`        | `False` | 将检测对象的裁剪图像保存。对于数据集增强、分析或创建特定对象的集中数据集很有用。                                             |
| `show_labels` | `bool`        | `True`  | 在视觉输出中显示每个检测对象的标签。提供对检测对象的即时理解。                                                                |
| `show_conf`   | `bool`        | `True`  | 在标签旁显示每个检测对象的置信度分数。提供对每个检测对象的模型置信度的了解。                                            |
| `show_boxes`  | `bool`        | `True`  | 在图像或视频帧上绘制检测对象的边界框。对于视觉识别和定位对象至关重要。                                          |
| `line_width`  | `None 或 int` | `None`  | 指定边界框的线宽。如果 `None`，线宽将根据图像大小自动调整。提供视觉清晰度的自定义选项。           |

## 图像和视频格式

YOLOv8 支持多种图像和视频格式，如 [ultralytics/data/utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/utils.py) 中所指定。请参见下表以了解有效的后缀和示例预测命令。

### 图像

下表包含有效的 Ultralytics 图像格式。

| 图像后缀 | 示例预测命令          | 参考文献                                                                     |
|----------------|----------------------------------|-------------------------------------------------------------------------------|
| `.bmp`         | `yolo predict source=image.bmp`  | [Microsoft BMP 文件格式](https://en.wikipedia.org/wiki/BMP_file_format)    |
| `.dng`         | `yolo predict source=image.dng`  | [Adobe DNG](https://helpx.adobe.com/camera-raw/digital-negative.html)         |
| `.jpeg`        | `yolo predict source=image.jpeg` | [JPEG](https://en.wikipedia.org/wiki/JPEG)                                    |
| `.jpg`         | `yolo predict source=image.jpg`  | [JPEG](https://en.wikipedia.org/wiki/JPEG)                                    |
| `.mpo`         | `yolo predict source=image.mpo`  | [多图片对象](https://fileinfo.com/extension/mpo)                    |
| `.png`         | `yolo predict source=image.png`  | [便携式网络图形](https://en.wikipedia.org/wiki/PNG)                |
| `.tif`         | `yolo predict source=image.tif`  | [标记图像文件格式](https://en.wikipedia.org/wiki/TIFF)                   |
| `.tiff`        | `yolo predict source=image.tiff` | [标记图像文件格式](https://en.wikipedia.org/wiki/TIFF)                   |
| `.webp`        | `yolo predict source=image.webp` | [WebP](https://en.wikipedia.org/wiki/WebP)                                    |
| `.pfm`         | `yolo predict source=image.pfm`  | [便携式浮点图](https://en.wikipedia.org/wiki/Netpbm#File_formats)        |

### 视频

下表包含有效的 Ultralytics 视频格式。

| 视频后缀 | 示例预测命令          | 参考文献                                                                        |
|----------------|----------------------------------|----------------------------------------------------------------------------------|
| `.asf`         | `yolo predict source=video.asf`  | [高级系统格式](https://en.wikipedia.org/wiki/Advanced_Systems_Format) |
| `.avi`         | `yolo predict source=video.avi`  | [音频视频交错](https://en.wikipedia.org/wiki/Audio_Video_Interleave)   |
| `.gif`         | `yolo predict source=video.gif`  | [图形交换格式](https://en.wikipedia.org/wiki/GIF)                 |
| `.m4v`         | `yolo predict source=video.m4v`  | [MPEG-4 第 14 部分](https://en.wikipedia.org/wiki/M4V)                              |
| `.mkv`         | `yolo predict source=video.mkv`  | [Matroska](https://en.wikipedia.org/wiki/Matroska)                               |
| `.mov`         | `yolo predict source=video.mov`  | [QuickTime 文件格式](https://en.wikipedia.org/wiki/QuickTime_File_Format)     |
| `.mp4`         | `yolo predict source=video.mp4`  | [MPEG-4 第 14 部分 - 维基百科](https://en.wikipedia.org/wiki/MPEG-4_Part_14)       |
| `.mpeg`        | `yolo predict source=video.mpeg` | [MPEG-1 第 2 部分](https://en.wikipedia.org/wiki/MPEG-1)                            |
| `.mpg`         | `yolo predict source=video.mpg`  | [MPEG-1 第 2 部分](https://en.wikipedia.org/wiki/MPEG-1)                            |
| `.ts`          | `yolo predict source=video.ts`   | [MPEG 传输流](https://en.wikipedia.org/wiki/MPEG_transport_stream)     |
| `.wmv`         | `yolo predict source=video.wmv`  | [Windows 媒体视频](https://en.wikipedia.org/wiki/Windows_Media_Video)         |
| `.webm`        | `yolo predict source=video.webm` | [WebM 项目](https://en.wikipedia.org/wiki/WebM)                               |

## 处理结果

所有 Ultralytics `predict()` 调用将返回一个 `Results` 对象列表：

!!! 示例 "Results"

    ```python
    from ultralytics import YOLO

    # 加载预训练的 YOLOv8n 模型
    model = YOLO('yolov8n.pt')

    # 对图像运行推理
    results = model('bus.jpg')  # 1 个 Results 对象的列表
    results = model(['bus.jpg', 'zidane.jpg'])  # 2 个 Results 对象的列表
    ```

`Results` 对象具有以下属性：

| 属性    | 类型                  | 描述                                                                              |
|--------------|-----------------------|------------------------------------------------------------------------------------------|
| `orig_img`   | `numpy.ndarray`       | 原始图像，作为 numpy 数组。                                                     |
| `orig_shape` | `tuple`               | 原始图像形状，格式为（高度、宽度）。                                      |
| `boxes`      | `Boxes, optional`     | 包含检测边界框的 Boxes 对象。                                  |
| `masks`      | `Masks, optional`     | 包含检测掩码的 Masks 对象。                                           |
| `probs`      | `Probs, optional`     | 包含分类任务类别概率的 Probs 对象。           |
| `keypoints`  | `Keypoints, optional` | 包含每个对象检测关键点的 Keypoints 对象。                        |
| `obb`        | `OBB, optional`       | 包含定向边界框的 OBB 对象。                                        |
| `speed`      | `dict`                | 每张图像的预处理、推理和后处理速度（以毫秒为单位）。 |
| `names`      | `dict`                | 类别名称字典。                                                             |
| `path`       | `str`                 | 图像文件的路径。                                                              |

`Results` 对象具有以下方法：

| 方法        | 返回类型     | 描述                                                                         |
|---------------|-----------------|-------------------------------------------------------------------------------------|
| `update()`    | `None`          | 更新 Results 对象的 boxes、masks 和 probs 属性。                |
| `cpu()`       | `Results`       | 返回包含所有张量在 CPU 内存中的 Results 对象副本。                 |
| `numpy()`     | `Results`       | 返回包含所有张量作为 numpy 数组的 Results 对象副本。               |
| `cuda()`      | `Results`       | 返回包含所有张量在 GPU 内存中的 Results 对象副本。                 |
| `to()`        | `Results`       | 返回包含张量在指定设备和 dtype 上的 Results 对象副本。 |
| `new()`       | `Results`       | 返回包含相同图像、路径和名称的新 Results 对象。                   |
| `plot()`      | `numpy.ndarray` | 绘制检测结果。返回带注释的图像的 numpy 数组。          |
| `show()`      | `None`          | 将带注释的结果显示在屏幕上。                                                   |
| `save()`      | `None`          | 将带注释的结果保存到文件。                                                     |
| `verbose()`   | `str`           | 返回每个任务的日志字符串。                                                    |
| `save_txt()`  | `None`          | 将预测结果保存到文本文件中。                                                   |
| `save_crop()` | `None`          | 将裁剪的预测结果保存到 `save_dir/cls/file_name.jpg` 中。          |
| `tojson()`    | `str`           | 将对象转换为 JSON 格式。                                                  |

有关更多详细信息，请参见 [`Results` 类文档](../reference/engine/results.md)。

### Boxes

`Boxes` 对象可以用于索引、操作和将边界框转换为不同格式。

!!! 示例 "Boxes"

    ```python
    from ultralytics import YOLO

    # 加载预训练的 YOLOv8n 模型
    model = YOLO('yolov8n.pt')

    # 对图像运行推理
    results = model('bus.jpg')  # 结果列表

    # 查看结果
    for r in results:
        print(r.boxes)  # 打印包含检测边界框的 Boxes 对象
    ```

以下是 `Boxes` 类的方法和属性表，包括它们的名称、类型和描述：

| 名称      | 类型                      | 描述                                                        |
|-----------|---------------------------|--------------------------------------------------------------------|
| `cpu()`   | 方法                    | 将对象移动到 CPU 内存中。                                     |
| `numpy()` | 方法                    | 将对象转换为 numpy 数组。                               |
| `cuda()`  | 方法                    | 将对象移动到 CUDA 内存中。                                    |
| `to()`    | 方法                    | 将对象移动到指定设备。                           |
| `xyxy`    | 属性 (`torch.Tensor`) | 以 xyxy 格式返回框。                                   |
| `conf`    | 属性 (`torch.Tensor`) | 返回框的置信度值。                         |
| `cls`     | 属性 (`torch.Tensor`) | 返回框的类别值。                              |
| `id`      | 属性 (`torch.Tensor`) | 返回框的跟踪 ID（如果可用）。                  |
| `xywh`    | 属性 (`torch.Tensor`) | 以 xywh 格式返回框。                                   |
| `xyxyn`   | 属性 (`torch.Tensor`) | 以 xyxy 格式返回框，归一化为原始图像大小。 |
| `xywhn`   | 属性 (`torch.Tensor`) | 以 xywh 格式返回框，归一化为原始图像大小。 |

有关更多详细信息，请参见 [`Boxes` 类文档](../reference/engine/results.md#ultralytics.engine.results.Boxes)。

### Masks

`Masks` 对象可以用于索引、操作和将掩码转换为分段。

!!! 示例 "Masks"

    ```python
    from ultralytics import YOLO

    # 加载预训练的 YOLOv8n-seg 分割模型
    model = YOLO('yolov8n-seg.pt')

    # 对图像运行推理
    results = model('bus.jpg')  # 结果列表

    # 查看结果
    for r in results:
        print(r.masks)  # 打印包含检测实例掩码的 Masks 对象
    ```

以下是 `Masks` 类的方法和属性表，包括它们的名称、类型和描述：

| 名称      | 类型                      | 描述                                                     |
|-----------|---------------------------|-----------------------------------------------------------------|
| `cpu()`   | 方法                    | 返回 CPU 内存中的掩码张量。                         |
| `numpy()` | 方法                    | 返回掩码张量作为 numpy 数组。                      |
| `cuda()`  | 方法                    | 返回 GPU 内存中的掩码张量。                         |
| `to()`    | 方法                    | 返回具有指定设备和 dtype 的掩码张量。   |
| `xyn`     | 属性 (`torch.Tensor`) | 以张量表示的归一化分段列表。           |
| `xy`      | 属性 (`torch.Tensor`) | 以像素坐标表示的分段列表。           |

有关更多详细信息，请参见 [`Masks` 类文档](../reference/engine/results.md#ultralytics.engine.results.Masks)。

### Keypoints

`Keypoints` 对象可以用于索引、操作和归一化坐标。

!!! 示例 "Keypoints"

    ```python
    from ultralytics import YOLO

    # 加载预训练的 YOLOv8n-pose 姿态模型
    model = YOLO('yolov8n-pose.pt')

    # 对图像运行推理
    results = model('bus.jpg')  # 结果列表

    # 查看结果
    for r in results:
        print(r.keypoints)  # 打印包含检测关键点的 Keypoints 对象
    ```

以下是 `Keypoints` 类的方法和属性表，包括它们的名称、类型和描述：

| 名称      | 类型                      | 描述                                                       |
|-----------|---------------------------|-------------------------------------------------------------------|
| `cpu()`   | 方法                    | 返回 CPU 内存中的关键点张量。                       |
| `numpy()` | 方法                    | 返回关键点张量作为 numpy 数组。                    |
| `cuda()`  | 方法                    | 返回 GPU 内存中的关键点张量。                       |
| `to()`    | 方法                    | 返回具有指定设备和 dtype 的关键点张量。 |
| `xyn`     | 属性 (`torch.Tensor`) | 以张量表示的归一化关键点列表。            |
| `xy`      | 属性 (`torch.Tensor`) | 以像素坐标表示的关键点列表。  |
| `conf`    | 属性 (`torch.Tensor`) | 返回关键点的置信度值（如果可用），否则为 None。   |

有关更多详细信息，请参见 [`Keypoints` 类文档](../reference/engine/results.md#ultralytics.engine.results.Keypoints)。

### Probs

`Probs` 对象可以用于索引、获取分类的 `top1` 和 `top5` 索引和分数。

!!! 示例 "Probs"

    ```python
    from ultralytics import YOLO

    # 加载预训练的 YOLOv8n-cls 分类模型
    model = YOLO('yolov8n-cls.pt')

    # 对图像运行推理
    results = model('bus.jpg')  # 结果列表

    # 查看结果
    for r in results:
        print(r.probs)  # 打印包含检测类别概率的 Probs 对象
    ```

以下是 `Probs` 类的方法和属性表，包括它们的名称、类型和描述：

| 名称       | 类型                      | 描述                                                             |
|------------|---------------------------|-------------------------------------------------------------------------|
| `cpu()`    | 方法                    | 返回 CPU 内存中的 probs 张量副本。                       |
| `numpy()`  | 方法                    | 返回 probs 张量作为 numpy 数组的副本。                    |
| `cuda()`   | 方法                    | 返回 GPU 内存中的 probs 张量副本。                       |
| `to()`     | 方法                    | 返回具有指定设备和 dtype 的 probs 张量副本。 |
| `top1`     | 属性 (`int`)          | top 1 类别的索引。                                               |
| `top5`     | 属性 (`list[int]`)    | top 5 类别的索引。                                           |
| `top1conf` | 属性 (`torch.Tensor`) | top 1 类别的置信度。                                          |
| `top5conf` | 属性 (`torch.Tensor`) | top 5 类别的置信度。                                       |

有关更多详细信息，请参见 [`Probs` 类文档](../reference/engine/results.md#ultralytics.engine.results.Probs)。

### OBB

`OBB` 对象可以用于索引、操作和将定向边界框转换为不同格式。

!!! 示例 "OBB"

    ```python
    from ultralytics import YOLO

    # 加载预训练的 YOLOv8n 模型
    model = YOLO('yolov8n-obb.pt')

    # 对图像运行推理
    results = model('bus.jpg')  # 结果列表

    # 查看结果
    for r in results:
        print(r.obb)  # 打印包含定向检测边界框的 OBB 对象
    ```

以下是 `OBB` 类的方法和属性表，包括它们的名称、类型和描述：

| 名称        | 类型                      | 描述                                                           |
|-------------|---------------------------|-----------------------------------------------------------------------|
| `cpu()`     | 方法                    | 将对象移动到 CPU 内存中。                                        |
| `numpy()`   | 方法                    | 将对象转换为 numpy 数组。                                  |
| `cuda()`    | 方法                    | 将对象移动到 CUDA 内存中。                                       |
| `to()`      | 方法                    | 将对象移动到指定设备。                              |
| `conf`      | 属性 (`torch.Tensor`) | 返回框的置信度值。                            |
| `cls`       | 属性 (`torch.Tensor`) | 返回框的类别值。                                 |
| `id`        | 属性 (`torch.Tensor`) | 返回框的跟踪 ID（如果可用）。                     |
| `xyxy`      | 属性 (`torch.Tensor`) | 以 xyxy 格式返回水平框。                           |
| `xywhr`     | 属性 (`torch.Tensor`) | 以 xywhr 格式返回旋转框。                             |
| `xyxyxyxy`  | 属性 (`torch.Tensor`) | 以 xyxyxyxy 格式返回旋转框。                          |
| `xyxyxyxyn` | 属性 (`torch.Tensor`) | 以 xyxyxyxy 格式返回旋转框，归一化为图像大小。 |

有关更多详细信息，请参见 [`OBB` 类文档](../reference/engine/results.md#ultralytics.engine.results.OBB)。

## 绘制结果

`Results` 对象中的 `plot()` 方法通过将检测到的对象（例如边界框、掩码、关键点和概率）覆盖在原始图像上，便于可视化预测结果。此方法返回带注释的图像作为 NumPy 数组，便于显示或保存。

!!! 示例 "绘制"

    ```python
    from PIL import Image
    from ultralytics import YOLO

    # 加载预训练的 YOLOv8n 模型
    model = YOLO('yolov8n.pt')

    # 对 'bus.jpg' 运行推理
    results = model(['bus.jpg', 'zidane.jpg'])  # 结果列表

    # 可视化结果
    for i, r in enumerate(results):
        # 绘制结果图像
        im_bgr = r.plot()  # BGR 顺序 numpy 数组
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB 顺序 PIL 图像
        
        # 将结果显示到屏幕上（在支持的环境中）
        r.show()

        # 将结果保存到磁盘
        r.save(filename=f'results{i}.jpg')
    ```

### `plot()` 方法参数

`plot()` 方法支持多种参数以自定义输出：

| 参数     | 类型            | 描述                                                                | 默认值       |
|--------------|---------------|----------------------------------------------------------------------------|---------------|
| `conf`       | `bool`          | 包括检测置信度分数。                                       | `True`        |
| `line_width` | `float`         | 边界框的线宽。如果 `None`，则根据图像大小缩放。            | `None`        |
| `font_size`  | `float`         | 文本字体大小。如果 `None`，则根据图像大小缩放。                          | `None`        |
| `font`       | `str`           | 文本注释的字体名称。                                            | `'Arial.ttf'` |
| `pil`        | `bool`          | 返回图像作为 PIL 图像对象。                                        | `False`       |
| `img`        | `numpy.ndarray` | 用于绘制的替代图像。如果 `None`，则使用原始图像。         | `None`        |
| `im_gpu`     | `torch.Tensor`  | 用于更快掩码绘制的 GPU 加速图像。形状：(1, 3, 640, 640)。   | `None`        |
| `kpt_radius` | `int`           | 绘制关键点的半径。                                                | `5`           |
| `kpt_line`   | `bool`          | 用线连接关键点。                                              | `True`        |
| `labels`     | `bool`          | 在注释中包含类别标签。                                       | `True`        |
| `boxes`      | `bool`          | 在图像上覆盖边界框。                                       | `True`        |
| `masks`      | `bool`          | 在图像上覆盖掩码。                                                | `True`        |
| `probs`      | `bool`          | 包括分类概率。                                      | `True`        |
| `show`       | `bool`          | 使用默认图像查看器直接显示带注释的图像。       | `False`       |
| `save`       | `bool`          | 将带注释的图像保存到 `filename` 指定的文件。                | `False`       |
| `filename`   | `str`           | `save` 为 `True` 时保存带注释图像的文件路径和名称。 | `None`        |

## 线程安全推理

在推理期间确保线程安全对于在不同线程中并行运行多个 YOLO 模型至关重要。线程安全推理确保每个线程的预测是独立的，不会相互干扰，避免竞争条件并确保输出的一致性和可靠性。

在多线程应用程序中使用 YOLO 模型时，重要的是为每个线程实例化单独的模型对象，或使用线程本地存储以防止冲突：

!!! 示例 "线程安全推理"

    在每个线程内实例化单个模型进行线程安全推理：
    ```python
    from ultralytics import YOLO
    from threading import Thread

    def thread_safe_predict(image_path):
        """使用本地实例化的 YOLO 模型对图像进行线程安全的预测。"""
        local_model = YOLO("yolov8n.pt")
        results = local_model.predict(image_path)
        # 处理结果


    # 启动每个都有自己的模型实例的线程
    Thread(target=thread_safe_predict, args=("image1.jpg",)).start()
    Thread(target=thread_safe_predict, args=("image2.jpg",)).start()
    ```

有关使用 YOLO 模型进行线程安全推理的详细信息和分步说明，请参阅我们的 [YOLO 线程安全推理指南](../guides/yolo-thread-safe-inference.md)。该指南将提供避免常见陷阱的所有必要信息，确保您的多线程推理顺利进行。

## 流源 `for` 循环

以下是一个使用 OpenCV (`cv2`) 和 YOLOv8 对视频帧运行推理的 Python 脚本。该脚本假定您已经安装了必要的软件包（`opencv-python` 和 `ultralytics`）。

!!! 示例 "流 `for` 循环"

    ```python
    import cv2
    from ultralytics import YOLO

    # 加载 YOLOv8 模型
    model = YOLO('yolov8n.pt')

    # 打开视频文件
    video_path = "path/to/your/video/file.mp4"
    cap = cv2.VideoCapture(video_path)

    # 循环遍历视频帧
    while cap.isOpened():
        # 从视频中读取一帧
        success, frame = cap.read()

        if success:
            # 对帧运行 YOLOv8 推理
            results = model(frame)

            # 在帧上可视化结果
            annotated_frame = results[0].plot()

            # 显示带注释的帧
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # 如果按下 'q' 键，退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # 如果到达视频末尾，退出循环
            break

    # 释放视频捕获对象并关闭显示窗口
    cap.release()
    cv2.destroyAllWindows()
    ```

该脚本将对视频的每一帧运行预测，可视化结果，并在窗口中显示它们。可以通过按下 'q' 键退出循环。

[car spare parts]: https://github.com/RizwanMunawar/ultralytics/assets/62513924/a0f802a8-0776-44cf-8f17-93974a4a28a1

[football player detect]: https://github.com/RizwanMunawar/ultralytics/assets/62513924/7d320e1f-fc57-4d7f-a691-78ee579c3442

[human fall detect]: https://github.com/RizwanMunawar/ultralytics/assets/62513924/86437c4a-3227-4eee-90ef-9efb697bdb43
