---
comments: true
description: 使用YOLOv8增强您的Python项目中的目标检测、分割和分类功能。探索如何轻松加载、训练、验证、预测、导出、跟踪和基准测试模型。
keywords: YOLOv8, Ultralytics, Python, 目标检测, 分割, 分类, 模型训练, 验证, 预测, 模型导出, 基准测试, 实时跟踪
---

# Python 使用指南

欢迎使用YOLOv8 Python使用文档！本指南旨在帮助您将YOLOv8无缝集成到您的Python项目中，实现目标检测、分割和分类。在这里，您将学习如何加载和使用预训练模型、训练新模型并对图像进行预测。这个易于使用的Python接口是希望将YOLOv8集成到Python项目中的任何人的宝贵资源，使您能够快速实现高级目标检测功能。让我们开始吧！

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=58"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看:</strong> 精通 Ultralytics YOLOv8: Python
</p>

例如，用户可以加载一个模型，对其进行训练，评估其在验证集上的性能，甚至只需几行代码就可以将其导出为ONNX格式。

!!! Example "Python"

    ```python
    from ultralytics import YOLO

    # 从头开始创建一个新的YOLO模型
    model = YOLO('yolov8n.yaml')

    # 加载一个预训练的YOLO模型（推荐用于训练）
    model = YOLO('yolov8n.pt')

    # 使用'coco8.yaml'数据集训练模型3个周期
    results = model.train(data='coco8.yaml', epochs=3)

    # 评估模型在验证集上的性能
    results = model.val()

    # 使用模型对图像进行目标检测
    results = model('https://ultralytics.com/images/bus.jpg')

    # 将模型导出为ONNX格式
    success = model.export(format='onnx')
    ```

## [Train](../modes/train.md)

训练模式用于在自定义数据集上训练YOLOv8模型。在此模式下，使用指定的数据集和超参数对模型进行训练。训练过程包括优化模型的参数，使其能够准确预测图像中物体的类别和位置。

!!! Example "Train"

    === "从预训练模型开始（推荐）"

        ```python
        from ultralytics import YOLO

        model = YOLO('yolov8n.pt') # 传入任意模型类型
        results = model.train(epochs=5)
        ```

    === "从头开始"

        ```python
        from ultralytics import YOLO

        model = YOLO('yolov8n.yaml')
        results = model.train(data='coco8.yaml', epochs=5)
        ```

    === "继续训练"

        ```python
        model = YOLO("last.pt")
        results = model.train(resume=True)
        ```

[Train Examples](../modes/train.md){ .md-button }

## [Val](../modes/val.md)

验证模式用于在模型训练后验证YOLOv8模型。在此模式下，模型在验证集上进行评估，以衡量其准确性和泛化性能。此模式可用于调整模型的超参数以提高其性能。

!!! Example "Val"

    === "训练后验证"

        ```python
          from ultralytics import YOLO

          model = YOLO('yolov8n.yaml')
          model.train(data='coco8.yaml', epochs=5)
          model.val()  # 它会自动评估您训练的数据。
        ```

    === "独立验证"

        ```python
          from ultralytics import YOLO

          model = YOLO("model.pt")
          # 如果您不设置数据，它将使用model.pt中的数据YAML文件。
          model.val()
          # 或者您可以设置要验证的数据
          model.val(data='coco8.yaml')
        ```

[Val Examples](../modes/val.md){ .md-button }

## [Predict](../modes/predict.md)

预测模式用于使用训练好的YOLOv8模型对新图像或视频进行预测。在此模式下，模型从检查点文件加载，用户可以提供图像或视频进行推理。模型预测输入图像或视频中物体的类别和位置。

!!! Example "Predict"

    === "从源预测"

        ```python
        from ultralytics import YOLO
        from PIL import Image
        import cv2

        model = YOLO("model.pt")
        # 接受所有格式 - image/dir/Path/URL/video/PIL/ndarray. 0 表示摄像头
        results = model.predict(source="0")
        results = model.predict(source="folder", show=True) # 显示预测结果。接受所有YOLO预测参数

        # 使用 PIL
        im1 = Image.open("bus.jpg")
        results = model.predict(source=im1, save=True)  # 保存绘制的图像

        # 使用 ndarray
        im2 = cv2.imread("bus.jpg")
        results = model.predict(source=im2, save=True, save_txt=True)  # 将预测结果保存为标签

        # 使用 PIL/ndarray 列表
        results = model.predict(source=[im1, im2])
        ```

    === "结果使用"

        ```python
        # 默认情况下，结果将是一个包含所有预测的Results对象列表
        # 但要注意，当有许多图像时，尤其是任务是分割时，它可能占用大量内存。
        # 1. 作为列表返回
        results = model.predict(source="folder")

        # 设置 stream=True，结果将是一个更友好的生成器
        # 2. 作为生成器返回
        results = model.predict(source=0, stream=True)

        for result in results:
            # 检测
            result.boxes.xyxy   # 以xyxy格式的框, (N, 4)
            result.boxes.xywh   # 以xywh格式的框, (N, 4)
            result.boxes.xyxyn  # 以xyxy格式但归一化的框, (N, 4)
            result.boxes.xywhn  # 以xywh格式但归一化的框, (N, 4)
            result.boxes.conf   # 置信度得分, (N, 1)
            result.boxes.cls    # 类别, (N, 1)

            # 分割
            result.masks.data      # 掩码, (N, H, W)
            result.masks.xy        # x,y 段（像素）, List[segment] * N
            result.masks.xyn       # x,y 段（归一化）, List[segment] * N

            # 分类
            result.probs     # 类别概率, (num_class, )

        # 默认情况下，每个结果都是由 torch.Tensor 组成，
        # 您可以轻松使用以下功能：
        result = result.cuda()
        result = result.cpu()
        result = result.to("cpu")
        result = result.numpy()
        ```

[Predict Examples](../modes/predict.md){ .md-button }

## [Export](../modes/export.md)

导出模式用于将YOLOv8模型导出为可用于部署的格式。在此模式下，模型转换为其他软件应用程序或硬件设备可使用的格式。当将模型部署到生产环境时，此模式非常有用。

!!! Example "Export"

    === "导出为ONNX"

        将官方YOLOv8n模型导出为具有动态批处理大小和图像大小的ONNX格式。
        ```python
          from ultralytics import YOLO

          model = YOLO('yolov8n.pt')
          model.export(format='onnx', dynamic=True)
        ```

    === "导出为TensorRT"

        将官方YOLOv8n模型导出为TensorRT，以便在CUDA设备上加速（device=0）。
        ```python
          from ultralytics import YOLO

          model = YOLO('yolov8n.pt')
          model.export(format='onnx', device=0)
        ```

[Export Examples](../modes/export.md){ .md-button }

## [Track](../modes/track.md)

跟踪模式用于使用YOLOv8模型进行实时目标跟踪。在此模式下，模型从检查点文件加载，用户可以提供实时视频流进行实时目标跟踪。此模式对于监控系统或自动驾驶汽车等应用非常有用。

!!! Example "Track"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n.pt')  # 加载官方检测模型
        model = YOLO('yolov8n-seg.pt')  # 加载官方分割模型
        model = YOLO('path/to/best.pt')  # 加载自定义模型

        # 使用模型进行跟踪
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")
        ```

[Track Examples](../modes/track.md){ .md-button }

## [Benchmark](../modes/benchmark.md)

基准测试模式用于分析各种导出格式的YOLOv8模型的速度和准确性。基准测试提供了导出格式的大小、`mAP50-95`指标（用于目标检测和分割）或`accuracy_top5`指标（用于分类）以及每张图像的推理时间（以毫秒为单位）等信息。这些信息可以帮助用户根据对速度和准确性的要求选择最合适的导出格式。

!!! Example "Benchmark"

    === "Python"

        对官方YOLOv8n模型进行所有导出格式的基准测试。
        ```python
        from ultralytics.utils.benchmarks import benchmark

        # 基准测试
        benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device=0)
        ```

[Benchmark Examples](../modes/benchmark.md){ .md-button }

## Explorer

Explorer API可用于通过高级语义、向量相似性和SQL搜索等功能探索数据集。它还利用LLM的力量启用基于自然语言的图像内容搜索。Explorer API允许您编写自己的数据集探索笔记本或脚本，以获取数据集的见解。

!!! Example "使用Explorer进行语义搜索"

    === "使用图像"

        ```python
        from ultralytics import Explorer

        # 创建Explorer对象
        exp = Explorer(data='coco8.yaml', model='yolov8n.pt')
        exp.create_embeddings_table()

        similar = exp.get_similar(img='https://ultralytics.com/images/bus.jpg', limit=10)
        print(similar.head())

        # 使用多个索引进行搜索
        similar = exp.get_similar(
                                img=['https://ultralytics.com/images/bus.jpg',
                                     'https://ultralytics.com/images/bus.jpg'],
                                limit=10
                                )
        print(similar.head())
        ```

    === "使用数据集索引"

        ```python
        from ultralytics import Explorer

        # 创建Explorer对象
        exp = Explorer(data='coco8.yaml', model='yolov8n.pt')
        exp.create_embeddings_table()

        similar = exp.get_similar(idx=1, limit=10)
        print(similar.head())

        # 使用多个索引进行搜索
        similar = exp.get_similar(idx=[1,10], limit=10)
        print(similar.head())
        ```

[Explorer](../datasets/explorer/index.md){ .md-button }

## 使用 Trainers

`YOLO` 模型类是Trainer类的高级封装。每个YOLO任务都有自己的Trainer，继承自`BaseTrainer`。

!!! Tip "检测Trainer示例"

        ```python
        from ultralytics.models.yolo import DetectionTrainer, DetectionValidator, DetectionPredictor

        # Trainer
        trainer = DetectionTrainer(overrides={})
        trainer.train()
        trained_model = trainer.best

        # 验证
        val = DetectionValidator(args=...)
        val(model=trained_model)

        # 预测
        pred = DetectionPredictor(overrides={})
        pred(source=SOURCE, model=trained_model)

        # 从最后的权重恢复
        overrides["resume"] = trainer.last
        trainer = detect.DetectionTrainer(overrides=overrides)
        ```

您可以轻松定制Trainer以支持自定义任务或探索研发创意。了解更多关于定制 `Trainers`、`Validators` 和 `Predictors` 以满足项目需求的信息，请参见自定义部分。

[Customization tutorials](engine.md){ .md-button }
