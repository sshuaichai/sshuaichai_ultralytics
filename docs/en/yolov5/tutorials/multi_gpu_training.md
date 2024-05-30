---
comments: true
description: 学习如何在单个或多个GPU上使用YOLOv5进行数据集训练。包括设置、训练模式和结果分析，以高效利用多个GPU。
keywords: YOLOv5, 多GPU训练, YOLOv5训练, 深度学习, 机器学习, 目标检测, Ultralytics
---

📚 本指南解释了如何正确使用**多个**GPU来使用YOLOv5 🚀在单台或多台机器上训练数据集。

## 开始之前

克隆仓库并在[**Python>=3.8.0**](https://www.python.org/)环境中安装[requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)，包括[**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)。[模型](https://github.com/ultralytics/yolov5/tree/master/models)和[数据集](https://github.com/ultralytics/yolov5/tree/master/data)将自动从最新的YOLOv5[发布](https://github.com/ultralytics/yolov5/releases)中下载。

```bash
git clone https://github.com/ultralytics/yolov5  # 克隆仓库
cd yolov5
pip install -r requirements.txt  # 安装依赖

```

💡 ProTip! **Docker Image** is recommended for all Multi-GPU trainings. See [Docker Quickstart Guide](../environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>
💡 专家提示！Docker镜像推荐用于所有多GPU训练。参见Docker快速入门指南 <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>


💡 ProTip! `torch.distributed.run` replaces `torch.distributed.launch` in **PyTorch>=1.9**. See [docs](https://pytorch.org/docs/stable/distributed.html) for details.
💡 专家提示！在PyTorch>=1.9中，torch.distributed.run替代了torch.distributed.launch。详见文档。


## Training

Select a pretrained model to start training from. Here we select [YOLOv5s](https://github.com/ultralytics/yolov5/blob/master/models/yolov5s.yaml), the smallest and fastest model available. See our README [table](https://github.com/ultralytics/yolov5#pretrained-checkpoints) for a full comparison of all models. We will train this model with Multi-GPU on the [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset.
训练
选择一个预训练模型开始训练。这里我们选择YOLOv5s，这是可用的最小和最快的模型。查看我们的README表格，了解所有模型的完整比较。我们将在COCO数据集上使用多GPU训练这个模型。
<p align="center"><img width="700" alt="YOLOv5 Models" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/model_comparison.png"></p>

### Single GPU 单GPU

```bash
python train.py  --batch 64 --data coco.yaml --weights yolov5s.pt --device 0
```

### Multi-GPU [DataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel) Mode (⚠️ not recommended)
多GPU DataParallel模式 (⚠️ 不推荐)
你可以增加device来使用多GPU的DataParallel模式。
You can increase the `device` to use Multiple GPUs in DataParallel mode.

```bash
python train.py  --batch 64 --data coco.yaml --weights yolov5s.pt --device 0,1
```

This method is slow and barely speeds up training compared to using just 1 GPU.
这种方法速度很慢，几乎没有加快训练速度，相比于仅使用1个GPU。

### Multi-GPU [DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel) Mode (✅ recommended)
多GPU DistributedDataParallel模式 (✅ 推荐)
你需要传递python -m torch.distributed.run --nproc_per_node，后跟通常的参数。
You will have to pass `python -m torch.distributed.run --nproc_per_node`, followed by the usual arguments.

```bash
python -m torch.distributed.run --nproc_per_node 2 train.py --batch 64 --data coco.yaml --weights yolov5s.pt --device 0,1
```

`--nproc_per_node` specifies how many GPUs you would like to use. In the example above, it is 2.
`--batch ` is the total batch-size. It will be divided evenly to each GPU. In the example above, it is 64/2=32 per GPU.
--nproc_per_node指定你想使用多少个GPU。在上面的例子中，是2个。
--batch是总批量大小。它将平均分配给每个GPU。在上面的例子中，是64/2=每个GPU 32。

上述代码将使用GPU 0... (N-1)。
The code above will use GPUs `0... (N-1)`.

<details>
  <summary>Use specific GPUs (click to expand)</summary>
  <summary>使用特定的GPU（点击展开）</summary>

You can do so by simply passing `--device` followed by your specific GPUs. For example, in the code below, we will use GPUs `2,3`.
你可以通过简单地传递--device后跟你的特定GPU来做到这一点。例如，在下面的代码中，我们将使用GPU 2,3。


```bash
python -m torch.distributed.run --nproc_per_node 2 train.py --batch 64 --data coco.yaml --cfg yolov5s.yaml --weights '' --device 2,3
```

</details>

<details>
  <summary>Use SyncBatchNorm (click to expand)</summary>
  <summary>使用SyncBatchNorm（点击展开）</summary>

[SyncBatchNorm](https://pytorch.org/docs/master/generated/torch.nn.SyncBatchNorm.html) could increase accuracy for multiple gpu training, however, it will slow down training by a significant factor. It is **only** available for Multiple GPU DistributedDataParallel training.
SyncBatchNorm可以提高多GPU训练的准确性，但是，它会显著减慢训练速度。它仅适用于多GPU DistributedDataParallel训练。

It is best used when the batch-size on **each** GPU is small (<= 8).
当每个GPU上的批量大小很小（<= 8）时，最好使用它。

To use SyncBatchNorm, simple pass `--sync-bn` to the command like below,
要使用SyncBatchNorm，只需将--sync-bn传递给命令，如下所示，

```bash
python -m torch.distributed.run --nproc_per_node 2 train.py --batch 64 --data coco.yaml --cfg yolov5s.yaml --weights '' --sync-bn
```

</details>

<details>
  <summary>Use Multiple machines (click to expand)</summary>
  <summary>使用多台机器（点击展开）</summary>

This is **only** available for Multiple GPU DistributedDataParallel training.
这仅适用于多GPU DistributedDataParallel训练。

Before we continue, make sure the files on all machines are the same, dataset, codebase, etc. Afterward, make sure the machines can communicate to each other.
在继续之前，确保所有机器上的文件相同，包括数据集、代码库等。之后，确保机器之间可以相互通信。

You will have to choose a master machine(the machine that the others will talk to). Note down its address(`master_addr`) and choose a port(`master_port`). I will use `master_addr = 192.168.1.1` and `master_port = 1234` for the example below.
你需要选择一台主机（其他机器将与之通信）。记下其地址（master_addr）并选择一个端口（master_port）。在下面的例子中，我将使用master_addr = 192.168.1.1和master_port = 1234。


To use it, you can do as the following,
要使用它，可以按照以下步骤进行，


```bash
# On master machine 0
python -m torch.distributed.run --nproc_per_node G --nnodes N --node_rank 0 --master_addr "192.168.1.1" --master_port 1234 train.py --batch 64 --data coco.yaml --cfg yolov5s.yaml --weights ''
```

```bash
# On machine R
python -m torch.distributed.run --nproc_per_node G --nnodes N --node_rank R --master_addr "192.168.1.1" --master_port 1234 train.py --batch 64 --data coco.yaml --cfg yolov5s.yaml --weights ''
```

where `G` is number of GPU per machine, `N` is the number of machines, and `R` is the machine number from `0...(N-1)`. Let's say I have two machines with two GPUs each, it would be `G = 2` , `N = 2`, and `R = 1` for the above.
其中G是每台机器的GPU数量，N是机器数量，R是机器编号从0...(N-1)。假设我有两台机器，每台有两个GPU，上述例子中G = 2 ，N = 2，R = 1。


Training will not start until <b>all </b> `N` machines are connected. Output will only be shown on master machine!
训练不会开始直到<b>所有</b>N台机器连接。输出将仅在主机上显示！


</details>

### Notes

- Windows support is untested, Linux is recommended.
- `--batch ` must be a multiple of the number of GPUs.
- GPU 0 will take slightly more memory than the other GPUs as it maintains EMA and is responsible for checkpointing etc.
- If you get `RuntimeError: Address already in use`, it could be because you are running multiple trainings at a time. To fix this, simply use a different port number by adding `--master_port` like below,
注意事项
Windows支持未经测试，推荐使用Linux。
--batch必须是GPU数量的倍数。
GPU 0将占用比其他GPU稍多的内存，因为它维护EMA并负责检查点等。
如果你收到RuntimeError: Address already in use，可能是因为你同时运行了多个训练。要解决此问题，只需使用不同的端口号，添加--master_port，如下所示，
```bash
python -m torch.distributed.run --master_port 1234 --nproc_per_node 2 ...
```

## Results
结果
在拥有8x A100 SXM4-40GB的AWS EC2 P4d实例上对YOLOv5l进行1个COCO epoch的DDP性能分析结果。
DDP profiling results on an [AWS EC2 P4d instance](../environments/aws_quickstart_tutorial.md) with 8x A100 SXM4-40GB for YOLOv5l for 1 COCO epoch.

<details>
  <summary>Profiling code</summary>
  <summary>性能分析代码</summary>

```bash
# 准备
t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/coco:/usr/src/coco $t
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
cd .. && rm -rf app && git clone https://github.com/ultralytics/yolov5 -b master app && cd app
cp data/coco.yaml data/coco_profile.yaml

# 性能分析
python train.py --batch-size 16 --data coco_profile.yaml --weights yolov5l.pt --epochs 1 --device 0
python -m torch.distributed.run --nproc_per_node 2 train.py --batch-size 32 --data coco_profile.yaml --weights yolov5l.pt --epochs 1 --device 0,1
python -m torch.distributed.run --nproc_per_node 4 train.py --batch-size 64 --data coco_profile.yaml --weights yolov5l.pt --epochs 1 --device 0,1,2,3
python -m torch.distributed.run --nproc_per_node 8 train.py --batch-size 128 --data coco_profile.yaml --weights yolov5l.pt --epochs 1 --device 0,1,2,3,4,5,6,7
```

</details>

| GPUs<br>A100 | batch-size | CUDA_mem<br><sup>device0 (G) | COCO<br><sup>train | COCO<br><sup>val |
|--------------|------------|------------------------------|--------------------|------------------|
| 1x           | 16         | 26GB                         | 20:39              | 0:55             |
| 2x           | 32         | 26GB                         | 11:43              | 0:57             |
| 4x           | 64         | 26GB                         | 5:57               | 0:55             |
| 8x           | 128        | 26GB                         | 3:09               | 0:57             |

## FAQ

If an error occurs, please read the checklist below first! (It could save your time)
如果出现错误，请首先阅读下面的清单！（它可以节省你的时间）


<details>
  <summary>Checklist (click to expand) </summary>

<ul>
    <li>Have you properly read this post?  </li>
    <li>Have you tried to re-clone the codebase? The code changes <b>daily</b>.</li>
    <li>Have you tried to search for your error? Someone may have already encountered it in this repo or in another and have the solution. </li>
    <li>Have you installed all the requirements listed on top (including the correct Python and Pytorch versions)? </li>
    <li>Have you tried in other environments listed in the "Environments" section below? </li>
    <li>Have you tried with another dataset like coco128 or coco2017? It will make it easier to find the root cause. </li>
</ul>
<ul>
    <li>你是否正确阅读了这篇文章？</li>
    <li>你是否尝试重新克隆代码库？代码<b>每日</b>都会更改。</li>
    <li>你是否尝试搜索你的错误？可能有人已经在这个仓库或其他地方遇到了这个问题并有了解决方案。</li>
    <li>你是否安装了顶部列出的所有依赖项（包括正确的Python和Pytorch版本）？</li>
    <li>你是否在"环境"部分列出的其他环境中尝试过？</li>
    <li>你是否尝试使用其他数据集，如coco128或coco2017？这样会更容易找到根本原因。</li>
</ul>
如果你已经完成了上述所有步骤，请按照模板提供尽可能多的详细信息提交一个Issue。
If you went through all the above, feel free to raise an Issue by giving as much detail as possible following the template.

</details>

## Supported Environments
支持的环境
Ultralytics提供了一系列即用型环境，每个环境都预装了基本依赖项，如CUDA、CUDNN、Python和PyTorch，以启动你的项目。

免费GPU笔记本: <a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
谷歌云: GCP快速入门指南
亚马逊: AWS快速入门指南
Azure: AzureML快速入门指南
Docker: Docker快速入门指南 <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>
Ultralytics provides a range of ready-to-use environments, each pre-installed with essential dependencies such as [CUDA](https://developer.nvidia.com/cuda), [CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/), and [PyTorch](https://pytorch.org/), to kickstart your projects.

- **Free GPU Notebooks**: <a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
- **Google Cloud**: [GCP Quickstart Guide](../environments/google_cloud_quickstart_tutorial.md)
- **Amazon**: [AWS Quickstart Guide](../environments/aws_quickstart_tutorial.md)
- **Azure**: [AzureML Quickstart Guide](../environments/azureml_quickstart_tutorial.md)
- **Docker**: [Docker Quickstart Guide](../environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## Project Status

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>

This badge indicates that all [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) Continuous Integration (CI) tests are successfully passing. These CI tests rigorously check the functionality and performance of YOLOv5 across various key aspects: [training](https://github.com/ultralytics/yolov5/blob/master/train.py), [validation](https://github.com/ultralytics/yolov5/blob/master/val.py), [inference](https://github.com/ultralytics/yolov5/blob/master/detect.py), [export](https://github.com/ultralytics/yolov5/blob/master/export.py), and [benchmarks](https://github.com/ultralytics/yolov5/blob/master/benchmarks.py). They ensure consistent and reliable operation on macOS, Windows, and Ubuntu, with tests conducted every 24 hours and upon each new commit.

## Credits

We would like to thank @MagicFrogSJTU, who did all the heavy lifting, and @glenn-jocher for guiding us along the way.
