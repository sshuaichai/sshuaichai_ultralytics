---
comments: true
description: 了解如何通过ClearML增强您的YOLOv5管道——跟踪训练运行、版本控制数据、远程监控模型和优化性能。
keywords: ClearML, YOLOv5, Ultralytics, AI工具箱, 训练数据, 远程训练, 超参数优化, YOLOv5模型
---

# ClearML 集成

<img align="center" src="https://github.com/thepycoder/clearml_screenshots/raw/main/logos_dark.png#gh-light-mode-only" alt="Clear|ML"><img align="center" src="https://github.com/thepycoder/clearml_screenshots/raw/main/logos_light.png#gh-dark-mode-only" alt="Clear|ML">

## 关于 ClearML

[ClearML](https://clear.ml/) 是一个[开源](https://github.com/allegroai/clearml)工具箱，旨在为您节省时间 ⏱️。

🔨 在<b>实验管理器</b>中跟踪每个YOLOv5训练运行

🔧 使用集成的ClearML <b>数据版本控制工具</b>进行版本控制并轻松访问您的自定义训练数据

🔦 使用ClearML Agent <b>远程训练和监控</b>您的YOLOv5训练运行

🔬 使用ClearML <b>超参数优化</b>获得最佳的mAP

🔭 使用ClearML Serving通过几个命令将新训练的<b>YOLOv5模型变成API</b>

<br>
您可以选择使用多少这些工具，可以仅使用实验管理器，或者将它们全部组合成一个令人印象深刻的管道！
<br>
<br>

![ClearML scalars dashboard](https://github.com/thepycoder/clearml_screenshots/raw/main/experiment_manager_with_compare.gif)

<br>
<br>

## 🦾 设置

要跟踪您的实验和/或数据，ClearML需要与服务器通信。您有两种选择：

可以免费注册[ClearML托管服务](https://clear.ml/)，或者可以设置自己的服务器，详见[这里](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server)。即使是服务器也是开源的，因此即使您处理敏感数据，也应该没问题！

- 安装 `clearml` python包：

    ```bash
    pip install clearml
    ```

- 通过[创建凭证](https://app.clear.ml/settings/workspace-configuration)将ClearML SDK连接到服务器（进入右上角的设置 -> 工作区 -> 创建新凭证），然后执行以下命令并按照指示操作：

    ```bash
    clearml-init
    ```

就是这样！您已经完成了😎

<br>

## 🚀 使用ClearML训练YOLOv5

要启用ClearML实验跟踪，只需安装ClearML pip包。

```bash
pip install clearml>=1.2.0

```

This will enable integration with the YOLOv5 training script. Every training run from now on, will be captured and stored by the ClearML experiment manager.
这将启用与YOLOv5训练脚本的集成。从现在开始，每次训练运行都将被ClearML实验管理器捕获和存储。


If you want to change the `project_name` or `task_name`, use the `--project` and `--name` arguments of the `train.py` script, by default the project will be called `YOLOv5` and the task `Training`. PLEASE NOTE: ClearML uses `/` as a delimiter for subprojects, so be careful when using `/` in your project name!
如果您想更改project_name或task_name，请使用train.py脚本的--project和--name参数，默认情况下项目将被称为YOLOv5，任务为Training。请注意：ClearML使用/作为子项目的分隔符，因此在使用/时请小心！


```bash
python train.py --img 640 --batch 16 --epochs 3 --data coco8.yaml --weights yolov5s.pt --cache
```

or with custom project and task name:
或者使用自定义项目和任务名称：

```bash
python train.py --project my_project --name my_training --img 640 --batch 16 --epochs 3 --data coco8.yaml --weights yolov5s.pt --cache
```

This will capture:
这将捕获：

源代码和未提交的更改
已安装的软件包
（超）参数
模型文件（使用--save-period n每n个epoch保存一次检查点）
控制台输出
标量（mAP_0.5、mAP_0.5:0.95、精度、召回率、损失、学习率等）
机器详细信息、运行时间、创建日期等一般信息
所有生成的图，如标签相关图和混淆矩阵
每个epoch的边框图像
每个epoch的拼图
每个epoch的验证图像
- Source code + uncommitted changes
- Installed packages
- (Hyper)parameters
- Model files (use `--save-period n` to save a checkpoint every n epochs)
- Console output
- Scalars (mAP_0.5, mAP_0.5:0.95, precision, recall, losses, learning rates, ...)
- General info such as machine details, runtime, creation date etc.
- All produced plots such as label correlogram and confusion matrix
- Images with bounding boxes per epoch
- Mosaic per epoch
- Validation images per epoch

That's a lot right? 🤯 Now, we can visualize all of this information in the ClearML UI to get an overview of our training progress. Add custom columns to the table view (such as e.g. mAP_0.5) so you can easily sort on the best performing model. Or select multiple experiments and directly compare them!
这很多，对吧？🤯现在，我们可以在ClearML UI中可视化所有这些信息，以获得训练进展的概览。将自定义列添加到表格视图（例如mAP_0.5），以便您可以轻松地对最佳表现的模型进行排序。或者选择多个实验并直接比较它们！


There even more we can do with all of this information, like hyperparameter optimization and remote execution, so keep reading if you want to see how that works!
我们还可以用所有这些信息做更多的事情，如超参数优化和远程执行，所以如果您想了解其工作原理，请继续阅读！


### 🔗 Dataset Version Management🔗 数据集版本管理


Versioning your data separately from your code is generally a good idea and makes it easy to acquire the latest version too. This repository supports supplying a dataset version ID, and it will make sure to get the data if it's not there yet. Next to that, this workflow also saves the used dataset ID as part of the task parameters, so you will always know for sure which data was used in which experiment!
将数据与代码分开进行版本控制通常是个好主意，并且使得获取最新版本变得容易。该仓库支持提供数据集版本ID，并确保在数据不存在时获取数据。此外，该工作流还将使用的数据集ID保存为任务参数的一部分，因此您将始终知道在每个实验中使用了哪些数据！


![ClearML Dataset Interface](https://github.com/thepycoder/clearml_screenshots/raw/main/clearml_data.gif)

### Prepare Your Dataset准备您的数据集


The YOLOv5 repository supports a number of different datasets by using YAML files containing their information. By default datasets are downloaded to the `../datasets` folder in relation to the repository root folder. So if you downloaded the `coco128` dataset using the link in the YAML or with the scripts provided by yolov5, you get this folder structure:
YOLOv5仓库支持通过使用包含其信息的YAML文件来支持许多不同的数据集。默认情况下，数据集将下载到与仓库根文件夹相关的../datasets文件夹中。因此，如果您使用YAML中的链接或yolov5提供的脚本下载了coco128数据集，您将得到以下文件夹结构：


```
..
|_ yolov5
|_ datasets
    |_ coco128
        |_ images
        |_ labels
        |_ LICENSE
        |_ README.txt
```

But this can be any dataset you wish. Feel free to use your own, as long as you keep to this folder structure.
但这可以是任何您希望的数据集。只要保持此文件夹结构，随时可以使用您自己的数据集。


Next, ⚠️**copy the corresponding YAML file to the root of the dataset folder**⚠️. This YAML files contains the information ClearML will need to properly use the dataset. You can make this yourself too, of course, just follow the structure of the example YAMLs.
接下来，⚠️将相应的YAML文件复制到数据集文件夹的根目录⚠️。此YAML文件包含ClearML正确使用数据集所需的信息。您也可以自己制作这个文件，只需遵循示例YAML的结构即可。


Basically we need the following keys: `path`, `train`, `test`, `val`, `nc`, `names`.
基本上我们需要以下键：path、train、test、val、nc、names。


```
..
|_ yolov5
|_ datasets
    |_ coco128
        |_ images
        |_ labels
        |_ coco128.yaml  # <---- HERE!
        |_ LICENSE
        |_ README.txt
```

### Upload Your Dataset
上传您的数据集

To get this dataset into ClearML as a versioned dataset, go to the dataset root folder and run the following command:
要将此数据集作为版本控制数据集上传到ClearML，请转到数据集根文件夹并运行以下命令：


```bash
cd coco128
clearml-data sync --project YOLOv5 --name coco128 --folder .
```

The command `clearml-data sync` is actually a shorthand command. You could also run these commands one after the other:
命令clearml-data sync实际上是一个简写命令。您也可以按顺序运行以下命令：


```bash
# 如果要基于另一个数据集版本，请选择--parent <parent_dataset_id>
# 这样不会上传重复的文件！
clearml-data create --name coco128 --project YOLOv5
clearml-data add --files .
clearml-data close
```

### Run Training Using A ClearML Dataset使用ClearML数据集运行训练

Now that you have a ClearML dataset, you can very simply use it to train custom YOLOv5 🚀 models!
现在您有了一个ClearML数据集，您可以非常简单地使用它来训练自定义的YOLOv5 🚀模型！

```bash
python train.py --img 640 --batch 16 --epochs 3 --data clearml://<your_dataset_id> --weights yolov5s.pt --cache
```

<br>

### 👀 Hyperparameter Optimization
👀 超参数优化

Now that we have our experiments and data versioned, it's time to take a look at what we can build on top!
现在我们已经对实验和数据进行了版本控制，是时候看看我们可以在此基础上构建什么了！


Using the code information, installed packages and environment details, the experiment itself is now **completely reproducible**. In fact, ClearML allows you to clone an experiment and even change its parameters. We can then just rerun it with these new parameters automatically, this is basically what HPO does!
使用代码信息、已安装的软件包和环境详细信息，实验本身现在是完全可重现的。实际上，ClearML允许您克隆一个实验并更改其参数。然后我们可以自动运行它，这基本上就是HPO（超参数优化）所做的！


To **run hyperparameter optimization locally**, we've included a pre-made script for you. Just make sure a training task has been run at least once, so it is in the ClearML experiment manager, we will essentially clone it and change its hyperparameters.
要本地运行超参数优化，我们为您提供了一个预制脚本。只需确保至少运行过一次训练任务，以便它在ClearML实验管理器中，我们将克隆它并更改其超参数。


You'll need to fill in the ID of this `template task` in the script found at `utils/loggers/clearml/hpo.py` and then just run it :) You can change `task.execute_locally()` to `task.execute()` to put it in a ClearML queue and have a remote agent work on it instead.
您需要在utils/loggers/clearml/hpo.py脚本中填写此模板任务的ID，然后运行它 :) 您可以将task.execute_locally()更改为task.execute()，以将其放入ClearML队列，并由远程代理处理。


```bash
# 要使用optuna，请先安装它，否则可以将优化器更改为RandomSearch
pip install optuna
python utils/loggers/clearml/hpo.py

```

![HPO](https://github.com/thepycoder/clearml_screenshots/raw/main/hpo.png)

## 🤯 Remote Execution (advanced) 🤯 远程执行（高级）


Running HPO locally is really handy, but what if we want to run our experiments on a remote machine instead? Maybe you have access to a very powerful GPU machine on-site, or you have some budget to use cloud GPUs. This is where the ClearML Agent comes into play. Check out what the agent can do here:
本地运行HPO非常方便，但如果我们想在远程机器上运行实验怎么办？也许您有访问现场非常强大的GPU机器的权限，或者您有一些预算用于云GPU。这就是ClearML Agent的用武之地。看看代理可以做些什么：


- [YouTube video](https://youtu.be/MX3BrXnaULs)
- [Documentation](https://clear.ml/docs/latest/docs/clearml_agent)

In short: every experiment tracked by the experiment manager contains enough information to reproduce it on a different machine (installed packages, uncommitted changes etc.). So a ClearML agent does just that: it listens to a queue for incoming tasks and when it finds one, it recreates the environment and runs it while still reporting scalars, plots etc. to the experiment manager.
简而言之：实验管理器跟踪的每个实验都包含足够的信息，以便在不同的机器上重现它（已安装的软件包、未提交的更改等）。因此，ClearML代理就是这样做的：它监听队列中的任务，一旦找到任务，它就会重新创建环境并运行它，同时仍然向实验管理器报告标量、图等。


You can turn any machine (a cloud VM, a local GPU machine, your own laptop ... ) into a ClearML agent by simply running:
您可以通过简单运行以下命令将任何机器（云VM、本地GPU机器、您的笔记本电脑……）变成ClearML代理：


```bash
clearml-agent daemon --queue <queues_to_listen_to> [--docker]
```

### Cloning, Editing And Enqueuing
克隆、编辑和入队

With our agent running, we can give it some work. Remember from the HPO section that we can clone a task and edit the hyperparameters? We can do that from the interface too!
代理运行后，我们可以给它一些工作。记得在HPO部分我们可以克隆任务并编辑超参数吗？我们也可以从界面中做到这一点！


🪄 Clone the experiment by right-clicking it
🪄 右键单击实验以克隆它


🎯 Edit the hyperparameters to what you wish them to be
🎯 编辑您希望的超参数


⏳ Enqueue the task to any of the queues by right-clicking it
⏳ 右键单击任务将其入队到任意队列


![Enqueue a task from the UI](https://github.com/thepycoder/clearml_screenshots/raw/main/enqueue.gif)

### Executing A Task Remotely远程执行任务


Now you can clone a task like we explained above, or simply mark your current script by adding `task.execute_remotely()` and on execution it will be put into a queue, for the agent to start working on!
现在您可以按照上述说明克隆任务，或者只需通过添加task.execute_remotely()标记当前脚本，并在执行时将其放入队列，由代理开始工作！


To run the YOLOv5 training script remotely, all you have to do is add this line to the training.py script after the clearml logger has been instantiated:
要远程运行YOLOv5训练脚本，只需在实例化ClearML logger后将此行添加到training.py脚本中：


```python
# ...
# 日志记录器
data_dict = None
if RANK in {-1, 0}:
    loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # 日志记录器实例
    if loggers.clearml:
        loggers.clearml.task.execute_remotely(queue="my_queue")  # <------ 添加此行
        # data_dict要么为空，如果用户没有选择ClearML数据集，要么由ClearML填充
        data_dict = loggers.clearml.data_dict
# ...

```

When running the training script after this change, python will run the script up until that line, after which it will package the code and send it to the queue instead!
在进行此更改后运行训练脚本时，python将运行脚本直到该行，然后将打包代码并发送到队列中！


### Autoscaling workers自动扩展工作者


ClearML comes with autoscalers too! This tool will automatically spin up new remote machines in the cloud of your choice (AWS, GCP, Azure) and turn them into ClearML agents for you whenever there are experiments detected in the queue. Once the tasks are processed, the autoscaler will automatically shut down the remote machines, and you stop paying!
ClearML还带有自动扩展器！该工具将在您选择的云（AWS、GCP、Azure）中自动启动新的远程机器，并将它们变成ClearML代理，只要检测到队列中有实验。一旦任务处理完毕，自动扩展器将自动关闭远程机器，您也停止付费！


Check out the autoscalers getting started video below.
请观看下面的自动扩展器入门视频。


[![Watch the video](https://img.youtube.com/vi/j4XVMAaUt3E/0.jpg)](https://youtu.be/j4XVMAaUt3E)
