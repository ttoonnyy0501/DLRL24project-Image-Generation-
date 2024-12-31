# 深度学习与强化学习2024大作业

## MaskGIT

1. 训练VQGAN。调整文件```maskgit/training_vqgan.py```里的超参数，并设置数据集的路径。运行下面命令来训练模型。

 ```bash
 python training_vqgan.py
 ```

 2. 训练bidirectional transformer。整文件```maskgit/training_transformer.py```里的超参数，并设置数据集的路径。运行下面命令来训练模型。

 ```bash
 python training_transformer.py
 ```

 ## Finetune on diffusion model

首先配置环境```Finetune/environment.yaml```。然后在该文件中下载好预训练模型的参数。最后依次运行```Finetune/main.sh``` 的命令即可微调模型。