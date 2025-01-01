# 深度学习与强化学习2024大作业

## MaskGIT

1. 配置环境。先安装pip，在运行下面命令安装环境依赖。

```bash
pip install -r maskgit/requirements.txt
```

2. 训练VQGAN。调整文件```maskgit/training_vqgan.py```里的超参数，并设置数据集的路径。运行下面命令来训练模型。

 ```bash
 python maskgit/training_vqgan.py
 ```

3. 训练bidirectional transformer。整文件```maskgit/training_transformer.py```里的超参数，并设置数据集的路径。运行下面命令来训练模型。

 ```bash
 python maskgit/training_transformer.py
 ```

 ## Finetune on diffusion model

1. 首先配置并激活环境```maskgit/finetune/environment.yaml```。

```bash
conda env create -f maskgit/finetune/environment.yaml
conda activate ldm
```

2. 然后在该文件中下载好预训练模型的参数。其中Stable Diffusion v1.4的参数可在https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/tree/main中下载。

3.最后依次运行```maskgit/finetune/main.sh``` 的命令即可对模型实现微调。