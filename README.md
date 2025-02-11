<h1 align="center">
TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models
for Robotic Manipulation</h1>


* **TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Modelsfor Robotic Manipulation** <br>
  [![arXiv](https://img.shields.io/badge/Arxiv-2402.03766-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2409.12514)
  


## 📰 News
* **`Feb. 9th, 2025`**: 🤟**TinyVLA**🤟 is <font color=red>accepted by IEEE Robotics and Automation Letters (RA-L) 2025 </font>!
* **`Nov. 19th, 2024`**: 🤟**TinyVLA**🤟 is out! **Paper** can be found [here](https://arxiv.org/abs/2409.12514). The **project web** can be found [here](https://tiny-vla.github.io/).

## Contents
- [Install](#install)
- [Pretrained VLM](#Pretrained-VLM)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

1. Clone this repository and navigate to diffusion-vla folder
```bash
git clone https://github.com/lesjie-wen/tinyvla.git
```

2. Install Package
```Shell
conda create -n tinyvla python=3.10 -y
conda activate tinyvla
pip install --upgrade pip  # 
pip install -e . # make sure you are in the root directory of this repository
# install act and diffusion
cd detr
pip install -e . 
# install llava-pythia
cd ../llava-pythia
pip install -e . 
```

## Data Preparation
1. Our data format is the same as [act](https://github.com/MarkFzp/act-plus-plus), so you need to transfer your data into h5py format. You can refer to the [rlds_to_h5py.py](https://github.com/lesjie-wen/Diffusion-VLA/blob/master/data_utils/rlds_to_h5py.py) which is used to transfer the data from rlds format to h5py format.
2. You have to add one entry in [constants.py](https://github.com/lesjie-wen/Diffusion-VLA/blob/master/aloha_scripts/constants.py) to specify the path of your data as follows.
```python
    'your_task_name':{
        'dataset_dir': DATA_DIR + '/your_task_path', # define the path of the dataset
        'episode_len': 1000, #max length of the episode,
        'camera_names': ['front', 'wrist'] # define the camera names which are used as the key when reading data
    }
```
## Pretrained-VLM
We construct the VLM backbone by integrating a series of tiny LLM([Pythia](https://github.com/EleutherAI/pythia)) into [Llava](https://github.com/haotian-liu/LLaVA) framework. We follow the standard training pipe line and data provided by [Llava](https://github.com/haotian-liu/LLaVA). All the weights of VLM used in our paper are listed as following: 
| Model | Usage | Link |
|-------|-------|------|
|Llava-Pythia(70M)|For TinyVLA-S||
|Llava-Pythia(410M)|For TinyVLA-M||
|Llava-Pythia(1.4B)|For TinyVLA-H||


## Train
The training script in [train.py](https://github.com/lesjie-wen/Diffusion-VLA/blob/master/scripts/train.sh).

1. Diffusion-VLA is designed to incorporate pre-trained Multimodal Models into a visuomotor learning framework, and we use the LLaVA-Pythia as our multimodal backbone, you can download its weight at [huggingface](https://huggingface.co/zxmonent/llava-phi).
You need to change `--model_name_or_path` and specify the `$LLM_MODEL_SIZE`. There is a large range model size from 14M to 3B, and you can choose the model size according to your computational resources.
2. You can specify the output folder by change `$OUTPUT`.
3. You need to change the `--task_name` to specify the dataset you want to train and `--task_name` must be the same in [constants.py](https://github.com/lesjie-wen/Diffusion-VLA/blob/master/aloha_scripts/constants.py).
4. You can change `--lora_module` to specify which part of the model you want to use the LoRA module. The default value is 'vit llm', which means both the vision and language part will be finetuned by lora.


## Evaluation

Due to the fact that different robotic environments have different evaluation settings, we only provides our evaluation script in [eval_real_franka.py](https://github.com/lesjie-wen/Diffusion-VLA/blob/master/eval_real_franka.py). You can refer to this script to evaluate your model in your own environment.

## Acknowledgement
We build our project based on:
- [LLaVA](https://github.com/haotian-liu/LLaVA): an amazing open-sourced project for vision language assistant
- [act-plus-plus](https://github.com/haotian-liu/LLaVA): an amazing open-sourced project for robotics visuomotor learning

## Citation

If you find Diffusion-VLA useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{
    @article{wen2024tinyvla,
    title={Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation},
    author={Wen, Junjie and Zhu, Yichen and Li, Jinming and Zhu, Minjie and Wu, Kun and Xu, Zhiyuan and Liu, Ning and Cheng, Ran and Shen, Chaomin and Peng, Yaxin and others},
    journal={arXiv preprint arXiv:2409.12514},
    year={2024}
}
```


