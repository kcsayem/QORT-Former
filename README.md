# [AAAI 2025] QORT-Former: Query-optimized Real-time Transformer for Understanding Two Hands Manipulating Objects
[Elkhan Ismayilzada](https://elkhanzada.github.io/)\*, [MD Khalequzzaman Chowdhury Sayem](https://kcsayem.github.io/)\*,  [Yihalem Yimolal Tiruneh](http://linkedin.com/in/yihalem-yimolal-tiruneh-852aab198), [Mubarrat Tajoar Chowdhury](https://sites.google.com/view/mubarrat-chowdhury), [Muhammadjon Boboev](https://sites.google.com/view/boboevm/home), [Seungryul Baek](https://sites.google.com/site/bsrvision00/)

<sub>\* Equal contribution.</sub>
## Installation
```bash
git clone https://github.com/kcsayem/QORT-Former.git
```
## Create a conda environment
```bash
conda create -n "qort_former" python=3.10
```
## Install PyTorch
```bash
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
```
## Install requirements
```bash
pip install -r requirements.txt
```
## Data Preparation
Download [FPHA](https://guiggh.github.io/publications/first-person-hands/), [H2O](https://taeinkwon.com/projects/h2o/) datasets.
## Pretrained weights
Pretrained weights are available for [H2O](https://drive.google.com/file/d/1lMZdr7X4Ze1jjY-Tt8rbtbL0uO8IhwG3) and FPHA ($${\color{green}\text{Coming soon...}}$$).
## Inference
```bash
python main.py --model_path {path_to_model} --source {image_folder}
```
The predictions will be generated and saved in the ```{image_folder}/predictions``` directory.
## Acknowledgements
This repository is based on
* [FastInst](https://github.com/junjiehe96/FastInst)
* [H2OTR](https://github.com/chohoseong/H2OTR)
* [GraFormer](https://github.com/Graformer/GraFormer/)
* [DETR](https://github.com/facebookresearch/detr)
