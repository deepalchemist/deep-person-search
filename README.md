## A PyTorch based strong baseline for person search
This repository develops a baseline model with high performance, and 
implements the widely used baseline OIM [1], and NAE [4].

- Separating detection and re-ID head on the top of model
- PK sampling for training re-ID head 
- Warm-up training 

### About this repository
- It is pure PyTorch code, which requires the PyTorch version >= 1.1.0
- It supports multi-image batch training.
- End-to-end training and evaluation. Both PRW and CUHK-SYSU are supported.
- Standard protocol (including PRW-mini in [3]) used by most research papers
- Highly extensible (easy to add models, datasets, training methods, etc.)
- Visualization tools (proposals, losses in training)
- High performance baseline.

![img_with_rois](./image/rois.png)

### Preparation

- Clone repository and build the environment 
```
git clone https://github.com/DeepAlchemist/deep-person-search.git && cd deep-person-search
conda env create -f env.yml
```

- Download [PRW](https://github.com/liangzheng06/PRW-baseline) and [CUHK-SYSU](https://github.com/ShuangLI59/person_search) (also named SSM)
to `path_to_your_data`.

- In the config file `./lib/cfg/config.py`, change `--data_root` to `path_to_your_data`, 
`--ckpt_dir` to `path_you_want_to_save_the_checkpoints`.

- Download ImageNet pre-trained ResNet models from [GoogleDrive](https://drive.google.com/drive/folders/1VEaUg1W79iDYtK8292D1_V99rZKDklop?usp=sharing)
to `deep-person-search/cache/pretrained_model/`

### Training & Evaluation

- Training

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --benchmark ssm --batch_size 5 \
    --backbone bsl --in_level C5 --cls_type oim \
    --lr 0.003 --warmup_epochs 1 --max_epoch 7 \
    --suffix "" 
```
`-dis` enable display (visualization), then `tensorboard --bind_all --logdir your_log_dir`, which shows the loss curves and the input image with proposals.

- Evaluation

```
CUDA_VISIBLE_DEVICES=0 python main.py --is_test \
    --benchmark ssm --eval_batch_size 5 \
    --backbone bsl --in_level C5 --cls_type oim \
    --load_ckpt "absolute_path_to_your_checkpoint" \
```

- Comparison

| |    | CUHK-SYSU  |  | PRW |
| ---- |  :----:  | :----:  | :----:  | :----:  |
| Method |  mAP   | rank1  | mAP   | rank1  |
| OIM [1] | [88.1](https://drive.google.com/file/d/1Im4o0d7hytno-aycSDPHgNkxnJN785v0/view?usp=sharing)  | 89.2 | [36.0](https://drive.google.com/file/d/1l7eKIwOYJxEopguMk_tl8bKQW0f83PJv/view?usp=sharing) | 76.7 |
| NAE [4] | [89.8](https://drive.google.com/file/d/1mCCEnvwQC8Ckn7ElIJFMGqvfZMD6MX1P/view?usp=sharing)  | 90.7 | [37.9](https://drive.google.com/file/d/1zUGlmIoScRR_qFhDGdl8jtmR9cy3YrBF/view?usp=sharing) | 77.3 |
| baseline | [90.0](https://drive.google.com/file/d/17ViFt0rFNXupSNri1DvEhSFpebtqa4Xl/view?usp=sharing) | 91.0 | [40.5](https://drive.google.com/file/d/1H3f2C5GplCxxsxtKgdtdzRsX9mincwC8/view?usp=sharing) | 81.3 |

The download link of the [trained models](https://drive.google.com/drive/folders/1k5fUtsakAiDbJcfCbpnpgfHlZW-JYLCN?usp=sharing) are available in the table. Note that all the models are trained with image size of `600x1000`, the larger image size, e.g., `900x1500`, would yield better performance.

### TODO
- DistributedDataParallel
- Training with larger image size, i.e., 900x1500
- Supporting more SOTA methods
- Visualizing ranking list in test
- A technological report for this repository

[1] Joint Detection and Identification Feature Learning for Person Search. In CVPR 2017.<br>
[2] Person Re-Identification in the Wild. In CVPR 2017.<br>
[3] Query-guided End-to-End Person Search. In CVPR 2019.<br>
[4] Norm-Aware Embedding for Efficient Person Search. In CVPR 2020.<br>
