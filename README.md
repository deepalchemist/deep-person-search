Person search

This repository implements the widely used baseline OIM [1], NAE [4].
Meanwhile, we develop a baseline with high performance.

- Separating detection and re-ID head on the top of model
- PK sampling for training re-ID head 
- Data Augmentation (paste the same person into different backgrounds)
- Warm-up training 
- GMP, box-weight=10

##### About this repository
- It is pure PyTorch code, which requires the PyTorch version >= 1.1.0
- It supports multi-image batch training.
- End-to-end training and evaluation. Both PRW and CUHK-SYSU are supported.
- Standard protocol (including PRW-mini in [3]) used by most research papers
- Highly extensible (easy to add models, datasets, training methods, etc.)
- Visualization tools (proposals, losses in training)
- High performance baseline.

##### TODO
- DistributedDataParallel
- Trained model and performance
- Visualizing ranking list in test
- A technological report for this repository

[1] Joint Detection and Identification Feature Learning for Person Search. In CVPR 2017.

[2] Person Re-Identification in the Wild. In CVPR 2017.

[3] Query-guided End-to-End Person Search. In CVPR 2019.

[4] Norm-Aware Embedding for Efficient Person Search. In CVPR 2020.
