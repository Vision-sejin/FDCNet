# [ACM MM'23] FDCNet: Feature Drift Compensation Network for Class-Incremental Weakly Supervised Object Localization
This is code base for the following paper:

### [FDCNet: Feature Drift Compensation Network for Class-Incremental Weakly Supervised Object Localization](https://arxiv.org/pdf/2309.09122v1.pdf)
Sejin Park, Taehyung Lee, Yeejin Lee, Byeongkeun Kang, ACM International Conference on Multimedia (MM'23), Ottawa, Canada, 2023, https://arxiv.org/pdf/2309.09122v1.pdf

Our base code is from [BAS](https://github.com/wpy1999/BAS) and please read our paper for details

# Abstract

This work addresses the task of class-incremental weakly supervised object localization (CI-WSOL). The goal is to incrementally learn object localization for novel classes using only image-level annotations while retaining the ability to localize previously learned classes. This task is important because annotating bounding boxes for every new incoming data is expensive, although object localization is crucial in various applications. To the best of our knowledge, we are the first to address this task. Thus, we first present a strong baseline method for CI-WSOL by adapting the strategies of class-incremental classifiers to mitigate catastrophic forgetting. These strategies include applying knowledge distillation, maintaining a small data set from previous tasks, and using cosine normalization. We then propose the feature drift compensation network to compensate for the effects of feature drifts on class scores and localization maps. Since updating network parameters to learn new tasks causes feature drifts, compensating for the final outputs is necessary. Finally, we evaluate our proposed method by conducting experiments on two publicly available datasets (ImageNet-100 and CUB-200). The experimental results demonstrate that the proposed method outperforms other baseline methods.

<p align="center"><img src="https://github.com/Vision-sejin/FDCNet/assets/117714660/acfe1b2d-173c-4d89-9073-ebb57a60de3c/ovv.png"width="700" height="350"/>
  
# Results

<p align="center"><img src="https://github.com/Vision-sejin/FDCNet/assets/117714660/1a6a6cdb-c98a-490e-abe3-3359e268391d/table.png"width="800" height="700"/>

  
# Inference
Baseline weight [Baseline](https://drive.google.com/file/d/143Z9M6EejuLaLj9ZVlJFXpm_1yyvEZFP/view?usp=sharing) \
FDC weight [Inference](https://drive.google.com/file/d/1mA_gWo9j2WIWTUPz6WkDjTb-RwU7ixKM/view?usp=sharing) 


Inference code:

```
python FDCNet_inference.py
```

# Citation


```
@inproceedings{10.1145/3581783.3612450,
author = {Sejin Park, Taehyung Lee, Yeejin Lee, and Byeongkeun Kang},
title = {FDCNet: Feature Drift Compensation Network for Class-Incremental Weakly Supervised Object Localization},
year = {2023},
isbn = {9781450392037},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3581783.3612450},
doi = {10.1145/3581783.3612450},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
location = {Ottawa, Canada},
series = {MM '23}
}
```
