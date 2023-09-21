# FDCNet: Feature Drift Compensation Network for Class-Incremental Weakly Supervised Object Localization
Official pytorch implementation of "FDCNet: Feature Drift Compensation Network for Class-Incremental Weakly Supervised Object Localization" \
Our great base code is from [BAS](https://github.com/wpy1999/BAS)

# Paper
[FDCNet: Feature Drift Compensation Network for Class-Incremental Weakly Supervised Object Localization](https://arxiv.org/pdf/2309.09122v1.pdf), Sejin Park, Taehyung Lee, Yeejin Lee, Byeongkeun Kang, https://arxiv.org/pdf/2309.09122v1.pdf, ACM International Conference on Multimedia (MM), Ottawa, Canada, 2023

# Abstract

This work addresses the task of class-incremental weakly supervised object localization (CI-WSOL). The goal is to incrementally learn object localization for novel classes using only image-level annotations while retaining the ability to localize previously learned classes. This task is important because annotating bounding boxes for every new incoming data is expensive, although object localization is crucial in various applications. To the best of our knowledge, we are the first to address this task. Thus, we first present a strong baseline method for CI-WSOL by adapting the strategies of class-incremental classifiers to mitigate catastrophic forgetting. These strategies include applying knowledge distillation, maintaining a small data set from previous tasks, and using cosine normalization. We then propose the feature drift compensation network to compensate for the effects of feature drifts on class scores and localization maps. Since updating network parameters to learn new tasks causes feature drifts, compensating for the final outputs is necessary. Finally, we evaluate our proposed method by conducting experiments on two publicly available datasets (ImageNet-100 and CUB-200). The experimental results demonstrate that the proposed method outperforms other baseline methods.

<p align="center"><img src="https://github.com/Vision-sejin/FDCNet/assets/117714660/acfe1b2d-173c-4d89-9073-ebb57a60de3c/ovv.png"width="700" height="350"/>
  
# Results

<div>
  <img src="https://github.com/Vision-sejin/FDCNet/assets/117714660/c6382aff-1e9e-4a51-9934-84f410768d0b/graph.png"width="400" height="400"/>
  <img src="https://github.com/Vision-sejin/FDCNet/assets/117714660/2e1397d8-64c2-4ab0-b864-f12a74d289ad/cam.png"width="500" height="400"/>
</div>
<p align="center"><img src="https://github.com/Vision-sejin/FDCNet/assets/117714660/1a6a6cdb-c98a-490e-abe3-3359e268391d/table.png"width="800" height="700"/>

  
# Inference

```
python FDCNet_inference.py
```

# Citation


```
@inproceedings{FDCNet,
  title={FDCNet: Feature Drift Compensation Network for Class-Incremental Weakly Supervised Object Localization},
  author={Sejin Park, Taehyung Lee, Yeejin Lee, Byeongkeun Kang},
  journal={[arXiv preprint arXiv:2309.09122v1]},
  year={2023}
}
