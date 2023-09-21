# FDCNet: Feature Drift Compensation Network for Class-Incremental Weakly Supervised Object Localization
Official pytorch implementation of "FDCNet: Feature Drift Compensation Network for Class-Incremental Weakly Supervised Object Localization" 
Our base code is from [BAS](https://github.com/wpy1999/BAS)

# PAPER
[FDCNet: Feature Drift Compensation Network for Class-Incremental Weakly Supervised Object Localization](https://arxiv.org/pdf/2309.09122v1.pdf), Sejin Park, Taehyung Lee, Yeejin Lee, Byeongkeun Kang, https://arxiv.org/pdf/2309.09122v1.pdf, ACM International Conference on Multimedia (MM), Ottawa, Canada, 2023

This work addresses the task of class-incremental weakly supervised object localization (CI-WSOL). The goal is to incrementally learn object localization for novel classes using only image-level annotations while retaining the ability to localize previously learned classes. This task is important because annotating bounding boxes for every new incoming data is expensive, although object localization is crucial in various applications. To the best of our knowledge, we are the first to address this task. Thus, we first present a strong baseline method for CI-WSOL by adapting the strategies of class-incremental classifiers to mitigate catastrophic forgetting. These strategies include applying knowledge distillation, maintaining a small data set from previous tasks, and using cosine normalization. We then propose the feature drift compensation network to compensate for the effects of feature drifts on class scores and localization maps. Since updating network parameters to learn new tasks causes feature drifts, compensating for the final outputs is necessary. Finally, we evaluate our proposed method by conducting experiments on two publicly available datasets (ImageNet-100 and CUB-200). The experimental results demonstrate that the proposed method outperforms other baseline methods.

<p align="center"><img src="https://github.com/Vision-sejin/FDCNet/assets/117714660/a7888df8-c05c-45f4-a90f-7a3475d3d409/overview.png"width="700" height="350"/>

# RESULTS
<p align="center"><img src="https://github.com/Vision-sejin/FDCNet/assets/117714660/0c115d88-0f3f-4522-bff8-3d8e8ceb9e8c/cam.png"width="700" height="350"/>

# CITATION
@inproceedings{FDCNet,
  title={FDCNet: Feature Drift Compensation Network for Class-Incremental Weakly Supervised Object Localization},
  author={Seji Park, Taehyung Lee, Yeejin Lee, Byeongkeun Kang},
  journal={https://arxiv.org/pdf/2309.09122v1.pdf},
  year={20213}
}
