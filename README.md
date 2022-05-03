# unsupervision-for-object-detection.github.io
本文主要记录无/弱/半监督的一些论文及其核心思想，同时关注其在目标检测领域的应用

|                            paper                             |   pub    |                          main idea                           |
| :----------------------------------------------------------: | :------: | :----------------------------------------------------------: |
| Leverage Your Local and Global Representations: A New Self-Supervised Learning Strategy.(LoGo) | CVPR2022 | 用**MLP**来取代cosine-sim作为local-local crops的相似度度量（能抓到更rich的local feature）。可附加于simsiam、moco等模型改善其效果 |
|  Exploring simple siamese representation learning.(simsiam)  | CVPR2021 | 针对一张图片的一对aug-views，交替对one of two branchs进行**stop-gradient** |
| Momentum contrast for unsupervised visual representation learning.(moco) | CVPR2020 |                                                              |
| Improved baselines with momentum contrastive learning.(moco v2) |   2020   |                                                              |
| UniVIP: A Unified Framework for Self-Supervised Visual Pre-training | CVPR2022 |                                                              |
| Revisiting the Transferability of Supervised Pretraining: an MLP Perspective | CVPR2022 |                                                              |

### 1. Siamese networks's undesired trivial solution

In un-/self-supervised representation learning field, methods generally involve certain forms of Siamese networks. An undesired trivial solution to Siamese networks is **all outputs “collapsing” to a constant**. There have been several general strategies for preventing Siamese networks from collapsing:

* Contrastive learning: add **negative pairs** ([SimCLR](http://proceedings.mlr.press/v119/chen20j.html), [Deep InfoMax](https://arxiv.org/abs/1808.06670) and [its multi-scale version](https://proceedings.neurips.cc/paper/2019/hash/ddf354219aac374f1d40b7e760ee5bb7-Abstract.html), [CMC](https://link.springer.com/chapter/10.1007/978-3-030-58621-8_45), [MoCo](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html), [MoCo v2](https://arxiv.org/abs/2003.04297))
* Clustering: incorporates **online clustering** ([SwAV](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html), [DeepCluster](https://openaccess.thecvf.com/content_ECCV_2018/html/Mathilde_Caron_Deep_Clustering_for_ECCV_2018_paper.html), [SeLa](https://arxiv.org/abs/1911.05371))
* Un-contrastive learning: **stop-gradient** operation for one branch([SimSiam](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html)); a **momentum encoder** ([BYOL](https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html))
* Transformer-based\*  [Dino network](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html)

