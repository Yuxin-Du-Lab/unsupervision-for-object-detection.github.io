# Unsupervision-for-object-detection.github.io
本文主要记录无/弱/半监督的一些论文及其核心思想，同时关注其在目标检测领域的应用

## 1. Siam-net


|                            paper                             |   pub    |                          main idea                           |
| :----------------------------------------------------------: | :------: | :----------------------------------------------------------: |
| Leverage Your Local and Global Representations: A New Self-Supervised Learning Strategy.(LoGo) | CVPR2022 | 用**MLP**来取代cosine-sim作为local-local crops的相似度度量（能抓到更rich的local feature）。可附加于simsiam、moco等模型改善其效果 |
|  Exploring simple siamese representation learning.(simsiam)  | CVPR2021 | 针对一张图片的一对aug-views，交替对one of two branchs进行**stop-gradient** |
| Momentum contrast for unsupervised visual representation learning.(moco) | CVPR2020 | 维护一个queue，存储过去mini-batch的represents，与batchsize decouple并得到一个大dictionary；以momentum的方式平滑更新key-encoder。从current mini-batch中构造positive pairs，从queue中构造negative pairs |
| Improved baselines with momentum contrastive learning.(moco v2) |   2020   |                                                              |
| UniVIP: A Unified Framework for Self-Supervised Visual Pre-training | CVPR2022 |                                                              |
| Revisiting the Transferability of Supervised Pretraining: an MLP Perspective | CVPR2022 | 预训练时，在encoder后面加MLP可以缓解encoder的overfitting，保留更多的intra-class variantion，改善后续迁移学习的效果 |

### 1.1. Siamese networks's undesired trivial solution

In un-/self-supervised representation learning field, methods generally involve certain forms of Siamese networks. An undesired trivial solution to Siamese networks is **all outputs “collapsing” to a constant**. There have been several general strategies for preventing Siamese networks from collapsing:

* ContraTransformer-basedstive learning: add **negative pairs** ([SimCLR](http://proceedings.mlr.press/v119/chen20j.html), [Deep InfoMax](https://arxiv.org/abs/1808.06670) and [its multi-scale version](https://proceedings.neurips.cc/paper/2019/hash/ddf354219aac374f1d40b7e760ee5bb7-Abstract.html), [CMC](https://link.springer.com/chapter/10.1007/978-3-030-58621-8_45), [MoCo](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html), [MoCo v2](https://arxiv.org/abs/2003.04297))
* Clustering: incorporates **online clustering** ([SwAV](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html), [DeepCluster](https://openaccess.thecvf.com/content_ECCV_2018/html/Mathilde_Caron_Deep_Clustering_for_ECCV_2018_paper.html), [SeLa](https://arxiv.org/abs/1911.05371))
* Un-contrastive learning: **stop-gradient** operation for one branch([SimSiam](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html)); a **momentum encoder** ([BYOL](https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html))
* Transformer-based\*  [Dino network](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html)



## 2. Pre-task design


|                            paper                             |    pub    | main idea |
| :----------------------------------------------------------: | :-------: | :-------: |
| UP-DETR: Unsupervised Pre-training for Object Detection with Transformers. (UP-DERT) | CVPR 2021 |           |
|   End-to-end object detection with transformers. (DERT) *    | ECCV 2020 |           |
| Unsupervised embedding learning via invariant and spreading instance feature.(Instance-based discrimination tasks) | IEEE 2019 |           |
| Unsupervised feature learning via non-parametric instance discrimination. (Instance-based discrimination tasks) | IEEE 2018 |           |
| Deep clustering for unsupervised learning of visual features. (clustering-based tasks) | ECCV 2018 |           |

**Instancebased discrimination tasks** and **clustering-based tasks** are two typical pretext tasks in recent studies. **UP-DETR** is a novel pretext task, which aims to pre-train transformers based *on the DETR architecture for object detection*.

### 好的模型2.1. instance discrimination

将一张图片x进行randomly crop并做augment后得到两个view: x1,x2 (Transformation),认为这两者similar,作为positive pair. 而数据集中其他所有图片都被认为和x1,x2是dissimilar, 作为negative pair.

(positive/negative定义非常灵活)



------

## Others

### 1. linear protocol

将pre-train model作为backbone并freeze，只训练后面的fc层，用来评估pre-train model的效果

