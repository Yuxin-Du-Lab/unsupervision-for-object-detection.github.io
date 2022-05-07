## 关于MoCo的琐碎

### 1. Abstract

#### 1.1. MoCo的counterpart

counterpart指的是使用相同的网络(e.g. Res50), 仅仅是训练方法不一样(supervision / self-supervison)。

#### 1.2. Downstream tasks

Transfer well. Outperform its supervised pre-train counterpart in 7 detection/segmentation task on PASCAL VOC, COCO and other dataset.(sometimes even surpassing it by large margins)

第一个在如此多的下游主流任务中表现如此好的无监督模型。

### 2. Introduction

#### 2.1. Compare to NLP

NPL的组成是离散的语义空间（词），语义信息明显，易于tokenize，通过字典进行建模，易于训练和优化

CV的组成是连续且高维的空间，语义信息不明显，不容易通过字典进行建模，故CV方向的无监督学习进展缓慢。

#### 2.2. 对比学习

![2022-05-07 14-15-07 的屏幕截图](https://github.com/Yuxin-Du-Lab/unsupervision-for-object-detection.github.io/blob/gh-pages/images/2022-05-07%2014-15-07%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

依据instance discrimination对图片x_1产生x11和x12，x11作为anchor通过encoder1提取特征f11，x12作为positive通过encoder2提取特征f12。而其余的图片x2,x3...均认为是negative,通过encoder2（因为encoder1对应的是anchor）得到特征f2,f3...。对比学习的目的是希望在特征空间中positive pairs(e.g. f11 和 f12)尽可能近，而negatvie pairs(e.g. f11和f2,f3...)尽可能远。

#### 2.3. Dictionary与对比学习

