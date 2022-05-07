## 关于MoCo的琐碎

### 1. Abstract

#### 1.1. MoCo的counterpart

counterpart指的是使用相同的网络(e.g. Res50), 仅仅是训练方法不一样(supervision / self-supervison)。

#### 1.2. Downstream tasks

Transfer well. Outperform its supervised pre-train counterpart in 7 detection/segmentation task on PASCAL VOC, COCO and other dataset.(sometimes even surpassing it by large margins)

第一个在如此多的下游主流任务中表现如此好的无监督模型。

#### 1.3 linear protocol

将pre-train model作为backbone并freeze，只训练后面的fc层，用来评估pre-train model的效果

### 2. Introduction

#### 2.1. Compare to NLP

NPL的组成是离散的语义空间（词），语义信息明显，易于tokenize，通过字典进行建模，易于训练和优化

CV的组成是连续且高维的空间，语义信息不明显，不容易通过字典进行建模，故CV方向的无监督学习进展缓慢。

#### 2.2. 对比学习

![2022-05-07 14-15-07 的屏幕截图](https://github.com/Yuxin-Du-Lab/unsupervision-for-object-detection.github.io/blob/gh-pages/images/2022-05-07%2014-15-07%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

依据instance discrimination对图片x1产生x11和x12，x11作为anchor通过encoder1提取特征f11，x12作为positive通过encoder2提取特征f12。而其余的图片x2,x3...均认为是negative,通过encoder2（因为encoder1对应的是anchor）得到特征f2,f3...。对比学习的目的是希望在特征空间中positive pairs(e.g. f11 和 f12)尽可能近，而negatvie pairs(e.g. f11和f2,f3...)尽可能远。

#### 2.3. Dictionary query与对比学习

![2022-05-07 14-36-27 的屏幕截图](https://github.com/Yuxin-Du-Lab/unsupervision-for-object-detection.github.io/blob/gh-pages/images/2022-05-07%2014-36-27%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

将sampled & encoded(2)的f11, f2, f3....都作为keys，f11作为query，那么对比学习就转换成了字典查询的问题。问题就转变为，训练encoders进行dictionary look-up，使得：**an encoded "query" should be similar to its matching key and dissimilar to others.**

MoCo作者认为先前的对比学习工作都可以看作上述的Dictionary query思路。

#### 2.4. Dictionary Keys 的特点/需求

* 大。大，query就更可能学到具有区分性的语义信息，而不是short cut
* 一致。keys不通过同一个（或尽可能相似的）encoder产生，那么query就可能匹配到和query-encoder最近似的key-encoder产生的key，也相当于short cut

> MoCo:
>
> 通过维护queue，使得Dictionary大，且不断更新，维持一致；
>
> 通过使用momentum使key-encoder缓慢更新，也维持了一致性。

