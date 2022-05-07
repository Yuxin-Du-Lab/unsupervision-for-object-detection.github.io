## 关于MoCo

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

![对比学习](https://github.com/Yuxin-Du-Lab/unsupervision-for-object-detection.github.io/blob/gh-pages/images/2022-05-07%2014-15-07%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

依据instance discrimination对图片x1产生x11和x12，x11作为anchor通过encoder1提取特征f11，x12作为positive通过encoder2提取特征f12。而其余的图片x2,x3...均认为是negative,通过encoder2（因为encoder1对应的是anchor）得到特征f2,f3...。对比学习的目的是希望在特征空间中positive pairs(e.g. f11 和 f12)尽可能近，而negatvie pairs(e.g. f11和f2,f3...)尽可能远。

#### 2.3. Dictionary query与对比学习

![Dictionary query](https://github.com/Yuxin-Du-Lab/unsupervision-for-object-detection.github.io/blob/gh-pages/images/2022-05-07%2014-36-27%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

将sampled & encoded(2)的f11, f2, f3....都作为keys，f11作为query，那么对比学习就转换成了字典查询的问题。问题就转变为，训练encoders进行dictionary look-up，使得：**an encoded "query" should be similar to its matching key and dissimilar to others.**

MoCo作者认为先前的对比学习工作都可以看作上述的Dictionary query思路。

#### 2.4. Dictionary Keys 的特点/需求

* 大。大，query就更可能学到具有区分性的语义信息，而不是short cut
* 一致。keys不通过同一个（或尽可能相似的）encoder产生，那么query就可能匹配到和query-encoder最近似的key-encoder产生的key，也相当于short cut

当时的方法都至少受限制于二者之一。

> MoCo:
>
> 通过维护queue，使得Dictionary大，且不断更新，维持一致；
>
> 通过使用momentum使key-encoder缓慢更新，也维持了一致性。

#### 2.5. MoCo的灵活性

MoCo只是提供了一个dynamic dictionary和momentum key-encoder，可以很好地应用各种pre-task。

### 3. Discussion

数据集从ImageNet(1M)变成IG(1B)后，模型的效果有提升，但提升不大，也就一个点不到。**作者认为大的数据没有被充分利用，更好的pre-task也许是解决的途径。**

MoCo和masked auto-encoding结合起来（类似于NLP的BERT）。

>  即MAE

### 4. Related Work

**无监督可做的方向：pre-task(define ground-truth)；loss func(metric diff)**

#### 4.1. Loss Func

生成式网络。自回归：将一个图片encode，然后再decode重建该图，然后对比前后的差异作为loss

判别式网络。将图片打成九宫格，给定中间的格，再随机给一个周围的格子，模型预测其所属位置。（pre-task -> 分类任务）

对比学习。训练过程中，对比学习的目标动态变化，而生成式和判别式目标不变。

对抗学习。衡量的是概率分布之间的差异(?)。先前用于无监督数据生成，后来也用于特征学习（认为既然能生成不错的fake image，那么应该学到了实际数据的分布）

#### 4.2. Pre-task

Denoising auto-encoder：重建整张图

Context auto-encoder：重建某个patch

Colorization：给图片上色

Tracking：视频相关

Clustering

### 5. Method

#### 5.1. Loss Design

##### 5.1.1. SoftMax

![softmax](https://github.com/Yuxin-Du-Lab/unsupervision-for-object-detection.github.io/blob/gh-pages/images/MommyTalk1651910281146.jpg)

##### 5.1.2. Cross Entropy Loss

![Cross Entropy Loss](https://github.com/Yuxin-Du-Lab/unsupervision-for-object-detection.github.io/blob/gh-pages/images/MommyTalk1651910311233.jpg)

supervised情况下，分母中的k对应的是数据集中的类别数(ImageNet~1,000)。

而在UN-supervised情况下，每张图片自成一类(Instance Discrimination)，k就是图片总数(ImageNet~128w)，难以计算

##### 5.1.3. Noise Constrastive Estimation(NCE Loss)

将这么多类看作二分类，分为data sample & noisy sample，即Noise Constrastive。但仍然将其他所有图片都看作noisy sample时，计算量并没有减少，故使用采样Estimation。而采样越多（对应MoCo等对比学习中的dictionary越大），Estimation越准确，结果越好。

##### 5.1.3. InfoNCE

认为将那么多negative classes都看作一个noisy sample类，不是很好，还是作为多分类任务比较合适。

![InfoNCE](https://github.com/Yuxin-Du-Lab/unsupervision-for-object-detection.github.io/blob/gh-pages/images/MommyTalk1651911058736.jpg)

tau是hyper-parameter，用来改变特征的分布，很有讲究。tau越大，则这些参与计算的特征变换后越小，分布越均匀；反之更peak。

tau过大，模型对所有负样本一视同仁，学习没有重点；tau过小，模型只关注特别困难的样本（但实际这些负样本中可能是存在正样本的），模型难以收敛/不容易泛化。

