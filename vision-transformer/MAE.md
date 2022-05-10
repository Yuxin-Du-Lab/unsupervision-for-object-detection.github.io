* 取名题目：做的比较大，叫scalable;做的比较快：efficient
* auto-encoder，auto不是自动的意思，是“自”的意思，e.g.自回归模型。特点是标号y(label)和你的输入x是来自同一个东西。例如NLP里面预测下一个词。这里的auto就是强调本文相对于CV领域其他用text做label的工作，是用图片自身（像素）来做label的
* 题目模板：Your Work is/are adj. n (....是一个好同志)，客观，凝练
* 相比vit作者最后说到的：自监督效果不是那么好，还是得有标号的、大规模的数据集进行训练才可以效果比较好；本文用vit-huge在image net 1k上仅通过自监督就能达到87.8%的准确度，而且在迁移学习上的表现也非常好。**比起vit，本文主要提出了三点改进，一个是mask更多的patch来改进任务的难度；另一个是用一个transformer based decoder来还原原图，使整个流程简单；加了vit后面的各种技术，使得训练更加robust**
* Tensorflow实现，128个tpu v3训练一天，代码尚未公布

--------------

![2022-05-11 01-30-44 的屏幕截图](https://github.com/Yuxin-Du-Lab/unsupervision-for-object-detection.github.io/blob/gh-pages/images/2022-05-11%2001-30-44%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

----------------

### 预训练阶段：

![img](https://i0.hdslb.com/bfs/note/7294f04595235a39b9b66a62ec67fb40510578e3.png)

mask掉75%的patch，将剩余的patch flatten丢掉encoder里面，得到特征向量。然后把原图应该有的所有patches位置 flatten，如果一个patch没有被mask掉。那么就把特征向量填到对应位置；如果被mask掉，那么只需要填位置编码就好。然后把这个东西（蓝+灰）丢到decoder里面，试图重建原图（预训练任务）。

首先mask 75%是一个非常有挑战的任务，使得模型不会走trivial solution；另外encoder只需要计算没有mask 的patch， 加速了3～4倍。最后encoder的模型是比较大的，是关键的部分；而decoder不是那么重要，用了一个比较简单的模型。最后只要编码器就可以了。

---------

### Introduction

作者认为之前BERT思路迁移到CV不太好的原因：

* 卷积不容易做mask，原因是如果mask掉一块像素值，卷积窗口扫来扫去，无法区分这个mask的边界，后面就难以还原这个patch。（**vit工作解决了这个问题**）
* NLP中，语义信息的密度相当高，一个词就有很多的语义信息（这就是为什么BERT不能mask太多的东西，语言完形填空本身就是一个比较难的任务）。但是在CV图像中，信息比较冗余，如果简单mask掉一个patch，模型很容易通过周围邻居插值来填补，而不会去学更多的语义信息，相当于走了一个trivial solution。**作者说，他有一个很simple的strategy，就是mask掉足够高比率的patch，来加大任务的挑战性，迫使模型学习更好的特征。**
* decoder的问题。就像第2点里面说的，NLP中要预测的是词，而词的语义层级比较高，所以decoder用一个简简单单的MLP就可以做得很好（对应CV里面做图像分类/目标检测）。但是要预测patch，就需要对pixel级别进行decode，这就不好做（对应语义分割任务，需要一个很大的解码器，例如转置的卷积神经网络）。作者的想法是，使用一个非对称（encoder看到的和decoder看到的不一样，encoder只能看到没有被mask的patch）的encoder-decoder结构，目的是减小encoder的计算量，加速训练（大量的patch都被mask，不用计算）

作者用的是vit-large/huge和image net 1k的数据，就能达到vit里面需要一百倍以上大小的训练数据才能达到的效果，而且是无标号的。下游任务迁移效果也非常好。

----

### Method

encoder就是vit，但是训练时只看可见的patch。

decoder也是一个transformer，能看到encoder产生的可见patch的语义向量和不可见patch的token，而且要加上位置编码。做下游任务时，decoder丢到就好。decoder很小，不到encoder计算量的1/10。构建patch： decoder最后一层是linear层，如果patch size是16x16，那么linear层就把decoder的输出投影成一个256维度的向量，然后reshape成16x16

loss用的MSE：像素的预测值和真实值相减，在平方和。（和BERT一样，只在mask对应的patch上计算MSE）

有个小trick，把要预测的mask patch里面的pixel做norm，使其均值为0方差为1，训练起来更稳定。（那预测的时候怎么办？decoder直接丢到所以不关心？）

简单的实现：将图片切成patches，然后每个patches投射成一个token，然后排成一列，**shuffle（随机了）**，然后只保留头部的25%token feed into encoder。encoder提取到这些token的特征以后，把Mask token（就是一些可学习的向量）跟在这些encoder得到的特征后面，然后**unshuffle**，加上位置编码，feed into decoder

通过shuffle和unshuffle的操作，不需要稀疏操作，非常简单快捷

----------

### EXP

* **对VIT加入强的正则化，训练稳定了，点数提6个点（VIT说需要很大的数据表现才比较好，但是大家发现其是加上合适的正则项，在小的数据集上表现也可以很好）**

* aug，随机大小crop效果不错

* decoder只用1个transformer block效果就不错

* 微调所有和只微调最后的线性层效果差距非常大，微调所有效果好很多。调比较后面的层就ok，前面的层比较底层，调了效果也一般

    

