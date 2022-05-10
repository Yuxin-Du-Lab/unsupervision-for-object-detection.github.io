![teaser](https://github.com/microsoft/Swin-Transformer/raw/cbaa0d8707db403d85ad0e13c59f2f71cd6db425/figures/teaser.png)

* 证明transformer应用在CV，方方面面都没有问题

* 下游任务基本刷榜了，认为能够取代conv（但ConvNext有新的突破）

* MLPmixer需要去看一下
* 研究动机：为了达到CNN层级式的效果，提出了patch merging；为了减小计算复杂度，提出了基于window和shifted-window计算self-attention
* pytorch-image-models（zoo）就有现成的库

--------------

### Introduction

![2022-05-10 13-16-34 的屏幕截图](/home/yuxin/weak-supervision-for-object-detection.github.io/images/2022-05-10 13-16-34 的屏幕截图.png)

对密集预测型任务如检测和分割来说，多尺度特征是相当重要的。

* CNN目前多用的是FPN
* 分割任务U-net提出skip connection方法：后续上采样，不仅从bottle neck里拿特征，也从之前的层里拿特征。这样上采样的过程就能把特征恢复出来了
* 分割里常用的网络结构还有pspNet，DeepLab，相应处理多尺度的方法如空洞卷积，psp、aspp层

密集预测型任务目前多输入尺寸为1000x1000或800x800，vit计算复杂度太高。而且vit的注意力是在全局（每个patch之间）计算的，复杂度和输入尺寸成平方关系。

swin transformer引入CNN中的“局部相关”先验，只在每个窗口内部做自注意力，使复杂度和输入尺寸成线性关系

CNN具有多尺度特点的关键操作是pooling，pooling操作能增大每个卷积核的感受野。swin transformer模仿CNN的pooling操作，提出了patch merging操作，把相邻的小patch合成成大patch。这样操作的过程中，就形成了多尺度的特征图，自然地就可以丢给FPN、或者U-NET结构。即作者反复强调的，swin transformer可以作为一个general的backbone做各种下游任务。

![img](https://i0.hdslb.com/bfs/note/9e48cc65a71007125067bbc3880b5c2e7542ab2e.png)

**Figure2:**

红色的是一个window，而灰色的是一个基本计算单元patch。每次self-attention在一个window里面做，计算各个patch的相互attention。shift操作是，向右下平移windows，使得window能够框住和上一层不完全一样的patches，即一部分原来自己的patches，一部分其他windows里面的patches，达到一个cross window connection的效果。随着层的叠加，patch merging操作使得一个patch的感受野已经很大了，而windows shift又能够收集到其他部分的信息，故最后能够达到全局信息的收集。

既省内存，效果也好。

### Model

![img](https://i0.hdslb.com/bfs/note/4c5c28a58001f6caf7bbcde4e38975505a51203b.png)

输入图像是224x224x3。基本块patch size是4x4。

patch partition操作就是把原图分割成patches，分成了(224/4) x (224/4)个patch，即56 x 56个patch，每个patch的维度是4x4x3=48。

通过linear embeding前将56x56x48做flatten，变成3136x48，然后做线性映射，映射transformer能够接受的维度（这里swin-tiny对应的是96）,映射后是3136x96

**（上面这些抽象步骤在代码里就通过一个kernel size为4，stride也为4，输出通道为96的卷积层一步实现了）**

```python
self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
```

这里的3136x96，seq length是3136，计算代价目前是不能承受的，所以引入了window(每个window只有7x7=49个patch，即seq len就降低成了49，可以承受)，在window里面计算self-attention

-----

Transformer Block:

> 改进window之间的通信：dat

<img src="https://i0.hdslb.com/bfs/note/de79e43763a634933ea6eb1eae348a68b7e93e78.png" alt="img" style="zoom:50%;" />

每个BLOCK里面有两层，先做一个基于window的W-MSA，再做一个基于shifted-window的SW-MSA，这样就能使window和window之间有connection，可以交换信息，学到global。

这也是为什么swin-transformer里面的Transformer Block都是偶数

----

经过一个Transformer Block以后，输出还是56x56x96（3136x96）

要想像CNN那样有层次，必须有类似pooling的操作，作者这里提出了patch merging

![img](https://i0.hdslb.com/bfs/note/de9552743e43fb6cd1aa8850c98f6a745fc7428f.png)

patch merging的过程：

对输入特征图HxWxC，在HxW维度隔一个抽一次值，对H维度变成了2个H/2，对W维度变成了2个W/2，那么合起来就是4个H/2 x W/2 x C。然后在C的维度对这四个小特征图进行拼接，变成H/2 x W/2 x 4C。为了和CNN里面的pooling操作进行对齐（CNN里面的VGG、ResNet，经过pooling操作都是宽高变一半，通道数变2倍，但这里通道数变4倍了），对4C这个维度做一个1x1卷积，输出通道数为2C。

经过patch merging以后呢，维度从56x56x96变成了28x28x192

后面同理，过一次patch merging，再过一个swin transformer block（block可以多个堆积，因为经过block维度不变）

（swin transformer没有用vit里面的cls token，非常像CNN）

如果做分类的话，最后得到7x7x768以后，直接做一个GAP得到1x1x768，然后过一个线性层变成1x1000（如果是给image net 1k做分类的话），和CNN是一样的

如果做其他下游任务，同理接一个其他分类头即可，7x7x768就和CNN出来的特征图一样来用

-------

### 全局patch计算self-attention 和 window内部patch计算self-attention 的计算复杂度对比：

![img](https://i0.hdslb.com/bfs/note/7125d0ccb3c6561a60de490d2b85dfde4035bf69.png)

![multi-head att](https://github.com/Yuxin-Du-Lab/unsupervision-for-object-detection.github.io/blob/gh-pages/images/multi-head%20att.png)

<center>multi-head attention计算过程<center>

> AxB的矩阵和BxC的矩阵相乘，计算复杂度是A\*B\*C

全局共有HxW个patches，window内有MxM个patches

全局patch计算self-attention： 乘系数矩阵的过程复杂度是**3HWC\^2**，然后key和query相乘计算注意力权重的过程是**H\^2W^2C**，权重矩阵att weight和value相乘的复杂度是**H\^2W\^2C**，最后线性映射的复杂度是HWCD，如果取特值C=D，那么**HWC\^2**

求和就是**4\*H W C\^2 + 2\*H\^2 W\^2 C**，对应公式(1)

window内部patch计算self-attention：对于一个窗口内部，H和W就变成M和M，只需要把上述公式里面的HW替换即可，即：**4\*M\^2 C\^2 + 2\*M\^4 C**。

一共有H/M x W/M，把这个再乘到上面去，得到**4\*H W C\^2 + 2\*H W M\^2 C**，对应公式(2)

--------------

### 基于MASK计算（trick）

![img](https://i0.hdslb.com/bfs/note/9e48cc65a71007125067bbc3880b5c2e7542ab2e.png)

基础的window和shifted-window的问题：（以Figure2为例）

* 四个窗口变成九个窗口
* 形状大小不一，做不到批量计算self-attention

一个naive的想法是，给边上比较小的window padding一些0，补充成和中心一样大的window。但是这样计算复杂度就高了

作者提出了类似七巧板的平移拼接的办法：

![img](https://i0.hdslb.com/bfs/note/6cd42ec9cff69a1c8ace52a0e60f60a3e8af05f8.png)

把左上比较小的ABC块平移到右下（cyclic shift），然后就又得到了4个规整的window，做self-attention，然后再移动回去

但是这样的操作，破坏了空间分布，window里面的有一些patch之间不应该做self-attention。故采用mask的办法去掉算出来的但我们不想要权重部分。

![img](https://i0.hdslb.com/bfs/note/bd3aa5ded3e7605bb8b584d2a518e361d60e5c44.png)

-------

计算复杂度：

* swin-tiny～res50
* swin-small～res101

不同量级的模型主要区别是两个超参数：transformer维度C和transformer block数(非常类似于resnet)

--------------

### EXP

* 分类任务，不论是在image net 1k上训练还是在image net 22k上训练，都是在image net 1k上测试的

ref: https://www.bilibili.com/read/cv14877004?from=note

首先是分类上的实验，这里一共说了两种预训练的方式

第一种就是在正规的ImageNet-1K(128万张图片、1000个类)上做预训练
第二种方式是在更大的ImageNet-22K（1,400万张图片、2万多个类别）上做预训练
当然不论是用ImageNet-1K去做预训练，还是用ImageNet-22K去做预训练，最后测试的结果都是在ImageNet-1K的测试集上去做的，结果如下表所示

![img](https://i0.hdslb.com/bfs/note/e03d9e9a52b6cae89f6486b233c68b19025cb8b5.png)

* 上半部分是ImageNet-1K预训练的模型结果
* 下半部分是先用ImageNet-22K去预训练，然后又在ImageNet-1K上做微调，最后得到的结果
* 在表格的上半部分，作者先是跟之前最好的卷积神经网络做了一下对比，RegNet 是之前 facebook 用 NASA 搜出来的模型，EfficientNet 是 google 用NASA 搜出来的模型，这两个都算之前表现非常好的模型了，他们的性能最高会到 84.3
* 接下来作者就写了一下之前的 Vision Transformer 会达到什么效果，对于 ViT 来说，因为它没有用很好的数据增强，而且缺少偏置归纳，所以说它的结果是比较差的，只有70多
* 换上 DeiT 之后，因为用了更好的数据增强和模型蒸馏，所以说 DeiT Base 模型也能取得相当不错的结果，能到83.1
* 当然 Swin Transformer 能更高一些，Swin Base 最高能到84.5，稍微比之前最好的卷积神经网络高那么一点点，就比84.3高了0.2
* 虽然之前表现最好的 EfficientNet 的模型是在 600*600 的图片上做的，而 Swin Base 是在 384*384 的图片上做的，所以说 EfficientNet 有一些优势，但是从模型的参数和计算的 FLOPs 上来说 EfficientNet 只有66M，而且只用了 37G 的 FLOPs，但是 Swin Transformer 用了 88M 的模型参数，而且用了 47G 的 FLOPs，所以总体而言是伯仲之间
* 表格的下半部分是用 ImageNet-22k 去做预训练，然后再在ImageNet-1k上微调最后得到的结果
* 这里可以看到，一旦使用了更大规模的数据集，原始标准的 ViT 的性能也就已经上来了，对于 ViT large 来说它已经能得到 85.2 的准确度了，已经相当高了
* 但是 Swin Large 更高，Swin Large 最后能到87.3，这个是在不使用JFT-300M，就是特别大规模数据集上得到的结果，所以还是相当高的

接下来是目标检测的结果，作者是在 COCO 数据集上训练并且进行测试的，结果如下图所示

![img](https://i0.hdslb.com/bfs/note/28a418d86e5392e975f92a514d9d99dd6d5a9d67.png)

* 表2（a）中测试了在不同的算法框架下，Swin Transformer 到底比卷积神经网络要好多少，主要是想证明 Swin Transformer 是可以当做一个通用的骨干网络来使用的，所以用了 Mask R-CNN、ATSS、RepPointsV2 和SparseR-CNN，这些都是表现非常好的一些算法，在这些算法里，过去的骨干网络选用的都是 ResNet-50，现在替换成了 Swin Tiny
* Swin Tiny 的参数量和 FLOPs 跟 ResNet-50 是比较一致的，从后面的对比里也可以看出来，所以他们之间的比较是相对比较公平的
* 可以看到，Swin Tiny 对 ResNet-50 是全方位的碾压，在四个算法上都超过了它，而且超过的幅度也是比较大的
* 接下来作者又换了一个方式做测试，现在是选定一个算法，选定了Cascade Mask R-CNN 这个算法，然后换更多的不同的骨干网络，比如 DeiT-S、ResNet-50 和 ResNet-101，也分了几组，结果如上图中表2（b）所示
* 可以看出，在相似的模型参数和相似的 Flops 之下，Swin Transformer 都是比之前的骨干网络要表现好的
* 接下来作者又做了第三种测试的方式，如上图中的表2（c）所示，就是系统层面的比较，这个层面的比较就比较狂野了，就是现在追求的不是公平比较，什么方法都可以上，可以使用更多的数据，可以使用更多的数据增强，甚至可以在测试的使用 test time augmentation（TTA）的方式
* 可以看到，之前最好的方法 Copy-paste 在 COCO Validation Set上的结果是55.9，在 Test Set 上的结果是56，而这里如果跟最大的 Swin Transformer--Swin Large 比，它的结果分别能达到58和58.7，这都比之前高了两到三个点

第三个实验作者选择了语义分割里的ADE20K数据集，结果如下图所示

![img](https://i0.hdslb.com/bfs/note/50f121284978f95ec6805cfe3c406109792054d4.png)

* 上图表3里可以看到之前的方法，一直到 DeepLab V3、ResNet 其实都用的是卷积神经网络，之前的这些方法其实都在44、45左右徘徊
* 但是紧接着 Vision Transformer 就来了，那首先就是 SETR 这篇论文，他们用了 ViT Large，所以就取得了50.3的这个结果
* Swin Transformer Large也取得了53.5的结果，就刷的更高了
* 其实作者这里也有标注，就是有两个“+”号的，意思是说这些模型是在ImageNet-22K 数据集上做预训练，所以结果才这么好

消融实验，实验结果如下图所示

![img](https://i0.hdslb.com/bfs/note/f211abe5c107d0a044d2a8c95aa44ac31d0103fc.png)

* 上图中表4主要就是想说一下移动窗口以及相对位置编码到底对 Swin Transformer 有多有用
* 可以看到，如果光分类任务的话，其实不论是移动窗口，还是相对位置编码，它的提升相对于基线来说，也没有特别明显，当然在ImageNet的这个数据集上提升一个点也算是很显著了
* 但是他们更大的帮助，主要是出现在下游任务里，就是 COCO 和 ADE20K 这两个数据集上，也就是目标检测和语义分割这两个任务上
* 可以看到，用了**移动窗口**和**相对位置编码**以后，都会比之前大概高了3个点左右，提升是非常显著的，这也是合理的，**因为如果现在去做这种密集型预测任务的话，就需要特征对位置信息更敏感，而且更需要周围的上下文关系，所以说通过移动窗口提供的窗口和窗口之间的互相通信，以及在每个 Transformer block都做更准确的相对位置编码，肯定是会对这类型的下游任务大有帮助的**

