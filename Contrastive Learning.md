## 计算机视觉 - 对比学习(Ref to [mli-paper-reading](https://github.com/mli/paper-reading/))


| 已录制 | 年份 | 名字                                               | 简介                                                 |                                                         引用 |
| ------ | ---- | -------------------------------------------------- | ---------------------------------------------------- | -----------------------------------------------------------: |
| ✅      | 2018 | [InstDisc](https://arxiv.org/pdf/1805.01978.pdf)   | 提出实例判别和memory bank做对比学习                  | 885 ([link](https://www.semanticscholar.org/paper/Unsupervised-Feature-Learning-via-Non-parametric-Wu-Xiong/155b7782dbd713982a4133df3aee7adfd0b6b304)) |
| ✅      | 2018 | [CPC](https://arxiv.org/pdf/1807.03748.pdf)        | 对比预测编码，图像语音文本强化学习全都能做           | 2187 ([link](https://www.semanticscholar.org/paper/Representation-Learning-with-Contrastive-Predictive-Oord-Li/b227f3e4c0dc96e5ac5426b85485a70f2175a205)) |
| ✅      | 2019 | [InvaSpread](https://arxiv.org/pdf/1904.03436.pdf) | 一个编码器的端到端对比学习                           | 223 ([link](https://www.semanticscholar.org/paper/Unsupervised-Embedding-Learning-via-Invariant-and-Ye-Zhang/e4bde6fe33b6c2cf9d1647ac0b041f7d1ba29c5b)) |
| ✅      | 2019 | [CMC](https://arxiv.org/pdf/1906.05849.pdf)        | 多视角下的对比学习                                   | 780 ([link](https://www.semanticscholar.org/paper/Contrastive-Multiview-Coding-Tian-Krishnan/97f4d09175705be4677d675fa27e55defac44800)) |
| ✅      | 2019 | [MoCov1](https://arxiv.org/pdf/1911.05722.pdf)     | 无监督训练效果也很好                                 | 2011 ([link](https://www.semanticscholar.org/paper/Momentum-Contrast-for-Unsupervised-Visual-Learning-He-Fan/ec46830a4b275fd01d4de82bffcabe6da086128f)) |
| ✅      | 2020 | [SimCLRv1](https://arxiv.org/pdf/2002.05709.pdf)   | 简单的对比学习 (数据增强 + MLP head + 大batch训练久) | 2958 ([link](https://www.semanticscholar.org/paper/A-Simple-Framework-for-Contrastive-Learning-of-Chen-Kornblith/34733eaf66007516347a40ad5d9bbe1cc9dacb6b)) |
| ✅      | 2020 | [MoCov2](https://arxiv.org/pdf/2003.04297.pdf)     | MoCov1 + improvements from SimCLRv1                  | 725 ([link](https://www.semanticscholar.org/paper/Improved-Baselines-with-Momentum-Contrastive-Chen-Fan/a1b8a8df281bbaec148a897927a49ea47ea31515)) |
| ✅      | 2020 | [SimCLRv2](https://arxiv.org/pdf/2006.10029.pdf)   | 大的自监督预训练模型很适合做半监督学习               | 526 ([link](https://www.semanticscholar.org/paper/Big-Self-Supervised-Models-are-Strong-Learners-Chen-Kornblith/3e7f5f4382ac6f9c4fef6197dd21abf74456acd1)) |
| ✅      | 2020 | [BYOL](https://arxiv.org/pdf/2006.07733.pdf)       | 不需要负样本的对比学习                               | 932 ([link](https://www.semanticscholar.org/paper/Bootstrap-Your-Own-Latent%3A-A-New-Approach-to-Grill-Strub/38f93092ece8eee9771e61c1edaf11b1293cae1b)) |
| ✅      | 2020 | [SWaV](https://arxiv.org/pdf/2006.09882.pdf)       | 聚类对比学习                                         | 593 ([link](https://www.semanticscholar.org/paper/Unsupervised-Learning-of-Visual-Features-by-Cluster-Caron-Misra/10161d83d29fc968c4612c9e9e2b61a2fc25842e)) |
| ✅      | 2020 | [SimSiam](https://arxiv.org/pdf/2011.10566.pdf)    | 化繁为简的孪生表征学习                               | 403 ([link](https://www.semanticscholar.org/paper/Exploring-Simple-Siamese-Representation-Learning-Chen-He/0e23d2f14e7e56e81538f4a63e11689d8ac1eb9d)) |
| ✅      | 2021 | [MoCov3](https://arxiv.org/pdf/2104.02057.pdf)     | 如何更稳定的自监督训练ViT                            | 96 ([link](https://www.semanticscholar.org/paper/An-Empirical-Study-of-Training-Self-Supervised-Chen-Xie/739ceacfafb1c4eaa17509351b647c773270b3ae)) |
| ✅      | 2021 | [DINO](https://arxiv.org/pdf/2104.14294.pdf)       | transformer加自监督在视觉也很香                      | 200 ([link](https://www.semanticscholar.org/paper/Emerging-Properties-in-Self-Supervised-Vision-Caron-Touvron/ad4a0938c48e61b7827869e4ac3baffd0aefab35)) |

### 1. 百花齐放

#### 1.1. InstDisc

提出了Instance Discrimination上游任务，memory bank方法

##### 1.1.1. 灵感

![2022-05-07 18-48-41 的屏幕截图](/home/yuxin/weak-supervision-for-object-detection.github.io/images/2022-05-07 18-48-41 的屏幕截图.png)

将一张图片(lion)输入到supervised model中，得到的分类结果，high response对应的都是和该图像**看起来高度相近**的类别(leopard, jaguar, cheetah...)；low response对应的都是基本没关系的类别。

作者认为，让这些相似类别的图像能够聚集的原因，并不是它们有相似的语义label，而是其图片/object本身就是看起来非常相似。

作者提出的unsupervised方式，就是把按类别区分图片的思路推到极致，即每个图片成一类，目的就是学习特征近而将图片区分开来。

##### 1.1.2. 方法设计

![2022-05-07 18-57-50 的屏幕截图](/home/yuxin/weak-supervision-for-object-detection.github.io/images/2022-05-07 18-57-50 的屏幕截图.png)

**Instance Discrimination**：作者希望，将一个图片输入一个CNN backbone，然后提取特征，最后在特征空间中使每张图片的特征向量都分散得很开（更好地区分每张图片）。

使用contrastive learning的方式，正样本来自一张图片的不同aug，负样本（所有其他图片的特征）存在memory bank中。

##### 1.1.3. 一些细节

proximal regularization，对memoy bank里的特征进行**动量化**的更新（很类似于MoCo）。

超参数设置：

* loss中的temperature tau=0.07
* batch size=256
* NCE loss负样本数m=4096
* epoch=200
* learning rate=0.03

### 1.2. InvaSpread: 看作SimCLR前身

没有大量存负样本，正负样本来自同一个mini-batch。只使用一个encoder进行end-to-end学习。

#### 1.2.1. 灵感

![2022-05-07 19-13-30 的屏幕截图](/home/yuxin/weak-supervision-for-object-detection.github.io/images/2022-05-07 19-13-30 的屏幕截图.png)

最基本的contrasive learning的想法。同一张/相似的图片通过encoder(CNN)出来的feature相似（invariant 不变性），不同图片feature不同（spreading 分散开）。

#### 1.2.2. 方法设计

![2022-05-07 19-17-56 的屏幕截图](/home/yuxin/weak-supervision-for-object-detection.github.io/images/2022-05-07 19-17-56 的屏幕截图.png)

pre-task也是instance discrimination

**如何选取正负样本：**

输入图片x1,x2,...,xn，n=batch_size=256。分别augment后，又得到n个图片：x_hat1,x_hat2,...,x_hatn。对x1来说，正样本为x_hat1，负样本为x2,...,xn和x_hat2,...,x_hatn（所有其他图片）

**此处与InstDisc不同，InstDisc从memory bank中取4096/更多个负样本**

**本文只从mini-batch中选负样本，是为了可以只用一个encoder进行e2e的训练。（即MoCo中提到的，end-to-end的方式）**

### 1.3 CPC (不同的pre-task：contrastive predictive coding)

生成式的pre-task：预测型任务。通用结构，可以处理audio, image, text and reinforcement learning(强化学习)。

#### 1.3.1. 方法设计

![2022-05-07 21-10-04 的屏幕截图](/home/yuxin/weak-supervision-for-object-detection.github.io/images/2022-05-07 21-10-04 的屏幕截图.png)

t为当前时刻。将之前时刻的输入feed to encoder，得到特征（蓝色），将特征feed to 自回归模型g_ar（auto-regressive，e.ge RNN,LSTM）,得到输出c_t（context representation，代表上下文的特征表示）。用c_t预测未来的z_t,z_t+1.....。

> 什么是自回归模型？

正样本为未来输入x_t+1,x_t+2...通过encoder得到的feature z_t+1, z_t+2...。负样本可以任意选取其他输入，然后通过encoder产生即可（应当与预测结果不相似）

#### 1.3.2. General usage

以cv为例，输入可以换成图片的patches，然后自左上而右下地输入，并进行预测

### 1.4. CMC: Contrastive Multiview Coding

一个物体的多个视角（多个模态）都可以当作正样本。maximize mutual information.

选取NYU RGBD数据集，有4个VIEW：原始图像，深度信息，surface normal（表面法线），分割图像。

#### 1.4.1. 灵感

人对世界的感知是多样的，但语义信息一致。

#### 1.4.2. 方法

![2022-05-07 21-26-21 的屏幕截图](/home/yuxin/weak-supervision-for-object-detection.github.io/images/2022-05-07 21-26-21 的屏幕截图.png)

正样本：同一张图片的不同view；负样本：任意其他图片的view

