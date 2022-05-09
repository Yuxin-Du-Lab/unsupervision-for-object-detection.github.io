note link:

https://www.bilibili.com/read/cv14221668?from=note

----

Notes:

* paper with code可以查到某数据集上当前效果最好的模型

* transformer encoder的输出（N+1）x D ，1是class token。NLP里用这个class token做分类，CV里也可以用；但CV经典的方法是对NxD做GAP，得到Nx1，然后丢到MLP里做预测**（这样的效果和class token差不多，但需要好好调参，否则会掉点）**

* 位置编码1-D，2-D，relative都无所谓，但不加会稍微掉点（作者认为是因为patch的位置信息比pixel更容易被模型习得，所以不需要特别精细地设计位置编码）。编码信息也是随机初始化的，可学习的NxD
* CNN接在transformer前面，对图像做预处理的思路，在预训练数据集比较小的时侯效果比较好，数据量上来以后就差不多了
* ViT微调的局限之一：增大输入尺寸，若patch_size不变，那么patch num就多了，seq num也大了，位置编码就不好弄了（插值是一个临时的解决方案，torch自带的interpolate函数就能做）

-------------

### model

![img](https://i0.hdslb.com/bfs/note/338990ad8ed798b3ab23d34fd8b4253feeeb8739.png)

![img](https://i0.hdslb.com/bfs/note/3db1489b5b9093bf21d646d11c661c17e9d53415.png)

E就是一个线性层，把原始patch的1x(PxPxC)变换到embedding的1xD

x_class是一个全局的token，因为self-attention的原因，在训练时会与其他所有的token交互信息

E_pos是随机初始化的可学习的位置编码，(N+1) x D，即每个token的位置编码也是D维的，直接加到对应的embedding token上

MSA是多头注意力，LN是layer norm，注意残差连接

最后y=LN(z0)，z0对应的就是x_class这个token，就用它来预测，loss函数也作用于它

-----------

### Exp

![img](https://i0.hdslb.com/bfs/note/be2095e43aecdc4bc332192a469a74e7dda9bc6f.png)

**Figure3:**

灰色区域代表resnet50～152取得的范围

image net 1k时TF全面不行（没有先验知识，需要更多data），21k就差不多了，300M时就全面超越

* 如果你的数据集是image net及一下，还是老老实实用resnet，imagenet21k及以上时，TF可能效果会更好

**Figure4:**

Linear few-shot evaluation：把pre-train得到的模型当作特征提取器freeze掉，仅训练linear层的训练，然后做few-shot的评估（5-shot）

* 这个东西做起来非常快，适合**快速做大量消融实验**
* 如何用ViT做小样本学习，是有前途的方向

![img](https://i0.hdslb.com/bfs/note/d1ab3842c94ff42ee989a1d5c677a4592ec4b6a0.png)

**Figure5：**

灰色是resnet，蓝色是ViT，叉叉是CNN+ViT混合模型。可以看出：

* 同等模型复杂度情况下，transformer都要好于resnet
* 模型比较小的时候，混合模型效果非常好；随着模型复杂度上升，混合模型基本和vit一致，有时还不如。为什么CNN抽取的特征没有帮助TF更好地学习，这是一个问题。**如何预处理图像，如何做tokenization是非常重要的，后续有很多工作。**
* 随数据增加，resnet和vit还没有观察到饱和（但混合模型似乎饱和了）

![img](https://i0.hdslb.com/bfs/note/fef2df5a0da2ed19c555326315de02a48f6bf7de.png)

patch embedding学到的和卷积第一层的差不多，都是一些纹理和颜色。所以可以做 plausible basis functions

![img](https://i0.hdslb.com/bfs/note/88ea7e4ae7ad89196385bb9cf5254f5c1cb9d0e2.png)

1D可学习的位置编码真的会学到位置概念

![img](https://i0.hdslb.com/bfs/note/40d7fdd6828ecebbc2b8cbb7b59619922addec0d.png)

自注意力:

横轴对应network layer深度，纵轴是mean attention distance.

其中attention distance = l(a, b) * weight(a, b)。即图像中点a和点b实际距离l(a, b)乘以两者的attention weight(a, b)

每一列代表一个multi-attention的headers，用不一样的颜色的小圆点表示。

可以看出，前面的层attention既可以注意到比较近的pixel的信息，也可以注意到比较远的信息，即global的（和CNN的不同，CNN前几层是local的）

后面层距离都比较远，代表自注意力真的学到了语义信息，而不是依靠临近的像素信息来判断了

---------

自监督（后来的MAE）

masked patch prediction模仿bert，但效果不是特别好

对比学习是2020年大火的方法，CV自监督效果最好的方法

Mocov3，dino，都是用contrastive learning训练vit

