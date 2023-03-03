# SI

高分大赛——面向海洋一号可见光图像中海冰目标监测 -> http://gaofen-challenge.com

### 数据处理

1. 数据清洗

训练集共1500个样本，大小分布（张）：
| 512*512 | 1024*1024 | 2048*2048 |
| :-: | :-: | :-: |
| 1412 | 66 | 22 |

经过对比发现存在部分样本存在问题：
- 残缺或者有划线（37）
- 图片与标签对应有疑点（16或者更多，这里我只挑选了16张）

去掉之后，大小分布（张）：
| 512*512 | 1024*1024 | 2048*2048 |
| :-: | :-: | :-: |
| 1359 | 66 | 22 |

下一步是裁剪，将其裁剪至大小均为`512*512`，即`1024*1024`为4张图，`2048*2048`为16张图：
于是样本总数为 `1359 + 66*4 + 22*16 = 1359 + 264 + 352 = 1975`

其次发现数据集中负样本/困难负样本较多，正负样本不均衡，于是对其进行统计：
| 负样本（gt为全黑） | 其他 |
| :-: | :-: |
| 1500 | 475 |

2. 数据增强

注：以下处理只针对正样本，负样本保持不变
- 由于样本为遥感图像，故对其进行`90 180 270`三个角度的旋转，在对其进行镜像处理，同样进行`90 180 270`三个角度的旋转，于是数据集扩充为8倍
- 之后对扩充8倍的数据集进行`brightness saturation constast`的随机改变，每次选择其中一种改变，改变的量也为随机，于是数据集扩充为16倍

此时我们有了3种不同比例的数据集：
| | 原始 | 正样本8倍 | 正样本16倍 |
| :-: | :-: | :-: | :-: |
| 比例 | 1 : 3.15 | 2.53 : 1 | 5.06 : 1|

之后会对不同比例的数据集进行验证

- 有文章指出多尺度训练甚至比前几种数据增强的总和还要好

### 模型

1. deeplab

   - lr = 0.001
   - Lovasz loss
   - scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

2. hrocr

   - lr = 0.001
   - Lovasz loss
   - scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
   - **虽然是平行连接多个CNN，但是最终计算loss时只用第一个高分辨率子网络的输出，即out_aux**

3. segmenter
   
    - lr = 0.00001
    - Lovasz loss
    - scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    - 对学习率十分敏感
    
    - 官方代码给了2种decoder架构：`DecoderLinear`、`MaskTransformer`
    
    - 问题
    
      - 排除错误：因为源代码使用了`rearrange`函数进行reshape，而我直接进行reshape，导致loss不下降
    
      - 解决方案：先交换维度，再reshape就可以
    
      ```python
      x = x.permute(0,2,1)
      x = x.reshape(1, self.n_cls, GS, -1)
      ```

4. segformer

   - lr = 0.00001
   - Lovasz loss
   - scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

   - encoder有 B0-B5 共6种情况，模型逐渐变大，指标也逐渐提升

5. 待补充

### 损失函数

1. CE
2. Focal
3. Dice
4. Lovasz
5. 以上组合

实验结果
对deeplab而言，使用4种loss以及其组合进行实验：

|  | CE | Focal | Dice | Lovasz | CE+Dice | Focal+Dice |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| train fwiou | 0.9690 | 0.9589 | 0.9567 | 0.9628| 0.9673 | 0.9650 |
| valid fwiou | 0.9293 | 0.9229 | 0.9133 | 0.9248| 0.9363 | 0.9362 |
| test fwiou(%) | - | - | - | 95.1960 | 95.1658 | - |

### 超参

- train / valid = 8 / 2
- n_epoch = 300
- batch_size = 8（模型为transformer架构时需要调整为1）
- transform = transforms.Compose([transforms.ToTensor()])（默认）
- optimizer = optim.Adam(model.parameters())（默认）
- 训练策略
  - 多尺度训练，分别选择1.25x、1x、0.75x，共3种情况
    - 依次循环进行训练
    - 每一个epoch随机选择其中一个尺度进行训练（ours）
  - OHEM，适配了各种loss
    - 在150轮之后加入OHEM
    - 其中Lovasz loss为特殊的class

### 后处理

1. 多模型融合
   - 软投票
   - 硬投票
   - 如果选择的模型本身指标不高，会导致低指标模型将高指标模型性能拉低的情况
   
2. CRF
   - 只有不同推理次数的CRF
   - CRF与原始预测图片加权融合
   - 结果不确定
   
3. 旋转预测

   - 分数低的模型提升

   - 分数高的模型下降

4. 裁剪预测（对于1024与2048的尺寸，裁剪预测再拼接）
   - 直接裁剪拼接
   - 重叠裁剪拼接（需要避免分块效应）
   - 都有提高，但是直接裁剪效果比重叠裁剪好

### 文件说明

```
以下文件夹为模型模块
- deeplab
- hrocr
- segformer
- segmenter

README.md    —readme
build.py     -建立部分
dataset.py   —读取数据集
losses.py    —各种损失函数
main.py      —程序入口
metrics.py   —各种评价指标
param.py     —参数和路径
preposs.py   —对数据集预处理
solver.py    —训练和验证
visualize    —可视化
```

