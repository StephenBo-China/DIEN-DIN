# DIEN-DIN

本项目使用tensorflow2.0复现阿里兴趣排序模型DIEN与DIN。

DIN论文链接: https://arxiv.org/pdf/1706.06978.pdf

DIEN论文链接: https://arxiv.org/pdf/1809.03672.pdf

数据集使用阿里数据集测试模型代码, 数据集链接: https://tianchi.aliyun.com/dataset/dataDetail?dataId=56 

# 调用方法:

## 0. 简介:

DIEN的输入特征中主要包含三个部分特征: 用户历史行为序列, 目标商品特征, 用户画像特征。
用户历史行为序列需包含点击序列与非点击序列。
请按如下1~2方法处理输入特征。

## 1. 初始化:

初始化DIEN时需传入5个参数:

(注:feature_list中的特征名称,需要与embedding_dict中的特征名称一样)

- embedding_count_dict:string->int格式,该变量记录需要embedding各个特征的词典个数,即最大整数索引+ 1的大小;

- embedding_dim_dict:string->int格式,该变量记录需要embedding各个特征的输出维数,即密集嵌入的尺寸;

- embedding_features_list:list(string)格式,该变量记录DIEN中user_profile部分所有需要embedding的feature名称;

- user_behavior_features:list(string)格式,该变量记录DIEN中user_behavior与target_item部分所有需要embedding的feature名称

- activation:string格式,默认值"PReLU",该变量空值全连接层激活函数,”PReLU“->PReLU,"Dice"->Dice

## 2. 模型调用：

模型调用需传入6个参数:

(注:feature_list中的特征名称,需要与dict中的特征名称一样)

- user_profile_dict:dict:string->Tensor格式,记录user_profile部分的所有输入特征的训练数据;

- user_profile_list:list(string)格式,记录user_profile部分的所有特征名称;
            
- click_behavior_dict:dict:string->Tensor格式,记录user_behavior部分所有点击输入特征的训练数据;
            
- noclick_behavior_dict:dict:string->Tensor格式,记录user_behavior部分所有未点击输入特征的训练数据;
            
- target_item_dict:dict:string->Tensor格式,记录target_item部分输入特征的训练数据;
            
- user_behavior_list:list(string)格式,记录user_behavior部分的所有特征名称。

# 调用演示代码：

## DIEN:

DIEN_train_example.ipynb

## DIN:

DIN_train_example.ipynb

# 代码:

- model.py: 定义模型代码

- layers.py: 自定义层

- loss.py: 定义Auxiliary Loss用到的NN

- activations.py: 定义Dice激活函数

- alibaba_data_reader.py: 输入数据处理函数(代码中使用数据已用spark处理后得到了所需序列数据, 及特征embedding词典数)
