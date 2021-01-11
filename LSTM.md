# Deep understanding of LSTM data

How to preprocess data for LSTM algorithm?

### Scene 1: covid 19 deaths predict.

一个地区，9天以来死亡人数，和其他数据信息，预测第10天死亡人数。

#### 1. input matrix

  m 条数据，n个维度，第一列是date，最后一列是Y。


#### 2. data preprocess

  step = 10

  *把10行数据去掉date列，flatten成一行，最末列Y，其他列为X*

  第二个10行，组成第二条X，Y。

  ...

  到最后10 行。

  总X 大约是 m/10 行，10n - 10 -1 列，记为p 行q列。总Y 长度为 p = m/10。

  **p 行数据时间上是顺序相连的。**

  假设：第n 天的Y，跟前 n - step 天都相关。

  如果都不相关，即X 的n - step 天中，其他不含Y 的维度可以独立推出 Y，上面的preprocess略不同。

#### 3. put data to LSTM

  LSTM 一个input，对应一个output和一个中间记忆状态。

  p行q列数据，p个input，产生p个output，计算loss(output, Y)，反向传播求导，改变LSTM网络参数值。

  **(p, q) reshape (p, 1, q) 放入LSTM，LSTM网络初始化要用到q 是input_size，1 是batch_size。**

  因为死亡人数增长的训练不依赖于其他的国家地区，反向传播仅在一个国家内，换国家地区，则中间记忆状态要detach，重置在RNN中记忆状态参数，同时提高训练速度。预测时，每个国家初始化中间记忆状态。


### Scene 2: Sentiment Analysis

#### 1. input matrix:

  LSTM自然语言处理，用词预测词。当前预测词 = LSTM(当前输入词 + 之前所有输入词产生的记忆状态)

  在物理上，跟Scene 1 对比，数据没有X，只有Y一个维度，就是词。

  输入连续的词，如一本书，把词切割成m条语料，每个语料n个词。一般不足n的padding 0，超过n的截断。又如IMDB 评论，每条评论是一条语料，每个语料大约500个词。

#### 2. data preprocess:

  m 条语料，一条n 个词，m 太大，一次载入 batch_size 个。

  词用词向量表示word vector，词向量维度100，则 (batch_size, n) 转换到 (batch_size, n, 100)。

  **LSTM模型输入数据，第一维需要时序上连续，第二维是batch_size，第三维是input_size，数据reshape (n, batch_size, 100)**

#### 3. put data to LSTM:

  LSTM 一个input，对应一个output和一个中间记忆状态。

  n 个词，n个input，n个output，中间output都丢掉了，只用最后一个output算loss和反向传播，如果是双向LSTM，只保留第一个和最后一个output。**IMDB sentiment label 是正负的分类，计算loss(output[-1], label)**，反向传播求导，改变LSTM网络参数值。

  重置中间记忆状态，这样反向传播只在一句一段之内，提高训练速度。预测时，每条数据初始化中间记忆状态。


### stateful and stateless

LSTM有输入门，输出门，遗忘门，记忆细胞组成6个公式。

遗忘门可以对上一时刻记忆细胞选择留下多少，进而达到“中间记忆状态重置”的效果，如果训练自然语言，在句子末尾加上<eos>，让LSTM遗忘门学到这个符号的含义。

> Tensorflow Keras 的LSTM有stateful 选项。stateful就是一直保存中间状态，stateless就是一批或几批训练后，重置中间记忆状态。为了提高反向传播的效率，一般都是stateless，有些情况比如预测第1001句话就需要stateful。
>
> model.reset_states()来重置模型中所有层的状态，layer.reset_states()来重置指定有状态 RNN 层的状态。