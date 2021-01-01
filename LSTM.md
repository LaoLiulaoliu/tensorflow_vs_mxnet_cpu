# Deep understanding of LSTM data

How to preprocess data for LSTM algorithm?

### Scene 1: covid 19 deaths predict.

一个地区，9天以来死亡人数，和其他数据信息，预测第10天死亡人数。

#### input matrix
m x n

m 条数据，n个维度，第一列是date，最后一列是Y。


#### data preprocess
step = 10

把10行数据去掉date列，flatten成一行，最末列Y，其他列为X。

第二个10行，组成第二条X，Y。

...

到最后10 行。

新数据X 大约是 m/10 行，10n - 10 -1 列，记为p 行q列。Y长度为 p = m/10。


假设：第n 天的Y，跟前 n - step 天都相关。

  如果都不相关，即X 的n - step 天中，其他不含Y 的维度可以独立推出 Y，上面的preprocess略不同。

#### put data to LSTM

LSTM 一个input，对应一个output和一个中间记忆状态。

p行q列数据，p个input，产生p个output，计算loss(output, Y)，反向传播求导，改变LSTM网络参数值。

(p, q) reshape (p, 1, q) 放入LSTM，LSTM网络初始化要用到q 是input_size，1 是batch_size。

中间记忆状态一直保持。如果换地区，则中间记忆状态要detach，从新在RNN中计算存储记忆。


### Scene 2: natural language processing

#### input matrix:

自然语言处理是用词预测词，LSTM中，当前预测词，是根据当前输入词，之前所有输入词产生的记忆状态，而生成。

输入的物理意义：只有一个Y，就是词，没有X。

输入连续的词，把词切割成m条语料，每个语料n个词。一般不足n的padding 0，超过n的截断。

m x n

例如：IMDB 评论，预处理成，每条评论是一条语料，每个语料500个词。

#### data preprocess:


#### put data to LSTM:


