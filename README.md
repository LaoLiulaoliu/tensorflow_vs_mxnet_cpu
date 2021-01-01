## Tensorflow vs MXNet in cpu

深度学习框架，CPU下轻量、高效的框架。

### 平台选择

现有主流深度学习框架：Tensorflow, PyTorch, PaddlePaddle, MXNet。比较出名的Caffe, Theano, CNTK已经停止迭代。

网上资料显示：Tensorflow 易部署，PyTorch 易实现算法，PaddlePaddle 在中文NLP方面有优势，MXNet 轻量化灵活。

Tensorflow的Keras，MXNet的Gluon 实现LSTM预测算法，并做如下对比：


| 环境 |   |
| ------ | ------ |
| CPU | Intel(R) Core(TM) i5 2.4 GHz Dual-Core |
| Memory |  8 GB 1600 MHz DDR3 |
| OS | macOS 10.15.7 |
| GPU | NA |
| Python | 3.7.0 |
| Tensorflow |  2.4.0 |
| MXNet |   1.7.0 |


| | Tensorflow | MXnet | 说明 |
| ---- | ---- | ---- | ---- |
| 模型训练平均时长 | one epoch(1077训练集 + 359验证集): 2.08s | one epoch(1077训练集 + 359验证集): 0.022s | 均关闭打印，MXNet 比Keras 快96倍 |
| 模型存储大小 | 136KB | 120KB |
| 模型加载时长 | 0.5124988555908203s | 0.003978729248046875s | MXNet 比Keras 快129倍 |
| 模型预测平均时长 | 0.0014372761867172538s | 1.6518622055691266e-05s | MXNet 比Keras 快87倍 |
| 模型训练资源占用 | CPU 202%, MEM 239M | CPU 420%, MEM 3.5%(420M) | MXNet 比 Keras 多用1.47倍CPU，单位CPU下还是快 |
| 收敛速度 | learning rate = 1e-4, 200步 r2 score: 0.5712 | learning rate = 1e-3, 200步 r2 score 0.3222 | learning rate需调 |
| LSTM vs GRU | LSTM keras: 189.9s | GRU keras: 187.3s | LSTM和GRU效率差不多 |
| LSTM vs GRU | LSTM mxnet: 5.86s | GRU mxnet: 6.55s | MXNet 比Keras 快30倍 |
| 代码复杂度 | API build model，fit, evaluate, predict调用 | API build model，初始化神经网络，Feed forward，back propagation要调用 | MXNet 比Keras 多一些流程控制代码 |
| 代码行数 | 28 | 35 |


##### 说明

     Tensorflow和 MXNet 均支持分布式、多GPU，支持嵌入式环境。
     Keras of Tensorflow是符号式编程，Gluon of MXNet 命令式和符号式编程都支持。符号式理论上Tensorflow 和MXNet 同等高效。

     猜想，Keras 是Tensorflow High Level封装，Tensorflow low level API 效率会不会高？Gluon 是MXNet High Level 封装，low level API Symbol 不知道是否更快？Oneflow 说是最先进最快的深度学习框架，不知当前场景下有多快？



##### 结论

     Tensorflow 和 MXNet 构建相同结构的基于LSTM 的神经网络，网络参数和大小相似，在训练、预测、模型存储和加载上，MXNet 比Keras 效率高出10~130倍。

     程序编写上，Keras跟Scikit-learn一样傻瓜化，MXNet 需要加入流程控制代码，这也体现其灵活性。

     IMDB评论做Sentiment Analysis 中，同样对比方法得到类似结论。

### Files

 1. IMDB sentiment analysis with Tensorflow and MXNet

    * sentiment_tensorflow.py
    * sentiment_mxnet.py
    * sentiment_mx_symbol.py (MXNet Symbol, not finished)

2. Time series algorithm with LSTM predition

    * tensorflow_vs_mxnet.py
