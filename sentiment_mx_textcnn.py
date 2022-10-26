#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import os
import random
import time
import tarfile
import mxnet as mx
from mxnet import gluon, nd

VOCAB_SIZE = 20000
MAX_SENTENCE_LEN = 512


def download_imdb(data_dir='./data'):
    url = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    sha1 = '01ada507287d82875905620988597833ad4e0903'
    fname = gluon.utils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, 'r') as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, data_dir)


def read_imdb(folder='train'):
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join('./data/aclImdb/', folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data


def get_tokenized_imdb(data):
    return [[tok.lower() for tok in review.split(' ')] for review, _ in data]


def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # default size is 46152
    counter = collections.Counter(dict(counter.most_common(VOCAB_SIZE)))
    return mx.contrib.text.vocab.Vocabulary(counter, min_freq=5,
                                            reserved_tokens=['<pad>'])


def preprocess_imdb(data, vocab):
    def pad(x):
        return x[:MAX_SENTENCE_LEN] if len(x) > MAX_SENTENCE_LEN else x + [
            vocab.token_to_idx['<pad>']] * (MAX_SENTENCE_LEN - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([score for _, score in data])
    return features, labels


class TextCNN(gluon.nn.Block):
    """ batch_size = 256
        sentence_length = 512
        vocabulary_size = 20000
        embedding_size = 100

        input: (256, 512)          -> (512, 256, 20000) implicit one_hot
        embedding: (20000, 100)    -> (256, 512, 100)
        concat:                    -> (256, 512, 200)
        transpose:                 -> (256, 200, 512)
        conv:       -> (256, 100, 510) (256, 100, 509) (256, 100, 508)
        pool:       -> (256, 100, 1)   (256, 100, 1)   (256, 100, 1)
        flatten:    -> (256, 100)      (256, 100)      (256, 100)
        concat:     -> (256, 3*100)
        decoder:    -> (256, 2)
    """

    def __init__(self, embed_size, kernel_sizes, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = gluon.nn.Embedding(VOCAB_SIZE, embed_size)
        # 不参与训练的嵌入层
        self.constant_embedding = gluon.nn.Embedding(VOCAB_SIZE, embed_size)
        self.dropout = gluon.nn.Dropout(0.5)
        self.decoder = gluon.nn.Dense(2)
        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = gluon.nn.GlobalMaxPool1D()
        self.convs = gluon.nn.Sequential()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(gluon.nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # 将两个形状是(批量大小, 词数, 词向量维度)的嵌入层的输出按词向量连结
        embeddings = nd.concat(
            self.embedding(inputs), self.constant_embedding(inputs), dim=2)
        # 根据Conv1D要求的输入格式，将词向量维，即一维卷积层的通道维，变换到前一维
        embeddings = embeddings.transpose((0, 2, 1))
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1) NDArray
        # flatten函数去掉最后一维，然后在通道维上连结
        encoding = nd.concat(
            *[nd.flatten(self.pool(conv(embeddings))) for conv in self.convs],
            dim=1)
        # 应用丢弃法后使用全连接层得到输出
        outputs = self.decoder(self.dropout(encoding))
        return outputs


def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    print('training on', ctx)
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, loss_count, m, start = 0.0, 0.0, 0, 0, time.time()
        for features, labels in train_iter:  # (256, 512) (256,)
            with mx.autograd.record():
                y_hats = net(features)
                l = loss(y_hats, labels)  # (256, 2) (256,)

            l.backward()
            trainer.step(features.shape[0])

            train_loss_sum += l.sum().asscalar()
            loss_count += l.size
            train_acc_sum += (y_hats.argmax(axis=1) == labels).sum().asscalar()
            m += labels.size
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec' % (
            epoch + 1, train_loss_sum / loss_count, train_acc_sum / m, time.time() - start))


batch_size = 256
train_data, test_data = read_imdb('train'), read_imdb('test')
vocab = get_vocab_imdb(train_data)
feature, label = preprocess_imdb(train_data, vocab)
train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(
    *preprocess_imdb(train_data, vocab)), batch_size, shuffle=True)
test_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(
    *preprocess_imdb(test_data, vocab)), batch_size)

embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
ctx = [mx.cpu()]
net = TextCNN(embed_size, kernel_sizes, nums_channels)
net.initialize(mx.init.Xavier(), ctx=ctx)

lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)
