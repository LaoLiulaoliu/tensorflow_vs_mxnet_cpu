import collections
import os
import random
import tarfile
import time
import mxnet as mx
from mxnet import gluon, nd

VOCAB_SIZE = 20000
MAX_SENTENCE_LEN = 256


def download_imdb(data_dir='./data'):
    url = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    sha1 = '01ada507287d82875905620988597833ad4e0903'
    fname = gluon.utils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, 'r') as f:
        f.extractall(data_dir)


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


class BiRNN(gluon.nn.Block):
    """ batch_size = 256
        sentence_length = 512
        vocabulary_size = 20000
        embedding_size = 64
        num_hiddens = 128

        input:     (256, 512)  -> (512, 256, 20000) one_hot
        Embedding: (20000, 64) -> (512, 256, 64) each time 256 words
        LSTM:      (64, 2*128) -> (512, 256, 2*128)
        concat:                -> (256, 4*128)
        decoder:               -> (256, 2)
    """

    def __init__(self, embed_size, batch_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = gluon.nn.Embedding(VOCAB_SIZE, embed_size)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = gluon.rnn.LSTM(num_hiddens, num_layers=num_layers,
                                      bidirectional=True, input_size=embed_size)
        self.decoder = gluon.nn.Dense(2)

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs.T)
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs = self.encoder(embeddings)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入，因为是双向，单向只取末尾。
        # 它的形状为(批量大小, 4 * 隐藏单元个数)。
        encoding = nd.concat(outputs[0], outputs[-1])
        outs = self.decoder(encoding)
        return outs


class HybridBiRNN(gluon.nn.HybridBlock):
    '''Hybridize way is deprecated:
       epoch 1, loss 0.6527, train acc 0.633, time 634.5 sec
       epoch 2, loss 0.4570, train acc 0.790, time 290.4 sec
       epoch 3, loss 0.2517, train acc 0.901, time 107.1 sec
       epoch 4, loss 0.1390, train acc 0.951, time 107.9 sec
       epoch 5, loss 0.0836, train acc 0.971, time 108.8 sec

       However BiRNN is OK:
       epoch 1, loss 0.6653, train acc 0.606, time 643.7 sec
       epoch 2, loss 0.5267, train acc 0.741, time 400.7 sec
       epoch 3, loss 0.3837, train acc 0.833, time 110.8 sec
       epoch 4, loss 0.2392, train acc 0.907, time 112.9 sec
       epoch 5, loss 0.1641, train acc 0.940, time 110.4 sec

    '''

    def __init__(self, embed_size, batch_size, num_hiddens, num_layers, **kwargs):
        super(HybridBiRNN, self).__init__(**kwargs)
        self.embedding = gluon.nn.Embedding(VOCAB_SIZE, embed_size)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = gluon.rnn.LSTM(num_hiddens, num_layers=num_layers,
                                      bidirectional=True, input_size=embed_size)
        self.decoder = gluon.nn.Dense(2)
        self.after_lstm_shape = (batch_size, num_hiddens * num_layers)

    def hybrid_forward(self, F, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(F.transpose(inputs))
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs = self.encoder(embeddings)
        # return (F.slice(outputs, -1, -2, -1), outputs)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入，因为是双向，单向只取末尾。
        # 它的形状为(批量大小, 4 * 隐藏单元个数)。
        encoding = F.concat(F.reshape(F.slice(outputs, 0, 1), self.after_lstm_shape),
                            F.reshape(F.slice(outputs, -1, -2, -1), self.after_lstm_shape))
        outs = self.decoder(encoding)
        return outs


def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        for X, y in zip(batch.data, batch.label):
            acc_sum += (net(X.T).argmax(axis=1) == y).sum()
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n


def train(features, labels, net, loss, trainer, ctx, batch_size, num_epochs):
    print('training on', ctx)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        train_iter = mx.io.NDArrayIter(features, labels, batch_size,
                                       shuffle=False,
                                       last_batch_handle='discard')
        for i, batch in enumerate(train_iter):
            with mx.autograd.record():
                y_hats = [net(X) for X in batch.data]  # [(256, 512)]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, batch.label)]  # [(256, 2)]，[(256,)]

            for l in ls:
                l.backward()
            trainer.step(len(batch.label[0]))

            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()
                                  for y_hat, y in zip(y_hats, batch.label)])
            m += sum([y.size for y in batch.label])
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec' % (
        epoch + 1, train_l_sum / n, train_acc_sum / m, time.time() - start))


def sentiment():
    batch_size = 256
    train_data = read_imdb('train')
    vocab = get_vocab_imdb(train_data)
    features, labels = preprocess_imdb(train_data, vocab)
    print('Training entries: {}, vocabulary size: {}'.format(len(train_data), len(vocab)))

    embed_size, num_hiddens, num_layers, ctx = 64, 128, 2, [mx.cpu()]
    net = HybridBiRNN(embed_size, batch_size, num_hiddens, num_layers)
    net.initialize(mx.init.Xavier())
    net.hybridize()

    lr, num_epochs = 1e-2, 10
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    train(features, labels, net, loss, trainer, ctx, batch_size, num_epochs)
    net.save_parameters('../saved_model/train_mxnet_weights')


def predict_sentiment(net, vocab, sentence):
    sentence = nd.array(vocab.to_indices(sentence))
    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
    return 'positive' if label.asscalar() == 1 else 'negative'


def load_net():
    batch_size = 256
    train_data, test_data = read_imdb('train'), read_imdb('test')
    vocab = get_vocab_imdb(train_data)

    embed_size, num_hiddens, num_layers, ctx = 64, 128, 2, [mx.cpu()]
    print('begin to load...')
    time.sleep(3)

    start = time.time()
    net = HybridBiRNN(embed_size, batch_size, num_hiddens, num_layers)
    net.load_parameters('../saved_model/train_mxnet_weights')
    print('load weights model cost: ', time.time() - start)

    sentence = 'When I watched this movie I went to sleep'

    loaded = time.time()
    result = predict_sentiment(net, vocab, sentence)
    # for i in range(99): predict_sentiment(net, vocab, sentence)
    print(f'predict one cost: {time.time() - loaded}s, ', result)


if __name__ == '__main__':
    sentiment()
    # load_net()
    # CPU 347% MEM 24%, 1303s
