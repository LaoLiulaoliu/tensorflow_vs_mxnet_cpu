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


def lstm_net(num_hidden=128, num_layers=2):
    stack = mx.rnn.SequentialRNNCell()
    for i in range(num_layers):
        stack.add(mx.rnn.LSTMCell(num_hidden=num_hidden, prefix=f'lstm_l{i}_'))


def sym_gen(seq_len, embed_size=64, num_hidden=128):
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    embed = mx.sym.Embedding(data=data, input_dim=VOCAB_SIZE, output_dim=embed_size, name='embed')

    outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=True)

    pred = mx.sym.Reshape(outputs, shape=(-1, num_hidden))
    pred = mx.sym.FullyConnected(data=pred, num_hidden=VOCAB_SIZE, name='pred')

    label = mx.sym.Reshape(label, shape=(-1,))
    pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')
    return pred, ('data',), ('softmax_label',)


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


def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    print('training on', ctx)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for i, batch in enumerate(train_iter):
            with mx.autograd.record():
                y_hats = [net(X) for X in batch.data]  # (256, 512)
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, batch.label)]  # (256, 1)，(256,)
            for l in ls:
                l.backward()
            trainer.step(len(batch.label))

            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])  # 256
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()
                                  for y_hat, y in zip(y_hats, batch.label)])
            # print([y.size for y in batch.label])
            m += sum([y.size for y in batch.label])
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / 1, time.time() - start))


def sentiment():
    batch_size = 256
    train_data, test_data = read_imdb('train')
    vocab = get_vocab_imdb(train_data)

    print('Training entries: {}, Test entries: {}, vocabulary size: {}'.format(len(train_data), len(test_data),
                                                                               len(vocab)))

    train_iter = mx.io.NDArrayIter(*preprocess_imdb(train_data, vocab), batch_size, shuffle=True,
                                   last_batch_handle='discard')
    test_iter = mx.io.NDArrayIter(*preprocess_imdb(test_data, vocab), batch_size, last_batch_handle='discard')

    embed_size, num_hiddens, num_layers, ctx = 64, 128, 2, [mx.cpu()]
    net = BiRNN(embed_size, num_hiddens, num_layers)
    net.initialize(mx.init.Xavier())

    lr, num_epochs = 1e-2, 10
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

    train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)
    net.save_parameters('../saved_model/train_mxnet_weights')


def predict_sentiment(net, vocab, sentence):
    sentence = nd.array(vocab.to_indices(sentence), ctx=[mx.cpu()])
    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
    return 'positive' if label.asscalar() == 1 else 'negative'


def load_net():
    batch_size = 256
    train_data, test_data = read_imdb('train'), read_imdb('test')
    vocab = get_vocab_imdb(train_data)

    embed_size, num_hiddens, num_layers, ctx = 64, 128, 2, [mx.cpu()]
    print('begin to load...')
    time.sleep(5)

    start = time.time()
    net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
    net.load_parameters('../saved_model/train_mxnet_weights')
    print('load weights model cost: ', time.time() - start)

    sentence = 'When I watched this movie I went to sleep'

    loaded = time.time()
    result = predict_sentiment(net, vocab, sentence)
    for i in range(99): predict_sentiment(net, vocab, sentence)
    print(f'predict one cost: {time.time() - loaded}s, ', result)


if __name__ == '__main__':
    sentiment()
    # CPU 347% MEM 24%, 1303s
