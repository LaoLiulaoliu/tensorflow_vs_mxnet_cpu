import pandas as pd
import numpy as np
import functools
import time
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow import keras

import mxnet as mx
from mxnet import gluon, nd

from sklearn.metrics import r2_score


def downsample_filter_normalization(fname):
    df = pd.read_csv(fname)
    df['time'] = df['time'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M', time.localtime(0.001 * x)))
    df = df.groupby('time').agg(lambda x: np.mean(x)).reset_index()
    df = df[(df['primary_air_rate'] > 0) & (df['primary_air_rate'] < 0.9)]  # in this data, no row lose

    columns_without_time = df.columns.to_list()
    columns_without_time.remove('time')

    Xmin = df[columns_without_time].min(axis=0)
    Xmax = df[columns_without_time].max(axis=0)
    Xnorm = (df[columns_without_time] - Xmin) / (Xmax - Xmin)

    return pd.concat([df['time'], Xnorm], axis=1)


def build_data(df):
    """inputs按分钟聚合，临近5min数据拼一行，每行数据时间上要求是顺序相连的。如果当前时间是t，则这一行包含[t-5, t] 的所有数据
        取出y，删除t-1, t-2 的y，和时间列。如果这一行有空值数据则删除此行。
    """
    steps = 5
    correlated = 3
    convert_strtime_datetime = functools.partial(datetime.strptime)
    time_column = list(map(lambda x: convert_strtime_datetime(x, '%Y-%m-%d %H:%M'), df['time']))
    Y = []

    data = None
    for t in range(len(time_column) - 1, steps - 1, -1):
        time_difference = timedelta(seconds=60)
        item = None
        for i in range(1, steps + 1):
            if time_column[t] - time_column[t - i] != time_difference * i:
                break
        else:
            item = df.iloc[t, 1:].values
            Y.append(item[-1])
            item = item[:-1]
            for i in range(1, steps + 1):
                if i >= correlated:
                    item = np.hstack((item, df.iloc[t - i, 1:].values))
                else:
                    item = np.hstack((item, df.iloc[t - i, 1:-1].values))

        if item is not None:
            data = np.vstack((data, item)) if data is not None else item
    return data, np.asarray(Y).reshape((-1, 1))


def gen_lstm_data(data, step=10):
    x, y =[], []
    for i in range(len(data) - step):
        x_temp = data[i:i+step, :-1]
        y_temp = data[i+step:i+step+1, -1]
        x.append(x_temp.tolist())
        y.append(y_temp.tolist())
    return np.array(x), np.array(y)


def build_lstm_model(X, learning_rate):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(64, input_shape=(X.shape[1], X.shape[2])))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Dropout(0.01))
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mae',
                  metrics=['accuracy'])
    return model


def train_keras(train_x, train_y, batch_size, num_epochs, learning_rate, validation_x=None, validation_y=None):
    model = build_lstm_model(train_x, learning_rate)
    model.summary()

    if validation_x is None:
        history = model.fit(train_x, train_y,
                            epochs=num_epochs,
                            batch_size=batch_size,
                            verbose=0,
                            shuffle=False)
    else:
        history = model.fit(train_x, train_y,
                            epochs=num_epochs,
                            batch_size=batch_size,
                            validation_data=(validation_x, validation_y),
                            verbose=0,
                            shuffle=False)
    model.save_weights('./keras.h5')


def load_keras(X, Y, learning_rate):
    start = time.time()
    lstm = build_lstm_model(X, learning_rate)
    lstm.load_weights('./keras.h5')

    loaded = time.time()
    print(f'keras load weights cost: {loaded - start}s')

    Y_hat = lstm.predict(X)
    print(f'keras predict one average cost: {(time.time() - loaded) / Y_hat.size}s')
    print(f'keras r2 score: {r2_score(Y_hat, Y)}')
    print('keras percentage error: {:.4f}%'.format(((Y_hat - Y) / Y).mean() * 100))


class LSTM(gluon.nn.Block):
    """
        input:     (1077, 1, 51)
        LSTM:      (51, 64) -> (1077, 1, 64)
        drop:              -> (1077, 1, 64)
        decoder:           -> (1077, 1)
    """

    def __init__(self, num_hiddens, input_size):
        super(LSTM, self).__init__()
        self.encoder = gluon.rnn.LSTM(hidden_size=num_hiddens, input_size=input_size)
        self.middle = gluon.nn.Dense(1)
        self.drop = gluon.nn.Dropout(0.01)
        self.decoder = gluon.nn.Dense(1)

    def forward(self, inputs):
        outputs = self.encoder(inputs)
        outs = self.decoder(self.drop(self.middle(outputs)))
        return outs


def train(features, labels, num_epochs, net, trainer, loss, validation_x=None, validation_y=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()

        with mx.autograd.record():
            y_hat = net(features)
            l = loss(y_hat, labels)

        l.backward()
        trainer.step(labels.shape[0])

        train_l_sum += l.sum().asscalar()
        n += l.size

        if validation_x is not None:
            validation_loss = loss(net(validation_x), validation_y).mean().asscalar()
            msg = 'epoch %d, train_loss %.4f, validation_loss %.4f, time %.4fs' % (
            epoch + 1, train_l_sum / n, validation_loss, time.time() - start)
        else:
            msg = 'epoch %d, train_loss %.4f, time %.4fs' % (epoch + 1, train_l_sum / n, time.time() - start)
        #print(msg)


def run_mxnet(train_x, train_y, num_epochs, learning_rate, validation_x=None, validation_y=None):
    loss = gluon.loss.L2Loss()
    net = LSTM(64, train_x.shape[2])
    net.initialize(mx.init.Xavier())
    net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})
    train(nd.array(train_x), nd.array(train_y), num_epochs, net, trainer, loss, nd.array(validation_x),
          nd.array(validation_y))
    net.save_parameters('./mxnet.net')


def load_mxnet(X, Y):
    start = time.time()
    net = LSTM(64, X.shape[2])
    net.load_parameters('./mxnet.net')
    net.hybridize()

    loaded = time.time()
    print(f'mxnet load weights cost: {loaded - start}s')

    Y_hat = net(nd.array(X)).asnumpy()
    print(f'mxnet predict one average cost: {(time.time() - loaded) / Y_hat.size}s')
    print(f'mxnet r2 score: {r2_score(Y_hat, Y)}')
    print('mxnet percentage error: {:.4f}%'.format(((Y_hat - Y) / Y).mean() * 100))


def main():
    df = downsample_filter_normalization('datas.csv')
    data, Y = build_data(df)
    print(df.shape, data.shape, Y.shape)  # (1441, 10) (1436, 51) (1436, 1)
    data = data.astype('float32')
    # Y = Y.astype('float32')

    validation_factor = int(0.75 * data.shape[0])
    train_x = data[:validation_factor, :]
    train_y = Y[:validation_factor]
    validation_x = data[validation_factor:, :]
    validation_y = Y[validation_factor:]

    batch_size = 1
    m, n = train_x.shape
    train_x = train_x.reshape((m, batch_size, n))
    m, n = validation_x.shape
    validation_x = validation_x.reshape((m, batch_size, n))

    print(train_x.shape, train_y.shape, validation_x.shape,
          validation_y.shape)  # (1077, 1, 51) (1077, 1) (359, 1, 51) (359, 1)

    s = time.time()
    num_epochs = 100

    # keras load weights cost: 0.18310260772705078s
    # keras predict one average cost: 0.000997248466300433s
    # keras r2 score: 0.6448245046829568
    # keras percentage error: 2.8355%
    # whole time: 161.03141260147095
    learning_rate = 1e-4
#    train_keras(train_x, train_y, batch_size, num_epochs, learning_rate, validation_x, validation_y)
#    load_keras(validation_x, validation_y, learning_rate)
    print(f'keras whole time: {time.time() - s}')

    # mxnet load weights cost: 0.004607439041137695s
    # mxnet predict one average cost: 8.711907856975757e-06
    # mxnet r2 score: 0.32217135154990173
    # mxnet percentage error: 3.436%
    # whole time: 1.1297211647033691
    s = time.time()
    learning_rate = 1e-3
    run_mxnet(train_x, train_y, num_epochs, learning_rate, validation_x, validation_y)
    load_mxnet(validation_x, validation_y)

    print(f'mxnet whole time: {time.time() - s}')


if __name__ == '__main__':
    main()

