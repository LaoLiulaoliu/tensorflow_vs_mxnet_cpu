import pandas as pd
import numpy as np
import functools
import time
import pickle
from datetime import datetime, timedelta

import tensorflow as tf # 2.5.0
from tensorflow import keras

import torch # 1.8.1

import mxnet as mx # 1.8.0

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


INPUT_SIZE_STR = 'input_size_str'
group_time = 60000
steps = 5
irrelevant = 2


def read_csv(fname):
    df = pd.read_csv(fname)
    return df[(df['primary_air_rate'] > 0) & (df['primary_air_rate'] < 0.9)]


def aggregation_resample(df):
    """bias based on resample feature
    """
    base = df.index[-1]
    bias = 0 if base.microsecond == 0 else 1
    return df.resample(
        '{}S'.format(int(group_time / 1000)),
        closed='right',
        label='right',
        base=(base.second + bias) % int(group_time / 1000)
    ).mean()


def aggregate_time_by_min(df, add_index=True):
    if add_index is True:
        df.index = pd.to_datetime(df['time'], unit='ms') 
        df.drop(columns=['time'], inplace=True)
        return aggregation_resample(df)
    else:
        df['time'] = df['time'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M', time.localtime(0.001 * x)))
        return df.groupby('time').agg(lambda x: np.mean(x)).reset_index()


def min_max_normalization(df, df_min=None, df_max=None):
    if df_min is None:
        df_min = df.min(axis=0)
    if df_max is None:
        df_max = df.max(axis=0)

    df_norm = (df - df_min) / (df_max - df_min)

    if 'time' in df.columns:
        df_norm['time'] = df['time']  # pd.concat([df['time'], df_norm], axis=1)

    return df_norm, df_min, df_max


def reconstruct_train_data(df):
    """inputs按分钟聚合，临近5min数据拼一行，每行数据时间上要求是顺序相连的。如果当前时间是t，则这一行包含[t-5, t-1] 的所有数据.
       取出y，删除t-1, t-2 的y，和时间列。如果这一行有空值数据则删除此行。
    """
    if len(df) < steps + 1:
        return None, None

    if 'time' in df.columns:
        convert_strtime_datetime = functools.partial(datetime.strptime)
        time_column = list(map(lambda x: convert_strtime_datetime(x, '%Y-%m-%d %H:%M'), df['time']))

    labels = []
    data = None

    for t in range(len(df) - 1, steps - 1, -1):
        time_difference_seconds = int(group_time / 1000)
        item = np.empty(0)
        for j in range(1, steps + 1):
            if (df.index[t] - df.index[t - j]).seconds != time_difference_seconds * j:
                break
        else:
            for i in range(1, steps + 1):
                item = np.hstack((item, df.iloc[t - i, :-1].values)) \
                    if i <= irrelevant else np.hstack((item, df.iloc[t - i, :].values))

            data = np.vstack((data, item)) if data is not None else item
            labels.append(df.iloc[t, -1])
    return data, np.asarray(labels).reshape((-1, 1))


def build_lstm_model(X, learning_rate):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(64, input_shape=(X.shape[1], X.shape[2])))
    # keras.layers.LSTM(64, batch_input_shape=(1, 1, X.shape[2]), stateful=True) # keep the cell status
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


class LSTMTORCH():
    def __init__(self, num_hiddens, input_size):
        super(LSTMTORCH, self).__init__()
        torch.nn.LSTM(input_size=input_size, hidden_size=num_hiddens)



class LSTMNet(mx.gluon.nn.Block):
    """
        input:   (1077, 1, 51)
        LSTM:    (51, 64) -> (1077, 1, 64)
        dense:            -> (1077, 1)
        decoder:          -> (1077, 1)
    """
    def __init__(self, num_hiddens, input_size):
        super(LSTMNet, self).__init__()
        self.encoder = mx.gluon.rnn.LSTM(hidden_size=num_hiddens, input_size=input_size)
        self.middle = mx.gluon.nn.Dense(1)
        self.drop = mx.gluon.nn.Dropout(0.01)
        self.decoder = mx.gluon.nn.Dense(1)

    def forward(self, inputs):
        outputs = self.encoder(inputs)
        outs = self.decoder(self.drop(self.middle(outputs)))
        return outs


class LSTMNetBetter(mx.gluon.nn.Block):
    """
        input:     (1436, 1, 43)
        LSTM1:     (43, 32) -> (1436, 1, 32)
        LSTM2:     (32, 64) -> (1436, 1, 64)
        decoder:            -> (1436, 1)
    """
    def __init__(self, num_hiddens, input_size):
        """
        :param num_hiddens: 64 is optimized in some algorighm
        :param input_size:
        """
        super(LSTMNetBetter, self).__init__()
        half_hiddens = int(num_hiddens * 0.5)
        self.encoder = mx.gluon.rnn.LSTM(hidden_size=half_hiddens, input_size=input_size)
        self.drop = mx.gluon.nn.Dropout(0.01)
        self.encoder2 = mx.gluon.rnn.LSTM(hidden_size=num_hiddens, input_size=half_hiddens)
        self.drop2 = mx.gluon.nn.Dropout(0.01)
        self.decoder = mx.gluon.nn.Dense(1)

    def forward(self, inputs):
        outputs1 = self.drop(self.encoder(inputs))
        outputs = self.drop2(self.encoder2(outputs1))
        outs = self.decoder(outputs)
        return outs


def training_procedure(features, labels, net, num_epochs, trainer, loss, validation_x=None, validation_y=None):
    start = time.time()
    for epoch in range(num_epochs):
        train_l_sum, n = 0.0, 0

        with mx.autograd.record():
            y_hat = net(features)
            l = loss(y_hat, labels)

        l.backward()
        trainer.step(labels.shape[0])

        train_l_sum += l.sum().asscalar()
        n += l.size

        if validation_x is not None:
            validation_loss = loss(net(validation_x), validation_y).mean().asscalar()
            msg = 'epoch %d, train_loss %.4f, validation_loss %.4f, %.4fs/epoch' % (
            epoch + 1, train_l_sum / n, validation_loss, (time.time() - start) / (epoch + 1))
        else:
            msg = 'epoch %d, train_loss %.4f, %.4fs/epoch' % (epoch + 1, train_l_sum / n, (time.time() - start) / (epoch + 1))
        if epoch & 63 == 0:
            print(msg)


def save_mxnet_model(net, fname, input_size):
    params = net._collect_params_with_prefix()
    model = {key: val._reduce() for key, val in params.items()}
    model[INPUT_SIZE_STR] = input_size
    with open(fname, 'wb') as fd:
        pickle.dump(model, fd)


def load_mxnet_model(fname):
    with open(fname, 'rb') as fd:
        model = pickle.load(fd)
    input_size = model.pop(INPUT_SIZE_STR)

    net = LSTMNet(64, input_size)
    params = net._collect_params_with_prefix()
    for name in model:
        if name in params:
            params[name]._load_init(model[name], mx.cpu(), cast_dtype=False, dtype_source='current')

    return net


def run_mxnet(train_x, train_y, num_epochs, learning_rate, validation_x=None, validation_y=None):
    input_size = train_x.shape[2]
    net = LSTMNet(64, input_size)
    net.initialize(mx.init.Xavier())
    net.hybridize()

    trainer = mx.gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})
    training_procedure(mx.nd.array(train_x), mx.nd.array(train_y), net,
        num_epochs, trainer,
        mx.gluon.loss.L2Loss(),
        mx.nd.array(validation_x),
        mx.nd.array(validation_y))
    save_mxnet_model(net, './mxnet.net', input_size) # net.save_parameters('./mxnet.net')


def load_mxnet(X, Y):
    start = time.time()
    net = load_mxnet_model('./mxnet.net')
    # net = LSTMNetBetter(64, X.shape[2])
    # net.load_parameters('./mxnet.net')
    net.hybridize()

    loaded = time.time()
    print(f'mxnet load weights cost: {loaded - start}s')

    Y_hat = net(mx.nd.array(X)).asnumpy()
    print(f'mxnet predict one average cost: {(time.time() - loaded) / Y_hat.size}s')
    print(f'mxnet r2 score: {r2_score(Y_hat, Y)}')
    print('mxnet percentage error: {:.4f}%'.format(((Y_hat - Y) / Y).mean() * 100))


def linear_model_way(train_x, train_y):
    lr = LinearRegression()
    scores = cross_val_score(lr, train_x, train_y, cv=3, scoring='r2')
    print(scores) # [0.88790513 0.78219812 0.75124809]


def main():
    df = read_csv('datas.csv')
    df = aggregate_time_by_min(df, True)
    df_norm, df_min, df_max = min_max_normalization(df)
    datas, labels = reconstruct_train_data(df_norm)

    print(df_norm.shape, datas.shape, labels.shape)  # (1441, 9) (1436, 43) (1436, 1)
    datas = datas.astype('float32')

    validation_factor = int(0.8 * datas.shape[0])
    train_x = datas[:validation_factor, :]
    train_y = labels[:validation_factor]
    validation_x = datas[validation_factor:, :]
    validation_y = labels[validation_factor:]

    linear_model_way(train_x, train_y)

    batch_size = 1
    m, n = train_x.shape
    train_x = train_x.reshape((m, batch_size, n))
    m, n = validation_x.shape
    validation_x = validation_x.reshape((m, batch_size, n))

    print(train_x.shape, train_y.shape, validation_x.shape,
          validation_y.shape)  # (1077, 1, 43) (1077, 1) (359, 1, 43) (359, 1)

    s = time.time()

    # keras load weights cost: 0.18310260772705078s
    # keras predict one average cost: 0.000997248466300433s
    # keras r2 score: 0.6448245046829568
    # keras percentage error: 2.8355%
    # whole time: 161.03141260147095
    learning_rate = 1e-4
    num_epochs = 10
    train_keras(train_x, train_y, batch_size, num_epochs, learning_rate, validation_x, validation_y)
    load_keras(validation_x, validation_y, learning_rate)
    print(f'keras whole time: {time.time() - s}')

    # mxnet load weights cost: 0.004607439041137695s
    # mxnet predict one average cost: 8.711907856975757e-06
    # mxnet r2 score: 0.32217135154990173
    # mxnet percentage error: 3.436%
    # whole time: 1.1297211647033691
    s = time.time()
    learning_rate = 0.002
    num_epochs = 500
    run_mxnet(train_x, train_y, num_epochs, learning_rate, validation_x, validation_y)
    load_mxnet(validation_x, validation_y)

    print(f'mxnet whole time: {time.time() - s}')


if __name__ == '__main__':
    main()

