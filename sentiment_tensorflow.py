# 深度学习库对比LSTM的训练预测效率
# tensorflow vs mxnet

import time
import tensorflow as tf
import numpy as np

from tensorflow import keras
from collections import Counter
from keras.datasets import imdb

word_index = imdb.get_word_index()  # length 88584

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2  # unknown
word_index['<UNUSED>'] = 3

index_to_word = {i: key for key, i in word_index.items()}
VOCAB_SIZE = 20000  # 46152


def load_data():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)
    print('Training entries: {}, Test entries: {}'.format(len(train_data), len(test_data)))
    return (train_data, train_labels), (test_data, test_labels)


def preprocess(data):
    return keras.preprocessing.sequence.pad_sequences(data,
                                                      value=word_index['<PAD>'],
                                                      padding='post',
                                                      maxlen=256)


def build_lstm_model(embed_size=64, num_hiddens=128):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(VOCAB_SIZE, embed_size))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(num_hiddens, return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(num_hiddens, return_sequences=True)))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['binary_accuracy'])
    return model


def sentiment():
    batch_size, embed_size, num_hiddens = 256, 64, 128
    (train_data, train_labels), (test_data, test_labels) = load_data()
    train_data = preprocess(train_data)
    test_data = preprocess(test_data)
    model = build_lstm_model(embed_size, num_hiddens)
    model.summary()

    history = model.fit(train_data,
                        train_labels,
                        epochs=10,
                        batch_size=batch_size)
    #                        validation_data=(test_data, test_labels))

    model.save('train_tf.h5')
    model.save_weights('train_weights_tf.h5')


def predict_all():
    (train_data, train_labels), (test_data, test_labels) = load_data()
    test_data = preprocess(test_data)
    model = keras.models.load_model('train_tf.h5')
    start = time.time()
    results = model.predict(test_data)
    print(f'cost {time.time() - start}s, ', results.shape)


def predict_one():
    sentence = 'When I watched this movie I went to sleep'
    review = list(map(lambda x: word_index[x], sentence.lower().split()))
    review = preprocess([review])

    start = time.time()
    lstm = build_lstm_model()
    lstm.load_weights('train_weights_tf.h5')
    loaded = time.time()
    print(f'load weights cost: {loaded - start}')
    results = lstm.predict(review)
    for i in range(99):  lstm.predict(review)
    print(f'predict one cost: {time.time() - loaded}s, ', results.shape)


def check_saved_weights():
    (train_data, train_labels), (test_data, test_labels) = load_data()
    test_data = preprocess(test_data)
    start = time.time()
    model = keras.models.load_model('train_tf.h5')
    print('load all model cost: ', time.time() - start)
    print('loss: {}, accuracy: {}'.format(
        *model.evaluate(test_data, test_labels)))  # [0.709635853767395, 0.8475698232650757]

    start = time.time()
    lstm = build_lstm_model()
    lstm.load_weights('train_weights_tf.h5')
    print('load weights model cost: ', time.time() - start)
    print('loss: {}, accuracy: {}'.format(
        *lstm.evaluate(test_data, test_labels)))  # [0.709635853767395, 0.8475698232650757]


if __name__ == '__main__':
    sentiment()
    #predict_one()
    # CPU 644% MEM 24%, 300s
