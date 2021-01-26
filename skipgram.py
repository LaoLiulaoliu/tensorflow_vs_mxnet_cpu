import collections
import math
import random
import sys
import time
import inspect
import re
import pickle
import mxnet as mx
from mxnet import gluon, nd

with open('./ptb.train.txt', 'r') as f:
    lines = f.readlines()
    raw_dataset = [sentence.split() for sentence in lines]


counter = collections.Counter([token for sentence in raw_dataset for token in sentence])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
idx_to_token = [token for token, _ in counter.items()]
token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}


dataset = [[token_to_idx[token] for token in sentence if token in token_to_idx] for sentence in raw_dataset]
num_tokens = sum([len(sentence) for sentence in dataset])


def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(1e-4 * num_tokens / counter[idx_to_token[idx]])

subsampled_dataset = [[token for token in sentence if not discard(token)] for sentence in dataset]

def compare_counts(token='the'):
    return '# %s: before=%d, after=%d' % (
      token,
      sum([sentence.count(token_to_idx[token]) for sentence in dataset]),
      sum([sentence.count(token_to_idx[token]) for sentence in subsampled_dataset]))


def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)  # 背景词窗口大小随机
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts


def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates = [], []
    population = list(range(len(sampling_weights)))

    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            try:
                neg = neg_candidates.pop()
                if neg not in set(contexts):  # 噪声词不能是背景词
                    negatives.append(neg)
            except IndexError:
                # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                neg_candidates = random.choices(population, sampling_weights, k=int(1e5))

        all_negatives.append(negatives)
    return all_negatives


def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)  # 60
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (nd.array(centers).reshape((-1, 1)),  # (512, 1)
            nd.array(contexts_negatives),  # (512, 60)
            nd.array(masks),  # (512, 60)
            nd.array(labels))  # (512, 60)


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = nd.batch_dot(v, u.swapaxes(1, 2))
    return pred


def train(net, loss, lr, num_epochs):
    net.initialize(force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})

    for epoch in range(num_epochs):
        start, loss_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = batch
            with mx.autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])  # (512, 1, 60)
                # 使用掩码变量mask来避免填充项对损失函数计算的影响
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(label.shape[0])

            loss_sum += l.sum().asscalar()
            n += l.size
        print('epoch %d, loss %.4f, time %.2fs'
              % (epoch + 1, loss_sum / n, time.time() - start))



def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        print(line)
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def save_mxnet_model(net, fname, embed_size):
    params = net._collect_params_with_prefix()
    model = {key: val._reduce() for key, val in params.items()}
    model[varname(embed_size)] = embed_size
    with open(fname, 'wb') as fd:
        pickle.dump(model, fd)


def load_mxnet_model(fname, embed_size=0):
    with open(fname, 'rb') as fd:
        model = pickle.load(fd)
    embed_size = model.pop(varname(embed_size))

    net = gluon.nn.Sequential()
    net.add(gluon.nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size, sparse_grad=True),
            gluon.nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size, sparse_grad=True))

    params = net._collect_params_with_prefix()
    for name in model:
        if name in params:
            params[name]._load_init(model[name], mx.cpu(), cast_dtype=False, dtype_source='current')

    return net



def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[token_to_idx[query_token]]
    # 添加的1e-9是为了数值稳定性
    cos = nd.dot(W, x) / (nd.sum(W * W, axis=1) * nd.sum(x * x) + 1e-9).sqrt()
    topk = nd.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # 除去输入词
        print('cosine sim=%.3f: %s' % (cos[i].asscalar(), (idx_to_token[i])))


all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)
sampling_weights = [counter[word]**0.75 for word in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)

batch_size = 512
num_workers = 1
dataset = gluon.data.ArrayDataset(all_centers, all_contexts, all_negatives)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True,
                             batchify_fn=batchify, num_workers=num_workers)


embed_size = 100
net = gluon.nn.Sequential()
net.add(gluon.nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size, sparse_grad=True),
        gluon.nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size, sparse_grad=True))

loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

train(net, loss, 0.01, 50)
save_mxnet_model(net, 'skipgram_embed', embed_size)

get_similar_tokens('costume', 3, net[0])