"""
2020.05.27~
昨年度のプログラムを見やすく書き直したい
これをもとにDELLのほうのプログラムを直したい
ていうか、DeLL動け
"""


"""
1.モジュールのインポート
"""

# %load_ext autoreload

# %autoreload 2
import time

import pandas
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


from chainer.datasets import tuple_dataset
from chainer.dataset import convert, concat_examples

import numpy
import os
import glob
import sys
from pathlib import Path

from chainer import Sequential
import easy_chainer

import matplotlib.pyplot as plt

class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)    # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

easy_chainer_choice = 0
if (easy_chainer_choice == 0):
    
    raw_data = pandas.read_excel("C:/Users/Owner/Desktop/Normalized/val/new2_1.xlsx")
    data, teach = raw_data.as_matrix()[:-1], raw_data.as_matrix()[-1]
    print("data : ", data.shape)
    print("teach : ", teach.shape)
    all_data_number = len(teach)

elif (easy_chainer_choice == 1):
    # 1,000datas
    data, teach = easy_chainer.load_Data("C:/Users/Owner/Desktop/Normalized/val/new2_1.xlsx")
    data = data.astype(numpy.float32)
    teach = teach
    all_data_number = len(teach)
    print("data : ", data.shape)
    print("teach : ", teach.shape)
    print(teach.shape)
    
else:
    print("easy_chainer_choiceを見直せ")

data = data.astype(numpy.float32)
teach = teach.astype(numpy.int8)

# 教師値にidを割り振る
id_all = numpy.arange(1, len(teach) + 1, 1).astype(numpy.int32) - 1
print(id_all)

#　seedは適当
# 訓練に使うデータ数はその時の最適なもので
numpy.random.seed(11)
id_train = numpy.random.choice(id_all, 80, replace=False)
# print(id_train)
# print(len(id_train))

id_test = numpy.delete(id_all, id_train)
# print(id_test)
# print(len(id_test))

# data_choice_random:0 →　ランダムにデータ選ばない
# data_choice_random:1 →　ランダムに選ぶ
# pythonのやり方と逆やから要検討
data_choice_random = 0

if (data_choice_random == 0):
    # 訓練データをランダムに選ばないとき
    kunren_data_number = 80
    x_train, y_train = data[:, 0:kunren_data_number], teach[0:kunren_data_number]
    x_test, y_test = data[:, kunren_data_number:all_data_number], teach[kunren_data_number:all_data_number]

else:
    # 訓練データをランダムに選ぶとき
    x_train, y_train = data[:, id_train], teach[id_train]
    x_test, y_test = data[:, id_test], teach[id_test]

"""
5. イテレータの生成
"""
train_data_iterators = 10
test_data_iterators = 10
train =  tuple_dataset.TupleDataset(x_train.T, y_train.reshape(-1,1 ))
test = tuple_dataset.TupleDataset(x_test.T, y_test.reshape(-1, 1))

train_iter = chainer.iterators.SerialIterator(train, train_data_iterators, repeat=True, shuffle=False)
test_iter = chainer.iterators.SerialIterator(test, test_data_iterators, repeat=False, shuffle=False)

"""
6. モデル定義
    どのモデルをつかうか？
"""
net = MLP(1000, 12)

model = L.Classifier(predictor = net,
                     lossfun = F.softmax_cross_entropy,
                    accfun = F.accuracy)

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
updater = training.updaters.StandardUpdater(
        train_iter, optimizer)
trainer = training.Trainer(updater, (100, 'epoch'), out="Result2020_oono/%s" % time.strftime("%Y%m%d%H%M%S"))

# Evaluate the model with the test dataset for each epoch
trainer.extend(extensions.Evaluator(test_iter, model))

# Dump a computational graph from 'loss' variable at the first iteration
# The "main" refers to the target link of the "main" optimizer.
trainer.extend(extensions.dump_graph('main/loss'))

# Take a snapshot for each specified epoch
# frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
# trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

# Write a log of evaluation statistics for each epoch
trainer.extend(extensions.LogReport())

# Save two plot images to the result dir
trainer.extend(
    extensions.PlotReport(['main/loss', 'validation/main/loss'],
                          'epoch', file_name='loss.png'))
trainer.extend(
    extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'],
        'epoch', file_name='accuracy.png'))

# Print selected entries of the log to stdout
# Here "main" refers to the target link of the "main" optimizer again, and
# "validation" refers to the default name of the Evaluator extension.
# Entries other than 'epoch' are reported by the Classifier link, called by
# either the updater or the evaluator.
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

# Print a progress bar to stdout
trainer.extend(extensions.ProgressBar())

# if args.resume:
#     # Resume from a snapshot
#     chainer.serializers.load_npz(args.resume, trainer)

trainer.run()


test_iter.reset()

test_batch = test_iter.next()
test_spc, test_ref = concat_examples(test_batch)  # Test Dataset
for i in range(36):
    ref = test_ref[i]
    pred = numpy.argmax(net(test_spc[i].reshape(1, -1)).data)
    print("Ref: %d, Pred: %d %s" % (ref, pred, ref == pred))
