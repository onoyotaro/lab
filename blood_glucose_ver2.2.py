"""
＊注意事項
    追記・編集するときは，変更した箇所に変更日時と変更事項をコメントアウトすること！
    変更する前にオリジナル（または前のver）のバックアップをとること！
    →オリジナルを変更したら殴る
"""

"""
2020.05.27
昨年度のプログラムを見やすく書き直したい
これをもとにDELLのほうのプログラムを直したい
ていうか、DeLL動け
add:: 2020.06.02 動きました
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

"""
2. 構造の定義
"""
#class MLP_01 →　class MLP_none_dropout
class MLP_none_dropout(chainer.Chain):
    def __init__(self, nunits, n_out):
        super(MLP_none_dropout, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

# class MLP_02 →　class MLP_on_dropout
class MLP_on_dropout(chainer.Chain):
    """
       モデルの実装
       """

    def __init__(self, n_units, n_out, train=True, drop_out_ratio=0.3):
        super(MLP_on_dropout, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

        # 学習の場合：True
        self.__train = train
        # drop outの実施有無
        self.__drop_out = True
        # drop outの比率
        self.drop_out_ratio = drop_out_ratio

    def __call__(self, x):
        drop_out = self.__train and self.__drop_out
        h1 = F.dropout(F.relu(self.l1(x)), ratio=self.drop_out_ratio)
        h2 = F.dropout(F.relu(self.l2(h1)), ratio=self.drop_out_ratio)
        return self.l3(h2)

    # 学習の場合；True
    def __get_train(self):
        return self.__train

    def __set_train(self, train):
        self.__train = train

    train = property(__get_train, __set_train)

    # Dropoutを使用する場合：True
    def __get_drop_out(self):
        return self.__drop_out

        def __set_drop_out(self, drop_out):
            '''
            drop outフラグの設定
            '''
            self.__drop_out = drop_out

        drop_out = property(__get_drop_out, __set_drop_out)

    def parse_device(args):
        gpu = None
        if args.gpu is not None:
            gpu = args.gpu
        elif re.match(r'(-|\+|)[0-9]+$', args.device):
            gpu = int(args.device)

        if gpu is not None:
            if gpu < 0:
                return chainer.get_device(numpy)
            else:
                import cupy
                return chainer.get_device((cupy, gpu))

        return chainer.get_device(args.device)

"""
3. データセットのインポート
"""
# 1,000datas
data, teach = easy_chainer.load_Data("ファイルの場所")　# add:: 2020.06.03 ファイルの場所指定を大野のPCの場所から変えました．
data = data.astype(numpy.float32)
teach = teach
all_data_number = len(teach)
print(teach)
print(teach.shape)

teach = teach.astype(numpy.float32)
# print(data.shape)

# 教師値にidを割り振る
id_all = numpy.arange(1, len(teach) + 1, 1).astype(numpy.int32) - 1
print(id_all)

#　seedは適当
# 訓練に使うデータ数はその時の最適なもので
numpy.random.seed(11)
id_train = numpy.random.choice(id_all, 300, replace=False)
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
    kunren_data_number = 300
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
net = MLP_on_dropout(1000,1)
model = L.Classifier(net,
                     lossfun=F.mean_squared_error,
                     accfun = F.r2_score)
model.compute_accuracy = False

# optimizerの定義
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

"""
7. 訓練
    訓練準備
    訓練開始
"""

# train_epoch:訓練回数
train_epoch = 5

updater = training.updaters.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (train_epoch, 'epoch'), out="Result2018_oono/%s" % time.strftime("%Y%m%d%H%M%S"))

# オプションあれこれ
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

# 訓練開始
trainer.run()

"""
8. 検証
    訓練データの検証
    テストデータの検証
"""
train_iter.reset()
train_batch = train_iter.next()
train_spc, train_ref = concat_examples(train_batch)
for i in range(10):
    train_cal_ref = train_ref[i]
    train_cal_pred = net(train_spc[i].reshape(1, -1)).data
    print("Train_Ref : %d, Train_Pred : %d" % (train_cal_ref, train_cal_pred))


test_iter.reset()
test_batch = test_iter.next()
test_spc, test_ref = concat_examples(test_batch)
for i in range(10):
    test_val_ref = test_ref[i]
    test_val_pred = net(test_spc[i].reshape(1, -1)).data
    print("Test_Ref : %d, Test_Pred : %d" % (test_val_ref, test_val_pred))

"""
＊注意事項
    追記・編集するときは，変更した箇所に変更日時と変更事項をコメントアウトすること！
    変更する前にオリジナル（または前のver）のバックアップをとること！
    →オリジナルを変更したら殴る
"""
