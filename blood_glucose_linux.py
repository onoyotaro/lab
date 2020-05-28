"""
・Linuxで動かす用のやーつ
    オリジナルは編集しないこと
    追記する場合は追記した日時,変更事項を記すこと
    編集：2020.05.28
    変更事項：全部
    
*注意事項
    追記・編集するときは，変更した箇所に変更日時と変更事項をコメントアウトすること！
    変更する前にオリジナル（または前のバージョン）のバックアップをとること
    →オリジナルを変更したら殴る
"""

"""
1.モジュールのインポート
"""

%load_ext autoreload

%autoreload 2
import time

import pandas
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import Link, Chain, ChainList

from chainer.datasets import tuple_dataset
from chainer.dataset import convert, concat_examples

from chainer import datasets, iterators, optimizers, serializers
from chainer import Function, report, training, utils, Variable

import numpy
import os
import glob
import sys
from pathlib import Path
import matplotlib.pyplot as plt

from chainer import Sequential
import easy_chainer

"""
2. 構造の定義
 2.1  MLP_none_dropout : MLP, dropout=None, Batch_Normalization=None
 2.2  MLP_on_dropout : MLP, Dropout, Batch_Normalization
"""

class MLP_none_dropout(chainer.Chain):

    def __init__(self, n_units, n_out):
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
   
# 4層NN(入力→中間，中間→出力)
class MLP_on_dropout(chainer.Chain):
    
    """
    モデルの実装
    """

    def __init__(self, n_units, n_out, train=True, drop_out_ratio=0.1):
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

    

# GPUの設定
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
3. データのインポート
  427data : blood_glucose_fulldata.xlsx
  
"""

data, teach = easy_chainer.load_Data("/home/fiber_classifier/Desktop/blood_glucose_fulldata.xlsx")
data = data.astype(numpy.float32)
teach = teach
all_data_number = len(teach)
#print(teach)
print(teach.shape)

"""
4. 訓練と検証に分けるよ
"""

# 回帰させるときに必要（分類はint型）
teach = teach.astype(numpy.float32)

id_all = numpy.arange(1, len(teach) + 1, 1).astype(numpy.int32) - 1
print(id_all.shape)

idtrain_data_number = 300


# seedは適当に決めた
numpy.random.seed(13)
id_train = numpy.random.choice(id_all, train_data_number, replace=False) #重複なし
#print(id_train.dtype)

id_test = numpy.delete(id_all, id_train)
#print(id_test)
  

# print("id_train : ", id_train)
print("id_train : ", id_train.shape)
print("id_train_type : ", id_train.dtype)

# print("id_test : ", id_test)
print("id_test : ", id_test.shape)
print("id_test : ", id_test.dtype)
  

# 訓練データの選び方
# train_data_choice = 0 : ランダムに訓練データを選ぶ
# train_data_choice = 1 : 前から何番目を訓練データにする

train_data_choice = 0

if (train_data_choice == 0):
    # ランダムに訓練idをふったとき
    x_train, y_train = data[:, id_train], teach[id_train]
    x_test, y_test = data[:, id_test], teach[id_test]
    
elif (train_data_choice == 1):
    # ランダムじゃないとき
    x_train, y_train = data[:, 0:idtrain_data_number], teach[0:idtrain_data_number]
    x_test, y_test = data[:, idtrain_data_number:all_data_number], teach[idtrain_data_number:all_data_number]
    
"""
n = 35
m = 373 - n

x_train, y_train = data[:, 0:m], teach[0:m]
x_test, y_test = data[:, m:373], teach[m:373]
"""
"""
5. イテレータの生成するよ
"""
train_data_iterators = 10
test_data_iterators = 10

train = tuple_dataset.TupleDataset(x_train.T, y_train.reshape(-1,1 ))
test = tuple_dataset.TupleDataset(x_test.T, y_test.reshape(-1, 1))

train_iter = chainer.iterators.SerialIterator(train, train_data_iterators, repeat=True,  shuffle=False)
test_iter = chainer.iterators.SerialIterator(test, test_data_iterators, repeat=False, shuffle=False)

"""
6. モデル定義
    どのモデルを使うの？
"""

net = MLP_on_dropout(1000,1)
model = L.Classifier(net,
                     lossfun=F.mean_squared_error,
                     accfun = F.r2_score)
model.compute_accuracy = False

# GPUを使うかどうか？gpu_id:0で使う,gpu_id:-1
gpu_id = 0
model.to_gpu()

# 最適化方法
#optimizer = chainer.optimizers.SGD()
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

"""
7. 訓練開始
    訓練準備
    訓練開始
"""

# 学習回数（epoch）とかを決める
train_epoch = 3000
updater = training.updaters.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (train_epoch, 'epoch'), out="Result2019_oono/%s" % time.strftime("%Y%m%d%H%M%S"))
out_directory = "./Result2019_oono/%s" % time.strftime("%Y%m%d_%H%M%S")

# オプションです

# Evaluate the model with the test dataset for each epoch
trainer.extend(extensions.Evaluator(test_iter, model, device=0))

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

# モデル保存
net.to_cpu()

chainer.serializers.save_hdf5("/home/fiber_classifier/Desktop/model.hdf" ,obj=net)

numpy.save(arr=y_test, file="/home/fiber_classifier/Desktop/test_id.npy"  )
numpy.save(arr=y_train, file="/home/fiber_classifier/Desktop/train_id.npy" )

# 検証（訓練データ）
train_iter.reset()

train_batch = train_iter.next()
train_spc, train_ref = concat_examples(train_batch)  # Train Dataset
for i in range(10):
    train_cal_ref = train_ref[i]
    train_cal_pred = net(train_spc[i].reshape(1, -1)).data
    print("Train_Ref: %d, Train_Pred: %d " % (train_cal_ref, train_cal_pred))
    
    
# 検証（テストデータ）
test_iter.reset()

test_batch = test_iter.next()
test_spc, test_ref = concat_examples(test_batch)  # Test Dataset
for i in range(10):
    test_val_ref = train_ref[i]
    test_val_pred = net(train_spc[i].reshape(1, -1)).data
    print("Test_Ref : %d, Test_Pred : %d " % (test_val_ref, test_val_pred))
    
 """
 *注意事項
    追記・編集するときは，変更した箇所に変更日時と変更事項をコメントアウトすること！
    変更する前にオリジナル（または前のバージョン）のバックアップをとること
    →オリジナルを変更したら殴る
 """
