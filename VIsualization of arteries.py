import numpy as np
import cv2
from matplotlib import pyplot as plt
from IPython.display import Image
import math
import datetime
import numpy
import pandas

import glob

# 動画ファイルを読み込んで，フレームに分割
import cv2

video = cv2.VideoCapture('C:/Users/Owner/Desktop/橈骨動脈可視化装置/data/30.avi')

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) # 動画の画面横幅
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 動画の画面縦幅
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # 総フレーム数
frame_rate = int(video.get(cv2.CAP_PROP_FPS)) # フレームレート(fps)
video_time = frame_count / frame_rate

print(width)
print(height)
print(frame_count)
print(frame_rate)
print(video_time)
print(video.isOpened())

# video_path = ('C:/Users/Owner/Desktop/橈骨動脈可視化装置/data/20200703103345.avi')
# cap = cv2.VideoCapture(video_path)

num = 0
while(video.isOpened()):
    ret, frame = video.read()
    if ret == True:
        cv2.imwrite("picture{:0=3}".format(num)+".jpg", frame)
        print("save picture{:0=3}".format(num)+ ".jpg")
        num += 1
    
    else:
        break
        
video.release()

# 8bit画像の画素値を取得する(256階調)

# gray_list = sorted(glob.glob("C:/Users/Owner/Desktop/sample/picture*"))
# print(gray_list)


for i in range(len(gray_list)):
    if (i == 0):
        image0 = cv2.imread(gray_list[i], cv2.IMREAD_GRAYSCALE)
        # print(i, "番目")
        image_area0 = image0[730:830, 660:760]
        array = image_area0.reshape(10000, 1)
    
    else:
        image = cv2.imread(gray_list[i], cv2.IMREAD_GRAYSCALE)
        image_area = image[730:830, 660:760]
        image_reshape = image_area.reshape(10000, 1)
        
        array = numpy.hstack((array, image_reshape))
        
    print(array.shape)

print(array.shape)

numpy.savetxt("array.csv", X=array, delimiter=',')

# 8bit画像を二値化して白部分と黒部分を数える
# 橈骨動脈の動画を読み込み
# フレームごとに画像を分割
# 300×300程度に画像をカット
# 画像を二値化
# 白黒部分を数える


gray_list = sorted(glob.glob("C:/Users/Owner/Desktop/arteries/pic/15/picture*"))
print(len(gray_list))


# 画像の範囲設定の確認
image0 = cv2.imread(gray_list[0], cv2.IMREAD_GRAYSCALE)
print(type(image0))
image_area0 = image0[500:600, 780:880]
# image_area = image0[490:790, 610:910]
cv2.imshow('img_th', image_area0)
cv2.waitKey()
cv2.destroyAllWindows()

# "q"で画像閉じる

gray_list = sorted(glob.glob("C:/Users/Owner/Desktop/arteries/pic/15/picture*"))
print(len(gray_list))

threshold = 100
num = 0

# 閾値の設定(画素値100以上を黒にする)
threshold = 100
black_list = []
white_list = []

for i in range(len(gray_list)):
    if (i == 0):
        image0 = cv2.imread(gray_list[i], cv2.IMREAD_GRAYSCALE)
        # print(i, "番目")
        image_area0 = image0[500:600, 780:880]
        ret, img_thresh = cv2.threshold(image_area0, threshold, 255, cv2.THRESH_BINARY)
        cv2.imwrite("C:/Users/Owner/Desktop/arteries/area_pic/15/picture.jpg", image_area0)
        
        u, counts = numpy.unique(img_thresh, return_counts=True)
        white_counts = counts[0] / 10000
        black_counts = counts[1] / 10000
        print(u)
        print("white[0] : ", counts[0])
        # print("black[255] : ", counts[1])
        black_list.append(black_counts)
        white_list.append(white_counts)
        print('0番目は ok')
    
    else:
        image = cv2.imread(gray_list[i], cv2.IMREAD_GRAYSCALE)
        image_area = image[500:600, 780:880]
        # image_reshape = image_area.reshape(10000, 1)
        
        ret, img_thresh = cv2.threshold(image_area, threshold, 255, cv2.THRESH_BINARY)
        
        cv2.imwrite("C:/Users/Owner/Desktop/arteries/area_pic/15/picture{:0=3}".format(num)+".jpg", image_area)
        
        u, counts = numpy.unique(img_thresh, return_counts=True)
        white_counts = counts[0] / 10000
        black_counts = counts[1] / 10000
        # print(u)
        # print("white : ", counts[0])
        # print("black[255] : ", counts[1])
        black_list.append(black_counts)
        white_list.append(white_counts)
        num += 1

        
print("black_list : ", len(black_list))
print("white_list : ", len(white_list))
print(black_list[0])
print(white_list[0])

# 二値化した白黒情報をExcelで出力
black_df = pandas.DataFrame(black_list)
white_df = pandas.DataFrame(white_list)

black_df.to_excel('C:/Users/Owner/Desktop/arteries/excel/15/black_15.xlsx')
white_df.to_excel('C:/Users/Owner/Desktop/arteries/excel/15/white_15.xlsx')

black_df.to_excel('C:/Users/Owner/Desktop/arteries/excel/black_15.xlsx')
white_df.to_excel('C:/Users/Owner/Desktop/arteries/excel/white_15.xlsx')


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
import math

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
data, teach = easy_chainer.load_Data("C:/Users/Owner/Desktop/arteries/excel/black.xlsx")
data = data.astype(numpy.float32)
teach = teach.astype(numpy.float32)
all_data_number = len(teach)
print(teach)
print(teach.shape)

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
    kunren_data_number = 20
    x_train, y_train = data[:, 0:kunren_data_number], teach[0:kunren_data_number]
    x_test, y_test = data[:, kunren_data_number:all_data_number], teach[kunren_data_number:all_data_number]
    id_train = teach[0:kunren_data_number]
    id_test = teach[kunren_data_number:all_data_number]

else:
    # 訓練データをランダムに選ぶとき
    x_train, y_train = data[:, id_train], teach[id_train]
    x_test, y_test = data[:, id_test], teach[id_test]
    

"""
5. イテレータの生成
"""
train_data_iterators = 10
test_data_iterators = 3
train =  tuple_dataset.TupleDataset(x_train.T, y_train.reshape(-1,1 ))
test = tuple_dataset.TupleDataset(x_test.T, y_test.reshape(-1, 1))

train_iter = chainer.iterators.SerialIterator(train, train_data_iterators, repeat=True, shuffle=False)
test_iter = chainer.iterators.SerialIterator(test, test_data_iterators, repeat=False, shuffle=False)


"""
6. モデル定義
    どのモデルをつかうか？
"""
net = MLP_on_dropout(600,1)
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
# 訓練データの精度検証
#　batch, jの関数名を考え直す
train_ref_list = []
train_pred_list = []
count_step = 0
j = 0

batch = len(id_train) // train_data_iterators
loop_kaisuu = batch + 1
print(batch)
amari = len(id_train) % train_data_iterators
print(amari)

for j in range(loop_kaisuu):
    if (count_step == 0):
        train_iter.reset()

        train_batch = train_iter.next()
        train_spc, train_ref = concat_examples(train_batch)  # Test Dataset
        for i in range(train_data_iterators):
            cal_ref = train_ref[i]
            cal_pred = net(train_spc[i].reshape(1, -1)).data
            # print(" %d" % (cal_ref))
            train_ref_list.append(math.floor(cal_ref))
            # print(" %d" % (cal_pred) )
            train_pred_list.append(math.floor(cal_pred))
            print("Ref : %d, Pred : %d" % (cal_ref, cal_pred))
            # 最初の10個だけ使って大きな間違いがないかを確認

       
        count_step = count_step + 1
        # print("count : ", count_step)
    

    elif (0< count_step < batch):
        
        train_batch = train_iter.next()
        train_spc, train_ref = concat_examples(train_batch)  # Test Dataset
        for i in range(train_data_iterators):
            cal_ref = train_ref[i]
            cal_pred = net(train_spc[i].reshape(1, -1)).data
            # print(" %d" % (cal_ref))
            train_ref_list.append(math.floor(cal_ref))
            # print(" %d" % (cal_pred) )
            train_pred_list.append(math.floor(cal_pred))
            # print("")
            
        
        count_step = count_step + 1    
        # print("count_step", count_step)

    elif (count_step == batch):
        print("last loop")
        train_batch = train_iter.next()
        train_spc, train_ref = concat_examples(train_batch)  # Test Dataset
        for i in range(amari):
            cal_ref = train_ref[i]
            cal_pred = net(train_spc[i].reshape(1, -1)).data
            print(" %d" % (cal_ref))
            train_ref_list.append(math.floor(cal_ref))
            print(" %d" % (cal_pred) )
            train_pred_list.append(math.floor(cal_pred))
        
        

print("")
print(len(train_ref_list))
print(len(train_pred_list))

train_ref_df = pandas.DataFrame(train_ref_list)
train_pred_df = pandas.DataFrame(train_pred_list)

df_concat = pandas.concat([train_ref_df, train_pred_df], axis=1)

df_concat.to_excel('C:/Users/Owner/Desktop/arteries/result/train/1.xlsx')
# train_ref_df.to_excel('C:/Users/Owner/Desktop/arteries/result/train/ref_1.xlsx')
# train_pred_df.to_excel('C:/Users/Owner/Desktop/arteries/result/train/pred_1.xlsx')


# テストデータを使った精度検証
# 
test_ref_list = []
test_pred_list = []
count_step_val = 0
val = 0

batch_val = len(id_test) // test_data_iterators
loop_kaisuu_val = batch_val + 1
print(batch_val)
print(loop_kaisuu_val)
amari_val = len(id_test) % test_data_iterators
print(amari_val)

for val in range(loop_kaisuu_val):
    if (count_step_val == 0):
        test_iter.reset()

        test_batch = test_iter.next()
        test_spc, test_ref = concat_examples(test_batch)  # Test Dataset
        for i in range(test_data_iterators):
            val_ref = test_ref[i]
            val_pred = net(test_spc[i].reshape(1, -1)).data
            # print(" %d" % (val_ref))
            test_ref_list.append(math.floor(val_ref))
            # print(" %d" % (val_pred) )
            test_pred_list.append(math.floor(val_pred))
            print("Ref : %d, Pred : %d" % (val_ref, val_pred))
            # 最初の10個だけ表示して大きな間違いがないかを確認

       
        count_step_val = count_step_val + 1
        # print("count : ", count_step)
    

    elif (0< count_step_val < batch_val):
        
        test_batch = test_iter.next()
        test_spc, test_ref = concat_examples(test_batch)  # Test Dataset
        for i in range(test_data_iterators):
            val_ref = test_ref[i]
            val_pred = net(test_spc[i].reshape(1, -1)).data
            # print(" %d" % (cal_ref))
            test_ref_list.append(math.floor (val_ref))
            # print(" %d" % (cal_pred) )
            test_pred_list.append(math.floor(val_pred))
            # print("")
            
        
        count_step_val = count_step_val + 1    
        # print("count_step", count_step)

    elif (count_step_val == batch_val):
        # print("last loop")
        test_batch = test_iter.next()
        test_spc, test_ref = concat_examples(test_batch)  # Test Dataset
        for i in range(amari_val):
            val_ref = test_ref[i]
            val_pred = net(test_spc[i].reshape(1, -1)).data
            # print(" %d" % (cal_ref))
            test_ref_list.append(math.floor(val_ref))
            # print(" %d" % (cal_pred) )
            test_pred_list.append(math.floor(val_pred))
        
        

print("")
print(len(test_ref_list))
print(len(test_pred_list))

test_ref_df = pandas.DataFrame(test_ref_list)
test_pred_df = pandas.DataFrame(test_pred_list)

df_test_concat = pandas.concat([test_ref_df, test_pred_df], axis=1)
df_test_concat.to_excel('C:/Users/Owner/Desktop/arteries/result/test/1.xlsx')

# test_ref_df.to_excel('C:/Users/Owner/Desktop/arteries/result/test/ref_1.xlsx')
# test_pred_df.to_excel('C:/Users/Owner/Desktop/arteries/result/test/pred_1.xlsx')
