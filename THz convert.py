# テラヘルツ出力
# txt → xlsx
"""

テラヘルツのファイルを変換する．
.txt→.xlsx

+α 強度→透過率→吸光度，波数→波長→周波数
テラヘルツの測定で得られたテキストファイルを読み込み
↓
波数を波長，周波数に換算（変換？）する
↓
強度を透過率，吸光度に換算する．
↓
Excelファイル(.xlsx)出力する

"""
# モジュールのインポート
import numpy

import pandas
import glob

"""
1.データ読み込み

"""
# BKGのデータを入れる
# BKG : BKGデータをpandasで読み込む
# BKG_drop : 文字部分を削除（数値のみ）
BKG = pandas.read_csv("H:/tamura/20191114/BKG_1.txt", encoding='shift-jis', header=None, sep='\t')
BKG_last = pandas.read_csv("H:/tamura/20191113/BKG_2.txt",
                          encoding='shift-jis', header=None, sep='\t')

# BKG_iloc : BKGの20~704行目を抽出
# bkg : dtypeをobject→floatに変更（文字型→浮動小数点）
bkg = BKG.iloc[19:705]
bkg_last = BKG_last.iloc[19:705]
#bkg = fa.astype(float)
print("BKGの配列 : ", bkg.shape)
print("BKG lastの配列 : ", bkg_last.shape)

# bkg_df : DataFrame → numpy.ndarray に変更
bkg_df = bkg.values
bkg_last_df = bkg_last.values
# x,y : 波数と強度に分割
bkg_x = bkg_df[:, 0].astype(numpy.float32)
bkg_y = bkg_df[:, 1].astype(numpy.float32)
print("wavenumber[0] : ", bkg_x[0])
print("SB[0] : ", bkg_y[0])

bkg_last_x = bkg_last_df[:, 0].astype(numpy.float32)
bkg_last_y = bkg_last_df[:, 1].astype(numpy.float32)
print("last_wavenumber[0] : ", bkg_last_x[0])
print("last_SB[0] : ", bkg_last_y[0])


# 測定結果
# 測定データの読み込み
thz_1 = pandas.read_csv("H:/tamura/20191114/3.0_0.txt",
                        encoding='shift-jis', header=None, sep='\t')

thz_2 = pandas.read_csv("H:/tamura/20191114/3.0_330.txt",
                        encoding='shift-jis', header=None, sep='\t')

thz_3 = pandas.read_csv("H:/tamura/20191114/3.0_300.txt",
                       encoding='shift-jis', header=None, sep='\t')

thz_4 = pandas.read_csv("H:/tamura/20191114/3.0_270.txt",
                      encoding='shift-jis', header=None, sep='\t')


# 測定データの切り出し
thz_a = thz_1.iloc[19:705]
thz_a_1 = thz_a.astype(float)

thz_b = thz_2.iloc[19:705]
thz_b_1 = thz_b.astype(float)

thz_c = thz_3.iloc[19:705]
thz_c_1 = thz_c.astype(float)

thz_d = thz_4.iloc[19:705]
thz_d_1 = thz_d.astype(float)

"""
"""

# bkg_df : DataFrame → numpy.ndarray に変更
thz_a_df = thz_a_1.values
# x,y : 波数と強度に分割
thz_a_sb = thz_a_df[:, 1]
print("3.3-1 SB : ", thz_a_sb[0])

# bkg_df : DataFrame → numpy.ndarray に変更
thz_b_df = thz_b_1.values
# x,y : 波数と強度に分割
thz_b_sb = thz_b_df[:, 1]
print("5.5-1 SB : ", thz_b_sb[0])


# bkg_df : DataFrame → numpy.ndarray に変更
thz_c_df = thz_c_1.values
# x,y : 波数と強度に分割
thz_c_sb = thz_c_df[:, 1]
print(" PET3.0-0 SB : ", thz_c_sb[0])

# bkg_df : DataFrame → numpy.ndarray に変更
thz_d_df = thz_d_1.values
# x,y : 波数と強度に分割
thz_d_sb = thz_d_df[:, 1]
print(" PET3.0-2 SB : ", thz_d_sb[0])

"""
3. 周波数への換算

"""

# 波数(wavenumber)→波長(wavelength), 周波数(THz)
## 波数（cm⁻¹）→波長（㎛）
wv = 10000 / bkg_x
## 波数(cm⁻¹)→周波数(THz)
fre = 300 / wv
#print(fre)
print("wavelength[0](㎛) = ", wv[0])
print("frequency[0](THz) = ", fre[0])

"""
4. 透過率，吸光度への換算

"""
# 透過率の計算(3-3-1)
T_a = (thz_a_sb / bkg_y ) * 100
#透過率（%T）（透過光強度/入射光強度（BKGの強度））*100(%)
print("0 透過率 : ", T_a[0])
#T_last_a = (thz_a_sb / bkg_last_y) * 100
#print(" 0 透過率: ", T_last_a[0])
# 吸光度の計算
# log10(BKG強度/透過光強度)
a = bkg_y / thz_a_sb
Abs_a = numpy.log10(a)
print("0 吸光度 : " , Abs_a[0])
#last_a = bkg_last_y / thz_a_sb
#Abs_last_a = numpy.log10(last_a)
#print("330 吸光度 : ", Abs_last_a[0])


# 透過率の計算(5-5-1)
T_b = (thz_b_sb / bkg_y ) * 100
#透過率（%T）（透過光強度/入射光強度（BKGの強度））*100(%)
print("330 透過率 : ", T_b[0])
#T_last_b = (thz_b_sb / bkg_last_y) * 100
#print("180(last) 透過率 : ", T_last_b[0])
# 吸光度の計算
# log10(BKG強度/透過光強度)
b = bkg_y / thz_b_sb
Abs_b = numpy.log10(b)
print("330 吸光度 : ", Abs_b[0])
#last_b = bkg_last_y / thz_b_sb
#Abs_last_b = numpy.log10(last_b)
#print("180(last) 吸光度 : ", Abs_last_b[0])


# 透過率の計算(3個め)
T_c = (thz_c_sb / bkg_y ) * 100
#透過率（%T）（透過光強度/入射光強度（BKGの強度））*100(%)
print("300 透過率 : ", T_c[0])
#T_last_c = (thz_c_sb / bkg_last_y) * 100
#print("210 (last) 透過率 : ", T_last_c[0])
# 吸光度の計算
# log10(BKG強度/透過光強度)
c = bkg_y / thz_c_sb
Abs_c = numpy.log10(c)
print("300 吸光度 : ", Abs_c[0])
#last_c = bkg_last_y / thz_c_sb
#Abs_last_c = numpy.log10(last_c)
#print("210(last) 吸光度 : ", Abs_last_c[0])

# 透過率の計算(4個め)
T_d = (thz_d_sb / bkg_y ) * 100
#透過率（%T）（透過光強度/入射光強度（BKGの強度））*100(%)
print("270 透過率 : ", T_d[0])
#T_last_d = (thz_d_sb / bkg_last_y) * 100
#print("270 透過率 : ", T_last_d[0])
# 吸光度の計算
# log10(BKG強度/透過光強度)
d = bkg_y / thz_d_sb
Abs_d = numpy.log10(d)
print("270 吸光度 : ", Abs_d[0])
#last_d = bkg_last_y / thz_d_sb
#Abs_last_d = numpy.log10(last_d)
#print("240(last) 吸光度 : ", Abs_last_d[0])


"""
5. データの結合
"""
# 波長，周波数の結合
wv_df = pandas.DataFrame(wv, columns=['wavelength(㎛)'])
fre_df = pandas.DataFrame(fre, columns=['frequency(THz)'])
df_index = pandas.concat([wv_df, fre_df], axis=1)

# 3-3-1のデータの結合
T_a_df = pandas.DataFrame(T_a, columns=['0 透過率(%T)'])
Abs_a_df = pandas.DataFrame(Abs_a, columns=['0 吸光度(Abs)'])
#T_last_a_df = pandas.DataFrame(T_last_a, columns=['150 透過率(%T) last'])
#Abs_last_a_df = pandas.DataFrame(Abs_last_a, columns=['150 吸光度 last'])
df_link_a = pandas.concat([ T_a_df, Abs_a_df], axis=1)

# 5-5-1 の結合
T_b_df = pandas.DataFrame(T_b, columns=['330 透過率(%T)'])
Abs_b_df = pandas.DataFrame(Abs_b, columns=['330 吸光度(Abs)'])
#T_last_b_df = pandas.DataFrame(T_last_b, columns=['180 透過率(%T) last'])
#Abs_last_b_df = pandas.DataFrame(Abs_last_b, columns=['180 吸光度 last'])
df_link_b = pandas.concat([T_b_df, Abs_b_df], axis=1)

# 3個め の結合
T_c_df = pandas.DataFrame(T_c, columns=['300 透過率(%T)'])
Abs_c_df = pandas.DataFrame(Abs_c, columns=['300 吸光度(Abs)'])
#T_last_c_df = pandas.DataFrame(T_last_c, columns=['210 透過率(%T) last'])
#Abs_last_c_df = pandas.DataFrame(Abs_last_c, columns=['210 吸光度 last'])
df_link_c = pandas.concat([T_c_df, Abs_c_df], axis=1)

# 4個め の結合
T_d_df = pandas.DataFrame(T_d, columns=['270 透過率(%T)'])
Abs_d_df = pandas.DataFrame(Abs_d, columns=['270 吸光度(Abs)'])
#T_last_d_df = pandas.DataFrame(T_last_d, columns=['240  透過率(%T) last'])
#Abs_last_d_df = pandas.DataFrame(Abs_last_d, columns=['240 吸光度 last'])
df_link_d = pandas.concat([T_d_df, Abs_d_df], axis=1)

# 全データの結合
df_join = pandas.concat([df_index, df_link_a, df_link_b, df_link_c,df_link_d], axis=1)
df_join.index = bkg_x

"""
6. Excel(.xlsx)として出力

"""
# .xlsxとして出力
df_join.to_excel("G:/akita.2191114.xlsx")


"""
7. オマケ　グラフの描画，出力
(今後実装予定)
"""


# オマケ　グラフの描画
# plt.plot(thz_x, thz_y)

# グラフ化可能　
# 軸の反転，複数のグラフの重ね書き，強度→透過率計算)