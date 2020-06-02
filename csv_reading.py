"""
2020.06.02
東北大学に送ったcsvファイルをExcelに変換する
関数名が適当やから後日直す
"""

import csv, os
import pandas
import glob
import numpy

A_list = sorted(glob.glob("C:/Users/Owner/Desktop/csv_folder/被験者A/A_*"))
B_list = sorted(glob.glob("C:/Users/Owner/Desktop/csv_folder/被験者B/B_*"))
C_list = sorted(glob.glob("C:/Users/Owner/Desktop/csv_folder/被験者C/C_*"))
D_list = sorted(glob.glob("C:/Users/Owner/Desktop/csv_folder/被験者D/D_*"))
E_list = sorted(glob.glob("C:/Users/Owner/Desktop/csv_folder/被験者E/E_*"))
F_list = sorted(glob.glob("C:/Users/Owner/Desktop/csv_folder/被験者F/F_*"))
G_list = sorted(glob.glob("C:/Users/Owner/Desktop/csv_folder/被験者G/G_*"))
H_list = sorted(glob.glob("C:/Users/Owner/Desktop/csv_folder/被験者H/H_*"))
I_list = sorted(glob.glob("C:/Users/Owner/Desktop/csv_folder/被験者I/I_*"))
J_list = sorted(glob.glob("C:/Users/Owner/Desktop/csv_folder/被験者J/J_*"))

# print(A_list)

for i, fname in enumerate(A_list):
    xls = pandas.read_csv(fname, header=None)
    name = os.path.split(fname)[1].split(".")[0]
    
    if i == 0:       
        arr = xls.as_matrix()[:,0]  # numpy.array
        hddr = name # add:: 2018.07.23
    else:
        arr = numpy.vstack((arr, xls.as_matrix()[:,0]))
        hddr = numpy.hstack((hddr, name)) # add:: 2018.07.23

print(arr.shape)        


for list_ in(B_list, C_list, D_list, E_list, F_list, G_list, H_list, I_list, J_list):
    for i, fname in enumerate(list_):
        xls = pandas.read_csv(fname,header=None)
        arr = numpy.vstack((arr, xls.as_matrix()[:,0]))
        hddr = numpy.hstack((hddr, os.path.split(fname)[1].split(".")[0])) # add:: 2018.07.23
    print(arr.shape)
    
arr_df_pre = pandas.DataFrame(arr)
arr_df = arr_df_pre.T

arr_df.to_excel("C:/Users/Owner/Desktop/csv_folder/fbg_systolic_file.xlsx")
arr_df.to_excel("C:/Users/Owner/Desktop/csv_folder/fbg_diastolic_file.xlsx")
arr_df.to_excel("C:/Users/Owner/Desktop/csv_folder/fbg_glucose_file.xlsx")

"""
systolic:収縮期血圧
diastolic:拡張期血圧
glucose:血糖値
"""