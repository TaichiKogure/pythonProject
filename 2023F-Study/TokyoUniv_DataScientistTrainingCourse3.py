#%%
import numpy as np
import numpy.random as random
import scipy as sp

#可視化ライブラリ
import matplotlib.pyplot as plt
import  matplotlib as mpl
%matplotlib inline

%precision 3

#%%
#Numpy　インデックス参照
sample_array = np.arange(100)
print('sample_array:',sample_array)
#%%
#元のデータ
print(sample_array)

#前から６つ取得してスライスに入れる
sample_array_slice = sample_array[0:6]
print(sample_array_slice)
