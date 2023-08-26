#%%
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg') #グラフ表示するときこれがないと出てこないぽい。
import seaborn as sns
import matplotlib.pyplot as plt

#%matplotlib inline

# 散布図
random.seed(0)
x = np.random.rand(30)
y = np.sin(x) + np.random.rand(30)

plt.figure(figsize=(20,6))
plt.plot(x, y, 'o')
plt.title('Title Name')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

# 以下のコマンドでも実行可能
#plt.scatter(x,y)

#%%
#　ZIPファイルとかをダウンロードするためのライブラリ
import numpy as np
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame

#可視化ライブラリ
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%precision 3

import requests, zipfile
from io import StringIO
import io

from sklearn import linear_model

#%
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00356/student.zip'
#Get data from URL
r = requests.get(url, stream=True)

# load and extract zipfile
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

#%
student_data_math = pd.read_csv('student-mat.csv')
student_data_math.head()

#%
#区切りを変える
student_data_math = pd.read_csv('student-mat.csv', sep= ';')
student_data_math.head()

#
student_data_math.info()
#%
student_data_math['sex'].head()
#%
student_data_math['absences'].head()
#%
student_data_math.groupby('sex')['age'].mean()
#%
student_data_math.groupby('address')['G1','G3'].mean()
#%
matplotlib.use('TkAgg') #グラフ表示するときこれがないと出てこないぽい。
sns.pairplot(student_data_math,vars=['age','Medu','Fedu','absences','traveltime','studytime'],hue='sex')
plt.show()

#%%
#ヒストグラムを書く
plt.hist(student_data_math['absences'])
plt.xlabel('absences')
plt.ylabel('count')
plt.grid(True)
#平均値
print('average:',student_data_math['absences'].mean())
#中央値
print('median:',student_data_math['absences'].median())
#最頻値
print('mode:',student_data_math['absences'].mode())
#分散
print('var:',student_data_math['absences'].var())
#標準偏差
print('stddev:',student_data_math['absences'].std())
#%%
#パーセンたいる
student_data_math['absences'].describe()

#%%
XX = student_data_math.describe()
print(XX)
#%%
#箱ひげず
plt.boxplot([student_data_math['age'],student_data_math['absences'],student_data_math['failures'],student_data_math['freetime']])
plt.grid(True)
#%%
plt.boxplot(student_data_math['absences'])
plt.grid(True)

#%%
data = (
    student_data_math['age'],student_data_math['absences'],student_data_math['failures'],student_data_math['freetime'],
student_data_math['Medu'],student_data_math['Fedu'],student_data_math['traveltime'],student_data_math['studytime'],
student_data_math['famrel'],student_data_math['goout'],student_data_math['G1'],student_data_math['G2'],student_data_math['G3']
,student_data_math['Walc'],student_data_math['health']
        )
fig1, ax1 = plt.subplots()
ax1.set_title('Student')
ax1.set_xticklabels(
    ['age', 'absences', 'failures','freetime','Medu','Fedu','traveltime','studytime','famrel','goout','G1','G2','G3','walk','health']
)
plt.ylim([-3,50])
ax1.boxplot(data)
plt.show()
#%%
# 変動係数
#student_data_math['absences'].std()/student_data_math['absences'].mean()
student_data_math.std()/student_data_math.mean()