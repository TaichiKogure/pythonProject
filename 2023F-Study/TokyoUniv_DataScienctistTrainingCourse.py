# %%
from typing import Any

import random
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # グラフ表示するときこれがないと出てこないぽい。
import seaborn as sns
import matplotlib.pyplot as plt

# %matplotlib inline

# 散布図
random.seed(0)
x = np.random.rand(30)
y = np.sin(x) + np.random.rand(30)

plt.figure(figsize=(20, 6))
plt.plot(x, y, 'o')
plt.title('Title Name')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

# 以下のコマンドでも実行可能
# plt.scatter(x,y)

# %%
# 　ZIPファイルとかをダウンロードするためのライブラリ
import numpy as np
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame

# 可視化ライブラリ
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
# %precision 3

import requests, zipfile
from io import StringIO
import io

from sklearn import linear_model

# %
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00356/student.zip'
# Get data from URL
r = requests.get(url, stream=True)

# load and extract zipfile
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

# %
student_data_math = pd.read_csv('student-mat.csv')
student_data_math.head()

# %
# 区切りを変える
student_data_math = pd.read_csv('student-mat.csv', sep=';')
student_data_math.head()

#
student_data_math.info()
# %
student_data_math['sex'].head()
# %
student_data_math['absences'].head()
# %
student_data_math.groupby('sex')['age'].mean()
# %
student_data_math.groupby('address')['G1', 'G3'].mean()
# %
matplotlib.use('TkAgg')  # グラフ表示するときこれがないと出てこないぽい。
sns.pairplot(student_data_math, vars=['age', 'Medu', 'Fedu', 'absences', 'traveltime', 'studytime'], hue='sex')
plt.show()

# %%
# ヒストグラムを書く
plt.hist(student_data_math['absences'])
plt.xlabel('absences')
plt.ylabel('count')
plt.grid(True)
# 平均値
print('average:', student_data_math['absences'].mean())
# 中央値
print('median:', student_data_math['absences'].median())
# 最頻値
print('mode:', student_data_math['absences'].mode())
# 分散
print('var:', student_data_math['absences'].var())
# 標準偏差
print('stddev:', student_data_math['absences'].std())
# %%
# パーセンたいる
student_data_math['absences'].describe()

# %%
XX = student_data_math.describe()
print(XX)
# %%
# 箱ひげず
plt.boxplot([student_data_math['age'], student_data_math['absences'], student_data_math['failures'],
             student_data_math['freetime']])
plt.grid(True)
# %%
plt.boxplot(student_data_math['absences'])
plt.grid(True)

# %%
data = (
    student_data_math['age'], student_data_math['absences'], student_data_math['failures'],
    student_data_math['freetime'],
    student_data_math['Medu'], student_data_math['Fedu'], student_data_math['traveltime'],
    student_data_math['studytime'],
    student_data_math['famrel'], student_data_math['goout'], student_data_math['G1'], student_data_math['G2'],
    student_data_math['G3']
    , student_data_math['Walc'], student_data_math['health']
)
fig1, ax1 = plt.subplots()
ax1.set_title('Student')
ax1.set_xticklabels(
    ['age', 'absences', 'failures', 'freetime', 'Medu', 'Fedu', 'traveltime', 'studytime', 'famrel', 'goout', 'G1',
     'G2', 'G3', 'walk', 'health']
)
plt.ylim([-3, 50])
ax1.boxplot(data)
plt.show()
# %%
# 変動係数
# student_data_math['absences'].std()/student_data_math['absences'].mean()
student_data_math.std() / student_data_math.mean()
# %%
plt.plot(student_data_math['G1'], student_data_math['G3'], 'o')
plt.xlabel('G1 grade')
plt.ylabel('G3 grade')
plt.grid(True)
plt.show()

# %%
# 共分散行列
np.cov(student_data_math['G1'], student_data_math['G3'])

# G1とG3の共分散、G1の分散、G3の分散が出てくる。
print('G1の分散', student_data_math['G1'].var())
print('G3の分散', student_data_math['G3'].var())

# %%
sp.stats.pearsonr(student_data_math['G1'], student_data_math['G3'])
np.corrcoef([student_data_math['G1'], student_data_math['G3']])
# %%
sns.pairplot(student_data_math[['Dalc', 'Walc', 'G1', 'G2', 'G3']])
plt.grid(True)
plt.show()

# %%

import numpy as np
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame

# 可視化ライブラリ
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
# %precision 3

import requests, zipfile
from io import StringIO
import io

from sklearn import linear_model

# %
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00356/student.zip'
# Get data from URL
r = requests.get(url, stream=True)

# load and extract zipfile
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

# %%
student_data_por = pd.read_csv('student-por.csv')
student_data_por.head()

# %%
# 区切りを変える
student_data_por = pd.read_csv('student-mat.csv', sep=';')
student_data_por.head()

#
student_data_por.info()
# %%
student_data_por['sex'].head()
# %%
student_data_por['absences'].head()
# %%
student_data_por.groupby('sex')['age'].mean()
# %%
student_data_por.groupby('address')['G1'].mean()
# %%
matplotlib.use('TkAgg')  # グラフ表示するときこれがないと出てこないぽい。
sns.pairplot(student_data_por, vars=['age', 'Medu', 'Fedu', 'absences', 'traveltime', 'studytime'], hue='sex')
plt.show()
# %%
plt.plot(student_data_math['G1'], student_data_math['G3'], 'o')
plt.grid(True)
plt.show()
# %%
# %%
# 　上の方から引用。　
# 　再開するときにここ押す
# 　ZIPファイルとかをダウンロードするためのライブラリ
import numpy as np
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame
# 可視化ライブラリ
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
# %precision 3

import requests, zipfile
from io import StringIO
import io

from sklearn import linear_model

# %
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00356/student.zip'
# Get data from URL
r = requests.get(url, stream=True)
# load and extract zipfile
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()
student_data_math = pd.read_csv('student-mat.csv')

student_data_math = pd.read_csv('student-mat.csv', sep=';')
student_data_math.head()
# %%
from sklearn import linear_model

reg = linear_model.LinearRegression()

X = student_data_math.loc[:, ['G1']].values
Y = student_data_math['G3'].values
reg.fit(X, Y)
print('回帰曲線', reg.coef_)
print('切片', reg.intercept_)
# %%
plt.scatter(X, Y)
plt.xlabel('G1 Grade')
plt.ylabel('G3 Grade')
plt.plot(X, reg.predict(X))
plt.grid(True)
plt.show()
# %%
# 決定係数
reg.score(X, Y)

# %%
# 総合練習３−１
wine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
wine.head()
# %%
DES = wine.describe()
print(DES)
# %%
sns.pairplot(wine)
plt.show()

# %%
sns.jointplot(x="fixed acidity", y="citric acid", data=wine)
plt.show()
# test2
# %%
# 確率と統計
import numpy as np
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

np.random.seed(0)
# %%
# サイコロが取る数字を配列に格納
dice_data = np.array([1, 2, 3, 4, 5, 6])
print('一つだけランダムに抽出', np.random.choice(dice_data, 1))

# サイコロを１０００回振ってみる
calc_steps = 1000
# １−６のデータの中から１０００回の抽出を実施
dice_rolls = np.random.choice(dice_data, calc_steps)
# それぞれの数字がどれくらいの確率で出たか計算
for i in range(1, 7):
    p = len(dice_rolls[dice_rolls == i]) / calc_steps
    print(i, 'が出る確率', p)

# %%
calc_steps = 1000

# １から６までのデータの中から１０００回の抽出
dice_rolls = np.random.choice(dice_data, calc_steps)

# それぞれの数字がどのくらいの割合で抽出されたか

prob_data = np.array([])
for i in range(1, 7):
    p = len(dice_rolls[dice_rolls == i]) / calc_steps
    prob_data = np.append(prob_data, len(dice_rolls[dice_rolls == i]) / calc_steps)

plt.bar(dice_data, prob_data)
plt.grid(True)
plt.show()

# %%
# 二項分布
np.random.seed(0)
x = np.random.binomial(300, 0.9, 10000, )
plt.hist(x, cumulative=False, color="green", bins=30)
plt.grid(True)
plt.show()
# %%
# ベルヌーイ分布
# 0を表，１を裏とする，さんぷるすうは8個
prob_be_data = np.array([])
coin_data = np.array([0, 0, 0, 0, 0, 1, 1, 1])
# Uniqueで一意な値を抽出
for i in np.unique(coin_data):
    p = len(coin_data[coin_data == i]) / len(coin_data)
    print(i, 'が出る確率', p)
    print(prob_be_data)
    prob_be_data = np.append(prob_be_data, p)

print(prob_be_data)



#%%
plt.bar([0, 1], prob_be_data, align='center')
plt.xticks([0, 1], ['head', 'tail'])
plt.grid(True)
plt.show()

#%%
#poisson分布
x = np.random.poisson(7,1000)
plt.hist(x)
plt.grid(True)
plt.show()

#正規分布
#平均、標準偏差、サンプル数
y = np.random.normal(6,10,40000)
plt.hist(y)
plt.grid(True)
plt.show()

#%%
import requests
import zipfile
import numpy as np
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import io
from io import StringIO

zip_file_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00356/student.zip'
 rr = requests.get(zip_file_url,stream=True)
 z = zipfile.ZipFile(io.BytesIO(rr.content))
 z.extractall()
 #データ読み込み
 student_data_math = pd.read_csv('student-mat.csv', sep=';')
 #カーネル密度関数
 student_data_math.absences.plot(kind='kde', style='k--')
 #単純なヒストグラム
 student_data_math.absences.hist(density = True)
 plt.grid(True)
 plt.show()
