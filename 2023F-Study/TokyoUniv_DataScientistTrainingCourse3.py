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
#tst

#%%
import pybamm
model = pybamm.lithium_ion.DFN()  # Doyle-Fuller-Newman model
sim = pybamm.Simulation(model)
sim.solve([0, 3600])  # solve for 1 hour
sim.plot()

#%%import pandas as pd
import pandas as pd
# データフレームの作成
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 32, 37],
    'City': ['NY', 'LA', 'SF']
})

# 'Age' が 30 以上のデータだけ抽出
df_over_30 = df[df['Age'] > 30]
print(df_over_30)

# 'City' が 'NY' のデータだけ抽出
df_ny = df[df['City'] == 'NY']
print(df_ny)

# 複数の条件を組み合わせることも可能
# 'Age' が 30 以上 かつ 'City' が 'LA' のデータだけを抽出
df_over_30_and_la = df[(df['Age'] > 30) & (df['City'] == 'LA')]
print(df_over_30_and_la)

#%%
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ランダムデータ生成
np.random.seed(0)  # 同じ結果を再現するためにシードを固定
group1 = np.random.normal(loc=0.0, scale=1.0, size=1000)  # 平均0, 標準偏差1の正規分布からのランダムデータ
group2 = np.random.normal(loc=0.5, scale=1.0, size=1000)  # 平均0.5, 標準偏差1の正規分布からのランダムデータ

# t検定
t_stat, p_val = stats.ttest_ind(group1, group2)

print('t統計量: ', t_stat)
print('p値: ', p_val)

# データのヒストグラム
bins = np.linspace(-4, 4, 100)
plt.hist(group1, bins, alpha=0.5, label='group1')
plt.hist(group2, bins, alpha=0.5, label='group2')
plt.legend(loc='upper right')
plt.show()
#%%
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 乱数のシードを固定
np.random.seed(0)

data = []

# 平均が10種類のデータ生成
for i in range(10):
    temp_data = np.random.normal(loc=i*0.5, scale=1.0, size=1000)  # 平均を0, 0.5, 1, ...と変える
    data.append(temp_data)

# データのヒストグラム
bins = np.linspace(-4, 8, 200)
for i, temp_data in enumerate(data):
    plt.hist(temp_data, bins, alpha=0.5, label=f'group{i+1}')

plt.legend(loc='upper right')
plt.show()

# t検定
for i in range(10):
    for j in range(i+1, 10):  # 各ペアに対して一回ずつt検定を行う
        t_stat, p_val = stats.ttest_ind(data[i], data[j])
        print(f't検定結果 (group{i+1} vs group{j+1}): t統計量 = {t_stat}, p値 = {p_val}')