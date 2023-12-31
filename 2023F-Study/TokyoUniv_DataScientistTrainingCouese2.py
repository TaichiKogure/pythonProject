# %%

# 大数の法則
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

calc_time = 300
# サイコロ
sample_array1 = np.array([1, 2, 3, 4, 5, 6])
number_cnt1 = np.arange(1, calc_time + 1)

# ４つのパスを生成
for i in range(200):
    p = np.random.choice(sample_array1, calc_time).cumsum()
    plt.plot(p / number_cnt1)
    plt.grid(True)

plt.show()

#%%

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

def plot_random_walks(dice_faces, trial_count, path_count=200):
    for _ in range(path_count):
        cumulative_path = np.random.choice(dice_faces, len(trial_count)).cumsum()
        plt.plot(cumulative_path / trial_count)
        plt.grid(True)
    plt.show()

# Law of large numbers
calc_time = 300
# Dice
dice_faces = np.array([1, 2, 3, 4, 5, 6])
trial_count = np.arange(1, calc_time + 1)
# Generate and plot paths
plot_random_walks(dice_faces, trial_count)
plt.show()
# %%
# 中心極限定理
import numpy as np
import matplotlib.pyplot as plt


def function_central_theory(N):
    sample_array = np.array([1, 2, 3, 4, 5, 6])
    numaber_cnt = np.arange(1, N + 1) * 1.0

    mean_array = np.array([])

    for i in range(100):
        cu_variables = np.random.choice(sample_array, N)
        cum_variables = np.cumsum(cu_variables) *1.0
        mean_array = np.append(mean_array, cum_variables[N-1] / N)

    plt.hist(mean_array, bins=10, density=True, alpha=0.9)
    plt.grid(True)
    plt.show()
#"動かない"

function_central_theory(10000)

#%%
num_simulations = 10000  # シミュレーションの回数
num_rolls = 1000  # サイコロを振る回数

sums = np.zeros(num_simulations)

for i in range(num_simulations):
    dice_rolls = np.random.randint(1, 7, num_rolls)  # サイコロをnum_rolls回振った結果
    sums[i] = np.sum(dice_rolls)  # サイコロの合計を記録

plt.hist(sums, bins=200, density=True, alpha=0.8, color='b', label='Simulated Sum')
plt.xlabel('Sum of Dice Rolls')
plt.ylabel('Probability Density')
plt.title('Central Limit Theorem Simulation')
plt.legend()
plt.show()
#%%
#カイ二条分布
#自由度２、１０、６０に従うかい2乗分布
for df, c in zip([2,4,10],["#6495ed","#daa520","#ff00ff"]):
    x = np.random.chisquare(df, 100000)
    plt.hist(x, 50, color = c)
plt.grid(True)
plt.show()
#%%
import pandas as pd
#統計的検定
student_data_math =pd.read_csv('student-mat.csv',sep =';')
student_data_por = pd.read_csv('student-por.csv',sep = ';')
#マージする
student_data_merge: DataFrame = pd.merge(student_data_math, student_data_por
                              , on=['school','sex','age','address','famsize','Pstatus','Medu',
                                    'Fedu','Mjob','reason','nursery','internet']
                              , suffixes=('_math','_por'))
print('G1数学の成績平均：', student_data_merge.G1_math.mean())
print('G2数学の成績平均：', student_data_merge.G2_math.mean())
print('G3数学の成績平均：', student_data_merge.G3_math.mean())
print('G1ポルトガル語の成績平均は', student_data_merge.G1_por.mean())
print('G2ポルトガル語の成績平均は', student_data_merge.G2_por.mean())
print('G3ポルトガル語の成績平均は', student_data_merge.G3_por.mean())

#%%
from scipy import stats
t, p = stats.ttest_rel(student_data_merge.G1_math, student_data_merge.G1_por)
print('p値 = ', p)





