# %%

# 大数の法則
import matplotlib.pyplot as plt
import numpy as np

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
