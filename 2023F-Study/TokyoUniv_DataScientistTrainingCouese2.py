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

    for i in range(1000):
        cum_variables = np.random.choice(sample_array, N).comsum() * 1.0
        mean_array = np.append(mean_array, cum_variables[N-1] / N)

    plt.hist(mean_array)
    plt.grid(True)
    plt.show()
#"動かない"

function_central_theory(3)

#%%
num_simulations = 10000  # シミュレーションの回数
num_rolls = 100000  # サイコロを振る回数

sums = np.zeros(num_simulations)

for i in range(num_simulations):
    dice_rolls = np.random.randint(1, 7, num_rolls)  # サイコロをnum_rolls回振った結果
    sums[i] = np.sum(dice_rolls)  # サイコロの合計を記録

plt.hist(sums, bins=30, density=True, alpha=0.4, color='b', label='Simulated Sum')
plt.xlabel('Sum of Dice Rolls')
plt.ylabel('Probability Density')
plt.title('Central Limit Theorem Simulation')
plt.legend()
plt.show(
    #%%
