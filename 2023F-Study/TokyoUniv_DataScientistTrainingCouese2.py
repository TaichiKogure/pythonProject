#%%

#大数の法則
import matplotlib.pyplot as plt
import numpy as np

calc_time = 300
#サイコロ
sample_array = np.array([1,2,3,4,5,6])
number_cnt = np.arange(1, calc_time + 1)

#４つのパスを生成
for i in range(200):
    p = np.random.choice(sample_array, calc_time).cumsum()
    plt.plot(p/ number_cnt)
    plt.grid(True)

plt.show()
