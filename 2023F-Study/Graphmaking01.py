#%%

import matplotlib.pyplot as plt
import numpy as np

# 定義範囲
t = np.arange(1, 10, 0.01)

# 三角関数を計算
y = np.sin(t)

# グラフのプロット
plt.plot(t, y)

# グラフ表示
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定義範囲
t = np.linspace(1, 10, 100)
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)

# 三角関数を計算
z = np.sin(theta)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

# 3Dグラフ作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, t, label='3D spiral')
ax.legend()

# グラフ表示
plt.show()