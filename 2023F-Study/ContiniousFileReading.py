import pandas as pd
import matplotlib.pyplot as plt
import glob

# CSVファイルが格納されているディレクトリを指定
csv_files = glob.glob('/path/to/your/csvfiles/*.csv')

plt.figure()

# 各CSVファイルに対して繰り返し処理
for csv_file in csv_files:
    # CSVファイルをデータフレームに変換
    df = pd.read_csv(csv_file)

    # データフレームをプロット（この例では0列目をx軸, 1列目をy軸と仮定）
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=csv_file)

plt.legend()
plt.show()

#%%
import pandas as pd
import matplotlib.pyplot as plt
import glob

# CSVファイルが格納されているディレクトリを指定
csv_files = glob.glob('/path/to/your/csvfiles/*.csv')

# csv_filesをソート
csv_files = sorted(csv_files)

plt.figure()

# 各CSVファイルに対して繰り返し処理
for csv_file in csv_files:
    # CSVファイルをデータフレームに変換
    df = pd.read_csv(csv_file)
    # データフレームをプロット（この例では0列目をx軸, 1列目をy軸と仮定）
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=csv_file)

plt.legend()
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# データの作成
np.random.seed(0)
x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)

# 描画の準備
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# プロット
ax.scatter(x, y, z)
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# データの作成
np.random.seed(0)
x = np.random.randn(100)
z = np.random.randn(100)
y = x + 2*z

# 描画の準備
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# プロット
ax.scatter(x, y, z)
plt.show()
#%%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Use rc_context as a context manager
with plt.rc_context({'animation.writer': 'pillow'}):

    # データの作成
    np.random.seed(0)
    x = np.random.randn(1000)
    z = np.random.randn(1000)
    y = 3*x + 2*z

    # 描画の準備
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    # 回転させるための関数
    def rotate(angle):
        ax.view_init(azim=angle)

    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=100)

    ani.save('rotation.gif', writer='pillow', fps=20)

    plt.show()
