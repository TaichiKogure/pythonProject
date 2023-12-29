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
#%%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Prepare the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

frames = []

# Generate frames
for z in np.arange(1, 11):
    x = np.full(100, z)
    y = np.full(100, z)**2 + 3 * z
    frame = ax.scatter(x, y, np.full(100, z))
    frames.append([frame])

# Create animation
with plt.rc_context({'animation.writer': 'pillow'}):
    ani = animation.ArtistAnimation(fig, frames, interval=500, blit=True,)

    # Save as GIF
    ani.save('rotation.gif', writer='pillow', fps=60)

plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# tの値を生成
t_values = [np.linspace(1, 10, 200), np.linspace(1, 20, 400), np.linspace(1, 30, 600)]

# 3Dフィギュアを作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for t in t_values:
    # xとyを生成
    x = t ** 2
    y = x ** 3
    # z軸にtを使う(もしくは別の関数)
    z = t*x
    # データをプロット
    ax.plot(x, y, z)

for t in t_values:
    # xとyを生成
    x = t ** 4
    y = x ** 2
    # z軸にtを使う(もしくは別の関数)
    z = t*x
    # データをプロット
    ax.plot(x, y, z)
# グラフを表示
plt.show()