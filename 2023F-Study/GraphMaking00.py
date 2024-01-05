import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(1, 100, 50)

for i in range(20):
    y = np.full(50, i)  # y value is fixed to differentiate the graphs along y axis
    z = x * i
    ax.plot(x, y, z)

ax.set_xlabel('X')
ax.set_ylabel('Graph index')
ax.set_zlabel('f(X)')

plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(1, 10, 100)


def plot_graph(i):
    ax.clear()  # clear previous plot
    for j in range(10):
        y = np.full(100, j)  # y value is fixed to differentiate the graphs along y axis
        z = np.exp(x + j)
        ax.plot(x, y, z)
        ax.set_xlabel('X')
        ax.set_ylabel('Graph index')
        ax.set_zlabel('f(X)')
    ax.view_init(elev=20., azim=i)


ani = FuncAnimation(fig, plot_graph, frames=np.arange(0, 360, 1), interval=100)
# ani.save('rotation.gif', writer='imagemagick')
ani.save('animation.gif', writer='pillow')
plt.show()

# %%

import json


def tag_files():
    # 入力を得る
    file_path = input("Enter the path of the file: ")
    tags = input("Enter the tags, separated by commas: ").split(',')

    # 既存のタグをロードする（存在すれば）
    try:
        with open("file_tags.json", "r") as f:
            file_tags = json.load(f)
    except:
        file_tags = {}

    # 新しいタグを保存
    file_tags[file_path] = tags

    with open("file_tags.json", "w") as f:
        json.dump(file_tags, f)


if __name__ == "__main__":
    tag_files()
