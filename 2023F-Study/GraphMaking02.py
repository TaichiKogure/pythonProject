import numpy as np
import matplotlib.pyplot as plt

# data for x-axis
x = np.linspace(-2, 2, 400)

# data for y-axis for three different functions
y1 = np.exp(x)
y2 = np.power(3,x)*x
y3 = 3*np.log(x)+10

# Creating subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# Plotting data on the first subplot
ax1.plot(x, y1)
ax1.set_title('Plot of exponential function: e^x')
ax1.set_xlabel('x')
ax1.set_ylabel('y = e*e^2*x')

# Plotting data on the second subplot
ax2.plot(x, y2, 'r')
ax2.set_title('Plot of power function: 2^x')
ax2.set_xlabel('x')
ax2.set_ylabel('y = 3^x^4')

# Plotting data on the third subplot
ax3.plot(x[x >= 0], y3[x >= 0], 'g')  # log is only defined for x > 0
ax3.set_title('Plot of logarithmic function: log(x)')
ax3.set_xlabel('x')
ax3.set_ylabel('y = 2*log(x)')

# show plot
plt.tight_layout()
plt.show()

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Insert a Seaborn palette here
palette = sns.color_palette("bright", 20)

sns.palplot(palette)
plt.show()
#%%

import matplotlib.pyplot as plt
import seaborn as sns

# データ作成
data = [90, 85, 77, 95, 80, 70, 80, 85, 88, 90, 0, 30, 60, 78, 50]

# 箱ひげ図の作成
sns.boxplot(data=data)

# グラフの表示
plt.show()
#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import losses
from tensorflow.keras import optimizers

# 入力とクラスの数
input_shape = (28, 28, 1) # MNIST画像の形式
num_classes = 10 # MNISTのクラス数 (0-9の手書き数字)

# モデルの定義
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# モデルのコンパイル
model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])

# モデルの概要を表示
model.summary()

# モデルの概要を表示
model.summary()
#%%
import tensorflow as tf
print(tf.__version__)