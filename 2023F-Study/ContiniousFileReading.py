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