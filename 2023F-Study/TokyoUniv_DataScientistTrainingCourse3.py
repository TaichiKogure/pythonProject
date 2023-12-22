#%%
import numpy as np
import numpy.random as random
import scipy as sp

#可視化ライブラリ
import matplotlib.pyplot as plt
import  matplotlib as mpl
%matplotlib inline

%precision 3

#%%
#Numpy　インデックス参照
sample_array = np.arange(100)
print('sample_array:',sample_array)
#%%
#元のデータ
print(sample_array)

#前から６つ取得してスライスに入れる
sample_array_slice = sample_array[0:6]
print(sample_array_slice)
#tst

#%%
import pybamm
model = pybamm.lithium_ion.DFN()  # Doyle-Fuller-Newman model
sim = pybamm.Simulation(model)
sim.solve([0, 3600])  # solve for 1 hour
sim.plot()

#%%import pandas as pd
import pandas as pd
# データフレームの作成
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 32, 37],
    'City': ['NY', 'LA', 'SF']
})

# 'Age' が 30 以上のデータだけ抽出
df_over_30 = df[df['Age'] > 30]
print(df_over_30)

# 'City' が 'NY' のデータだけ抽出
df_ny = df[df['City'] == 'NY']
print(df_ny)

# 複数の条件を組み合わせることも可能
# 'Age' が 30 以上 かつ 'City' が 'LA' のデータだけを抽出
df_over_30_and_la = df[(df['Age'] > 30) & (df['City'] == 'LA')]
print(df_over_30_and_la)

#%%
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ランダムデータ生成
np.random.seed(0)  # 同じ結果を再現するためにシードを固定
group1 = np.random.normal(loc=0.0, scale=1.0, size=1000)  # 平均0, 標準偏差1の正規分布からのランダムデータ
group2 = np.random.normal(loc=0.5, scale=1.0, size=1000)  # 平均0.5, 標準偏差1の正規分布からのランダムデータ

# t検定
t_stat, p_val = stats.ttest_ind(group1, group2)

print('t統計量: ', t_stat)
print('p値: ', p_val)

# データのヒストグラム
bins = np.linspace(-4, 4, 100)
plt.hist(group1, bins, alpha=0.5, label='group1')
plt.hist(group2, bins, alpha=0.5, label='group2')
plt.legend(loc='upper right')
plt.show()
#%%
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 乱数のシードを固定
np.random.seed(0)

data = []

# 平均が10種類のデータ生成
for i in range(10):
    temp_data = np.random.normal(loc=i*0.5, scale=1.0, size=1000)  # 平均を0, 0.5, 1, ...と変える
    data.append(temp_data)

# データのヒストグラム
bins = np.linspace(-4, 8, 200)
for i, temp_data in enumerate(data):
    plt.hist(temp_data, bins, alpha=0.5, label=f'group{i+1}')

plt.legend(loc='upper right')
plt.show()

# t検定
for i in range(10):
    for j in range(i+1, 10):  # 各ペアに対して一回ずつt検定を行う
        t_stat, p_val = stats.ttest_ind(data[i], data[j])
        print(f't検定結果 (group{i+1} vs group{j+1}): t統計量 = {t_stat}, p値 = {p_val}')


#%%
import pandas as pd

# Excelファイルを読み込む
xlsx = pd.ExcelFile('Booktest.xlsx')

# Excelの各シートをループして処理
for sheet_name in xlsx.sheet_names:
    # シートをDataFrameに読み込む
    df = pd.read_excel(xlsx, sheet_name=sheet_name)

    # ユーザに対話形式でCSVファイルの名前を入力させる
    csv_file_name = input(f"Enter the name of the CSV for the sheet '{sheet_name}' (without .csv): ")

    # DataFrameをCSVとして出力
    df.to_csv(f"{csv_file_name}.csv", index=False)
#%%
import os
import pandas as pd

filename = '../Booktest.xlsx'

# Ensure the file exists before trying to open it
if os.path.isfile(filename):
    xlsx = pd.ExcelFile(filename)
else:
    print(f"The file '{filename}' does not exist.")

#%%
import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込み、データフレームとして保存
df1 = pd.read_csv('separate1.csv')
df2 = pd.read_csv('separate2.csv')
df3 = pd.read_csv('separate3.csv')

plt.figure()

# プロットを作成します
plt.plot(df1, label='separate1')
plt.plot(df2, label='separate2')
plt.plot(df3, label='separate3')

# y軸の範囲を指定 （例：0から20）
plt.ylim(0, 50)

# 凡例を表示
plt.legend()

plt.show()
#%%
import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込み、データフレームとして保存
df1 = pd.read_csv('separate1.csv')
df2 = pd.read_csv('separate2.csv')
df3 = pd.read_csv('separate3.csv')

plt.figure()

# プロットを作成します
plt.semilogy(df1, label='separate1')
plt.semilogy(df2, label='separate2')
plt.semilogy(df3, label='separate3')

# 凡例を表示
plt.legend()

plt.show()
#%%
%matplotlib tk
import pybamm
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model)
sim.solve([0, 3600])  # Solve for 1 hour
sim.plot()
#%%
%matplotlib tk
import pybamm

# Define a model
model = pybamm.lithium_ion.DFN()

# Default parameter values
param = model.default_parameter_values

# Change a specific parameter value
param.update({"Current function [A]": 6})  # for example, set the current function to 3 A

# Set up and solve a simulation
sim = pybamm.Simulation(model, parameter_values=param)

sim.solve([0, 3600])
sim.plot()
#%%

import pybamm
import numpy as np
import matplotlib.pyplot as plt

# モデルの一覧を作成します
models = [pybamm.lithium_ion.SPM(), pybamm.lithium_ion.DFN()]

# 各モデルでシミュレーションを行い、結果をプロットします
for model in models:
    experiment = pybamm.Experiment(
        ["Charge at 0.5C until 4.2V",
         "Hold at 4.2V until C/10",
         "Rest for 0.2 hour",
         "Discharge at 2C until 3V",
         "Rest for 0.1 hour"]
    )
    sim = pybamm.Simulation(model, experiment=experiment)
    sim.solve()

    # シミュレーション結果をプロットします
    plt.figure()
    sim.plot()
    #%%
    import pybamm
    import numpy as np
    import matplotlib.pyplot as plt

    # Declare a variable
    t = pybamm.t  # time
    x = pybamm.Variable("x", domain="test domain")

    # Create a model
    model = pybamm.BaseModel()

    # Define the model equations
    model.rhs = {x: -x}  # dx/dt = -x
    model.initial_conditions = {x: 1}  # initial condition x(t=0) = 1
    model.variables = {"x": x}

    # Use the `pybamm.Simulation` class to solve the model
    simulation = pybamm.Simulation(model)
    solution = simulation.solve(t_eval=np.linspace(0, 1, 100))

    # plot the solution
    plt.plot(solution.t, solution["x"].data)
    plt.xlabel("Time [s]")
    plt.ylabel("x")
    plt.show()

    #%%
    import pybamm
    import numpy as np

    # parameters
    params = pybamm.ParameterValues({
        "Negative electrode thickness [m]": 0.0001,
        "Positive electrode thickness [m]": 0.0001,
        "Maximum concentration in negative electrode [mol.m-3]": 25000,
        "Maximum concentration in positive electrode [mol.m-3]": 50000,
        "Negative electrode conductivity [S.m-1]": 100,
        "Positive electrode conductivity [S.m-1]": 10,
        "Negative particle radius [m]": 1e-6,
        "Positive particle radius [m]": 1e-6,
        # Include more parameters as needed
    })

    model = pybamm.lithium_ion.DFN()  # For example, if you were using DFN model

    experiment = pybamm.Experiment(
        ["Discharge at C/10 for 10 hours"],
        period="1 hour",
    )

    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)
    sim.solve()

    sim.plot(["Time [h]", "Terminal voltage [V]"])

    #%%
    # Assuming parameter_values is what you are working with
    if "Nominal cell capacity [A.h]" not in parameter_values:
        raise KeyError(
            "The key 'Nominal cell capacity [A.h]' does not exist in the parameter values. Please check the spelling or the existence of the key.")
    else:
        # do your computation
        capacity = parameter_values["Nominal cell capacity [A.h]"]
        #%%
        parameter_values = pybamm.ParameterValues("Chen2020")
        model = pybamm.lithium_ion.DFN()
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        sim.solve([0, 3600])
        sim.plot()

        #%%
        # Solve the simulation until 1 hour (3600s)
    sim.solve([0, 3600])

    # Access the solution object
    solution = sim.solution

    # Specify the time point as 1800s
    time_point = 1800

    # Specify the variable you are interested in
    # Let's assume you want current and voltage here (replace it with your variables of interest)
    variable_of_interest = ["Current [A]", "Terminal voltage [V]"]

    # Get the data at 1800s
    data_at_1800s = {var: solution[var](time_point) for var in variable_of_interest}

    # Print the data at 1800s
    for var, value in data_at_1800s.items():
        print(f"{var} at {time_point}s: {value}")

#%%
        parameter_values = pybamm.ParameterValues("Chen2020")
        model = pybamm.lithium_ion.DFN()
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        sim.solve([0, 3600])
        sim.plot()
        # Solve the simulation until 1 hour (3600s)
        sim.solve([0, 3600])

        # Access the solution object
        solution = sim.solution

        # Specify the time point as 1800s
        time_point = 1800

        # Specify the variable you are interested in
        # Let's assume you want current and voltage here (replace it with your variables of interest)
        variable_of_interest = ["Current [A]", "Terminal voltage [V]"]

        # Get the data at 1800s
        data_at_1800s = {var: solution[var](time_point) for var in variable_of_interest}

        # Print the data at 1800s
        for var, value in data_at_1800s.items():
            print(f"{var} at {time_point}s: {value}")

#%%
# Assuming 'model' is your PyBaMM model
for var_name in model.variables.keys():
    print(var_name)
#%%
import pandas as pd

# Solve the simulation until 1 hour (3600s)
sim.solve([0, 3600])

# Specify the time point as 1800s
time_point = 1800

# Access the solution object
solution = sim.solution

# We create a new dictionary to store data
data_at_1800s = {}

# We extract data for each variable at the given time point
for var_name in sim.model.variables.keys():
    try:
        data_at_1800s[var_name] = solution[var_name](time_point)
    except:
        data_at_1800s[var_name] = "Not a number"

# Create dataframe from the dictionary
data_frame = pd.DataFrame.from_records([data_at_1800s])

# Write dataframe to csv file
data_frame.to_csv("variables_at_1800s.csv", index=False)
#%%
import matplotlib.pyplot as plt
import numpy as np

# Define x and y
x = np.arange(1, 101)
y = np.cumsum(np.random.randn(100))

# Compute lines of best fit
poly1 = np.poly1d(np.polyfit(x, y, 1))
poly2 = np.poly1d(np.polyfit(x, y, 2))
poly3 = np.poly1d(np.polyfit(x, y, 3))

# Create a space for x values to evaluate the polynomial
x_for_poly = np.linspace(x[0], x[-1], 500)

# Create figure
plt.figure(figsize=(10,6))

# Plot original data
plt.plot(x, y, label='Original data')

# Plot polynomial fits
plt.plot(x_for_poly, poly1(x_for_poly), 'r', label='Linear fit: %s' % poly1)
plt.plot(x_for_poly, poly2(x_for_poly), 'b', label='Quadratic fit: %s' % poly2)
plt.plot(x_for_poly, poly3(x_for_poly), 'g', label='Cubic fit: %s' % poly3)

plt.title('Monotonically Increasing Graph with Best Fit Lines')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.grid(True)
plt.show()

print('The equation of the linear fit is: \n%s' % poly1)
print('The equation of the quadratic fit is: \n%s' % poly2)
print('The equation of the cubic fit is: \n%s' % poly3)
