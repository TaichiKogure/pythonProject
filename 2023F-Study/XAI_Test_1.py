#%%
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

iris = sns.load_dataset('iris')

sns.histplot(iris, x='sepal_length')
plt.show()

#%%
# Pybamm基本コード
import pybamm
import os

pybamm.set_logging_level("INFO")
experiment = pybamm.Experiment(
    [("Discharge at C/20 for 1 hour", "Rest for 1 hour")] * 20,
)
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, experiment=experiment, solver=pybamm.CasadiSolver())
sim.solve()
sim.plot()

#Mac上だと、Pyデータを使わないとクイックプロットが書けない様子。

#%%
# Pybamm基本コード
import pybamm
experiment = pybamm.Experiment(
    [
        ("Discharge at C/10 for 10 hours or until 3.3 V",
        "Rest for 1 hour",
        "Charge at 1 A until 4.1 V",
        "Hold at 4.1 V until 50 mA",
        "Rest for 1 hour")
    ]
    * 3,
)
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, experiment=experiment, solver=pybamm.CasadiSolver())
sim.solve()
sim.plot()

#%%

#######################################
#Practical Explainable AI Using Python#
#######################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy import linalg

df = pd.read_csv('automobile.csv')
df.info()
df.head()

##%%

df['Mileage']= pd.to_numeric(df.Mileage.replace('+AC0-1','0'))
df['EngineCC']= pd.to_numeric(df.EngineCC.replace('+AC0-1','0'))
df['PowerBhp']= pd.to_numeric(df.PowerBhp.replace('+AC0-1','0'))

##%%
import matplotlib
matplotlib.use('TkAgg') #グラフ表示するときこれがないと出てこないぽい。
import seaborn as sns
sns.pairplot(df[['Price','Age','Odometer','Mileage','EngineCC','PowerBhp']])

##%%
import seaborn as sns

# plt.figure(figsize=(10, 6))
# sns.pairplot(df,vars=('Price','Age','Odometer','Mileage','EngineCC','PowerBhp'), height=2.5)
# plt.show()

##%%
corrl = (df[['Price','Age','Odometer','Mileage','EngineCC','PowerBhp']]).corr()
corrl

##%%

#相関係数の計算
corrl.style.background_gradient(cmap='coolwalm')

##%%

#相関係数の統計学的優位性の判定
np.where((df[['Price', 'Age', 'Odometer', 'Mileage', 'EngineCC', 'PowerBhp']]).corr() > 0.6, 'Yes', 'No')

##%%
#ダミー変数の導入

Location_dummy = pd.get_dummies(df.Location,prefix='Location',drop_first=True)
FuelType_dummy = pd.get_dummies(df.FuelType,prefix='FuelType',drop_first=True)
Transmission_dummy = pd.get_dummies(df.Transmission,prefix='Transmission',drop_first=True)
OwnerType_dummy = pd.get_dummies(df.OwnerType,prefix='OwnerType',drop_first=True)
combine_all_dummy = pd.concat([df, Location_dummy, FuelType_dummy, Transmission_dummy, OwnerType_dummy],
    axis=1)
combine_all_dummy.head()
combine_all_dummy.columns

#使わない？カラムをドロップする。
clean_df = combine_all_dummy.drop(
    columns=['Make','Location','FuelType','Transmission','OwnerType'])
clean_df.columns



