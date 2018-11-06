__author__ = "serkan korkusuz"

import math
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot
from random import uniform
import scipy.stats as stats

source_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
myData = pd.read_csv (source_url, header=None, prefix="V")

print(myData)
print(myData.head())
print(myData.tail())
print(myData.describe())

for i in range(len(myData)):
    if myData.iat[i,60] == "M":
        row_color="red"
    else:
        row_color="blue"

    dataRow = myData.iloc[i, 0:60]
    dataRow.plot(color=row_color)
plot.xlabel("Attribute Index")
plot.ylabel(("Attribute Values"))
plot.show()

dataRow2 = myData.iloc[1, 0:60]
dataRow3 = myData.iloc[2, 0:60]
plot.scatter(dataRow2, dataRow3)
plot.xlabel("2nd Attribute")
plot.ylabel("3rd Attribute")
plot.show()

dataRow60 = myData.iloc[60, 0:60]
plot.scatter(dataRow2, dataRow60)
plot.xlabel("2nd Attribute")
plot.ylabel("60th Attribute")
plot.show()

label = []
for i in range(len(myData)):
    if myData.iat[i, 60] == "M":
        label.append(1.0)
    else:
        label.append(0.0)
att35 = myData.iloc[:, 36]
plot.scatter(att35, label)
plot.xlabel("att35")
plot.ylabel("label")
plot.show()

label = []
for i in range(len(myData)):
    if myData.iat[i, 60] == "M":
        label.append(1.0 + uniform(-0.1, 0.1))
    else:
        label.append(0.0 + uniform(-0.1, 0.1))
att35 = myData.iloc[:, 36]
plot.scatter(att35, label, alpha=0.3, s=120)
plot.xlabel("att35")
plot.ylabel("label")
plot.show()

corr2_3 = stats.pearsonr(myData.iloc[:,1],  myData.iloc[:,2])
corr2_21 = stats.pearsonr(myData.iloc[:,1],  myData.iloc[:,20])
print("Correlation between attribute 2 and attribute 3 is " + str(corr2_3))
print("Correlation between attribute 2 and attribute 21 is " + str(corr2_21))


