__machine_teacher__ = "serkan korkusuz"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

source_url = ("https://raw.githubusercontent.com/SerkanKorkusuz/Machine-Learning/master/KNN-data.csv")
myData = pd.read_csv(source_url, header = 0)
myData.drop(["id", "Unnamed: 32"], axis = 1, inplace = True)
#print(myData.head())

M = myData[myData.diagnosis == "M"]
B = myData[myData.diagnosis == "B"]
#print(M.info())
#print(B.info())

plot.scatter(M.radius_mean, M.area_mean, color = "red", label = "bad", alpha = 0.4)
plot.scatter(B.radius_mean, B.area_mean, color  = "blue", label = "good")

#to be continued...
