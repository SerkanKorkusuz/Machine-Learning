__machine_teacher__ = "serkan korkusuz"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from pandas import DataFrame

source_url = ("https://raw.githubusercontent.com/SerkanKorkusuz/Machine-Learning/master/naive-bayes-dataset.csv")
myData = pd.read_csv(source_url, header = 0)
myData.drop(["id", "Unnamed: 32"], axis = 1, inplace = True)
#print(myData.head())

M = myData[myData.diagnosis == "M"]
B = myData[myData.diagnosis == "B"]
#print(M.info())
#print(B.info())

plot.scatter(M.radius_mean, M.area_mean, color = "red", label = "Malign", alpha = 0.4)
plot.scatter(B.radius_mean, B.area_mean, color  = "blue", label = "Benign")
plot.xlabel("radius_mean")
plot.ylabel("area_mean")
plot.legend()
plot.show()

plot.scatter(M.radius_mean, M.texture_mean, color = "red", label = "Malign", alpha = 0.4)
plot.scatter(B.radius_mean, B.texture_mean, color  = "blue", label = "Benign")
plot.xlabel("radius_mean")
plot.ylabel("texture_mean")
plot.legend()
plot.show()

myData.diagnosis = [1 if each == "M" else 0 for each in myData.diagnosis]
label = myData.diagnosis.values
att = myData.drop(["diagnosis"], axis = 1)

att = (att - np.min(att)) / (np.max(att) - np.min(att))

att_train, att_test, label_train, label_test = train_test_split(att, label, test_size = 0.3, random_state = 42)

my_model = GaussianNB()
my_model.fit(att_train, label_train)

