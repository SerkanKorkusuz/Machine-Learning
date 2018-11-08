import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from scipy import stats

source = ("C:\\Users\\casper\\Desktop\\Python\\Projects\\Machine Learning in Python\\linear-regression-dataset.csv")
myData = pd.read_csv(source, header=0, prefix="V", sep=";")
print(myData)

att = []
label = []

for i in range(len(myData)):
    att.append(myData.iloc[i, 0])
    label.append(myData.iloc[i, 1])

slope, intercept, r_value, pvalue, std_err = stats.linregress(att, label)
print("Square of r_value is "+ str(r_value**2))

def myPredict(feature):
    labelList = []
    for j in range(len(feature)):
        labelList.append(feature[j] * slope + intercept)
    return labelList

predictLabel = myPredict(att)
plot.scatter(att, label)
plot.plot(att, predictLabel, c="r")
plot.xlabel("Experience")
plot.ylabel("Salary")
plot.title("Thanks to scipy")
plot.show()

#another way to form a model

from sklearn.linear_model import LinearRegression

linear_reg_model = LinearRegression()

attArr = myData.deneyim.values.reshape(-1,1)
labelArr = myData.maas.values.reshape(-1,1)
#print(attArr.shape())

linear_reg_model.fit(attArr, labelArr)
print("y=mx+n =>   n=" + str(linear_reg_model.predict([[0]])))
print("y=mx+n =>   n=" + str(linear_reg_model.intercept_))
print("y=mx+n =>   m=" + str(linear_reg_model.coef_))

newAtt = np.linspace(0, 15, 15).reshape(-1,1)
newLabel = linear_reg_model.predict(newAtt)
plot.scatter(attArr, labelArr)
plot.plot(newAtt, newLabel, c="r")
plot.xlabel("Experience")
plot.ylabel("Salary")
plot.title("Thanks to skicit-learn")
plot.show()





