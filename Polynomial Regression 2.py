import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

source_url = ("C:\\Users\\casper\\Desktop\\Machine Learning\\Machine Learning in Python\\Udemy-DATAI\\Polynomial Regression\\polynomial-regression.csv")
myData = pd.read_csv(source_url, header = 0, prefix = "V", sep = ";")
print(myData)

att = myData.araba_fiyat.values.reshape(-1,1)
label = myData.araba_max_hiz.values.reshape(-1,1)
print(att)
print(label)

plot.scatter(att, label)
plot.xlabel("Price of Car")
plot.ylabel("Velocity")
plot.show()

#trying to execute linear regression
linear_model = LinearRegression()
my_model = linear_model.fit(att, label)

#print(my_model.predict([[0]]))
#print(my_model.intercept_)
#print(my_model.coef_)
#print(my_model.predict([[15000]]))

plot.scatter(att,label)
attX=np.linspace(0, 3000, 30).reshape(-1,1)
plot.plot(attX, my_model.predict((attX)))
plot.xlabel("Price of Car")
plot.ylabel("Velocity")
plot.show()
#noticed that it is not a good approach to implement linear regression method on this problem

#So trying Polynomialfatures seems better to fit on the graph.
polynomial_model = PolynomialFeatures(degree=4)
x_polynomial = polynomial_model.fit_transform(att)
linear_model2 = LinearRegression()
linear_model2.fit(x_polynomial, label)

plot.scatter(att,label)
labelY = linear_model2.predict(x_polynomial)
plot.plot(att, labelY)
plot.xlabel("Price of Car")
plot.ylabel("Velocity")
plot.show()

#with numpy, I think it gives better result
attForNumpy = myData.araba_fiyat.values
labelForNumpy = myData.araba_max_hiz.values

polyRegress =np.poly1d(np.polyfit(attForNumpy, labelForNumpy, 4))

plot.scatter(att,label)
plot.plot(attX, polyRegress(attX))
plot.xlabel("Price of Car")
plot.ylabel("Velocity")
plot.show()

print(polyRegress(10000))


