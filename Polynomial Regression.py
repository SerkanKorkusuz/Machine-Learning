import numpy as np
import matplotlib.pyplot as plot

att = []
label = []

np.random.seed(2)
for i in range(50):
    att.append(list(np.random.normal(3.0, 1.0, 1000)))
    label.append(list(np.random.normal(50, 10, 1000) / att[i]))

print(att)
print(label)
plot.scatter(att, label)
plot.show()

attArr = np.array(att)[0,:]
labelArr = np.array(label)[0,:]

polyRegress = np.poly1d(np.polyfit(attArr, labelArr, 4))

attP = np.linspace(0, 7, 100)
plot.scatter(attArr, labelArr)
plot.plot(attP, polyRegress(attP), c="r")
plot.show()

r2 = r2_score(labelArr, polyRegress(attArr))
print(r2)





