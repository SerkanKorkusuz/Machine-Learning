__machine_teacher__ = "serkan korkusuz"

import numpy as np
import pandas as pd
from matplotlib import pyplot as plot
from sklearn.cluster import KMeans

att1 = np.random.normal(25, 5, 1000)
label1 = np.random.normal(25, 5, 1000)

att2 = np.random.normal(55, 5, 1000)
label2 = np.random.normal(60, 5, 1000)

att3 = np.random.normal(55, 5, 1000)
label3 = np.random.normal(15, 5, 1000)

att = np.concatenate((att1, att2, att3), axis = 0)
label = np.concatenate((label1, label2, label3), axis = 0)

dictionary = {"att" : att, "label" : label}
#print(dictionary)

myData = pd.DataFrame(dictionary)
#print(myData)
#print(myData.info())
#print(myData.describe())

plot.scatter(att1, label1, c = "black")
plot.scatter(att2, label2, c = "black")
plot.scatter(att3, label3, c = "black")
plot.show()


#to be continued
