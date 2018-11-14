__author__ = "serkan korkusuz"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.tree import DecisionTreeRegressor

source_url = ("https://github.com/SerkanKorkusuz/Machine-Learning/blob/master/decision-tree-regression-dataset.csv")
myData = pd.read_csv(source_url, header = None, prefix = "V", sep = ";")
print(myData)

att = myData.V0.values.reshape(-1,1)
label = myData.V1.values.reshape(-1,1)
print(att)
print(label)

my_tree_model = DecisionTreeRegressor()
my_tree_model.fit(att, label)

print(my_tree_model.predict([[6]]))

attX = np.arange(min(att), max(att), 0.01).reshape(-1,1)
labelY = my_tree_model.predict(attX).reshape(-1,1)
plot.scatter(att, label)
plot.plot(attX, labelY, color = "orange")
plot.xlabel("Tribune Sector")
plot.ylabel("Ticket Price")
plot.show()
