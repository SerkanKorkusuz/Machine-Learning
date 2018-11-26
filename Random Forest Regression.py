__machine_teacher__ = "serkan korkusuz"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

source_url = ("C:\\Users\\casper\\Desktop\\Machine Learning\\Machine Learning in Python\\Udemy-DATAI\\Random forest Regression\\random-forest-regression-dataset.csv")
myData = pd.read_csv(source_url, header = None, prefix = "V", sep = ";")
print(myData)

att = myData.iloc[:, 0].values.reshape(-1,1)
label = myData.iloc[:, 1].values.reshape(-1, 1)

my_rf_model = RandomForestRegressor(n_estimators = 100, random_state = 42)
#n_estimator determines the numbers of trees
my_rf_model.fit(att, label)
print(my_rf_model.predict([[7.8]]))
