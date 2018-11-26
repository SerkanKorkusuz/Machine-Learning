__machine_teacher__ = "serkan korkusuz"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

source_url = ("C:\\Users\\casper\\Desktop\\Machine Learning\\Machine Learning in Python\\Udemy-DATAI\\Random forest Regression\\random-forest-regression-dataset.csv")
myData = pd.read_csv(source_url, header = None, prefix = "V", sep = ";")
print(myData)
