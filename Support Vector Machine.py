__machine_teacher__ = "serkan korkusuz"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from pandas import DataFrame

source_url = ("https://raw.githubusercontent.com/SerkanKorkusuz/Machine-Learning/master/support%20vector%20machine%20dataset.csv")
myData = pd.read_csv(source_url, header = 0)
myData.drop(["id", "Unnamed: 32"], axis = 1, inplace = True)
#print(myData.head())
