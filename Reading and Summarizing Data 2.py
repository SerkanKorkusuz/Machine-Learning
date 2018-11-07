__author__ = "serkan korkusuz"

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot

source_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
myData = pd.read_csv(source_url, header = None, prefix = "V")

myCorr = DataFrame(myData.corr())
#print(myCorr.iat[1,20])
plot.pcolor(myCorr)
plot.show()
