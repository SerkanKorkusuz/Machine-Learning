__author__ = "serkan korkusuz"

import sys, urllib, scipy, pylab
import numpy as np
from urllib.request import urlopen
import matplotlib.pyplot as plot
from itertools import groupby
from collections import OrderedDict #Çok ilginç bir şekilde bunu yazmayınca "label types" listesini [R,M] yerine [M,R] şeklinde veriyor!!!
import scipy.stats as stats

source_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
my_data = urlopen(source_url)

attList = []

for line in my_data:
    row = line.strip().split(b",") #Eğer b yazılmazsa TypeError veriyor. str den binary e geçtik bu şekilde
    attList.append(row)

row_length = len(attList)
col_length = len(attList[0])
#print(row_length)
#print(col_length)

print("Col\t\t" + "Number\t\t" + "String\t\t" + "Other\t\t") 
count = [0] * 3
col_count = 0

for j in range(col_length):
    for i in range(row_length):
        try:
            a = float(attList[i][j])
            if isinstance (a, float):
                count[0] += 1
        except ValueError:
            if len (attList[i][j])>0:
                count[1] += 1
            else:
                count[2] += 1
    print(str(col_count) + "\t\t" + str(count[0]) + "\t\t" + str(count[1]) + "\t\t" + str(count[2]) + "\n")
    count = [0] * 3
    col_count +=1
#print(count)

col_for_calc = 3
col3_data = []
for row in attList:
    col3_data.append(float(row[col_for_calc]))
myArray = np.array(col3_data)
mean_col3 = myArray.mean()
s_dev_col3 = myArray.std()
#print(myArray)
print("For column 3:\n" + "Mean: " + str(mean_col3) + "\nStandart Deviation: " + str(s_dev_col3))

perc_eq = 4
percentile_col3 = []
for k in range(perc_eq+1):
    percentile_col3.append(np.percentile(myArray, k*100/perc_eq))
print("Boundaries for 4 Equal Percentiles: "+ str(percentile_col3))

perc_eq = 10
percentile_col3 = []
for k in range(perc_eq+1):
    percentile_col3.append(np.percentile(myArray, k*100/perc_eq))
print("Boundaries for 10 Equal Percentiles: "+ str(percentile_col3))

col_for_calc = 60
data_col60 = []
for row in attList:
    data_col60.append(row[col_for_calc].decode('ascii'))
label_types = list(set(data_col60))
copy = label_types
print("Categorical Levels and Their Iterations:\n" + str(copy))
print([len(list(group)) for key, group in groupby(data_col60)])

scipy.stats.probplot (col3_data, dist = "norm", plot = pylab)
"""scipy.stats.probplot(x, sparams=(), dist='norm', fit=True, plot=None) Generates a probability plot of sample data
against the quantiles of a specified theoretical distribution (the normal distribution by default). probplot optionally calculates a best-fit line for the data and
plots the results using Matplotlib or a given plot function."""
pylab.show()






        


