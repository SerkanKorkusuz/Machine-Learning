__author__ = "serkan korkusuz"

from math import sqrt

label = [1.5, 2.1, 3.3, -4.7, -2.3, 0.75]
label_predict = [0.5, 1.5, 2.1, -2.2, 0.1, -0.5]

error = []
for i in range(len(label)):
    error.append(label[i]-label_predict[i])
print("Error: ", error)

squared_error = []
absolute_error = []
for j in error:
    squared_error.append(j*j)
    absolute_error.append(abs(j))
print("Squared Error: ", squared_error)
print("Absolute Error: ", absolute_error)

mean_squared_error = sum(squared_error) / len(squared_error)
print("Mean Squared Error: ", mean_squared_error)

mean_absolute_error = sum(absolute_error) / len(absolute_error)
print("Mean Absolute Error: ", mean_absolute_error)

root_mean_squared_error = sqrt(mean_squared_error)
print("Root Mean Squared Error: ", root_mean_squared_error)
