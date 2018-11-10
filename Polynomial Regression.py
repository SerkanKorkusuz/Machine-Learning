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


