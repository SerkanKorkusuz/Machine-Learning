__machine_teacher__ = "serkan korkusuz"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

source_url = ("https://raw.githubusercontent.com/SerkanKorkusuz/Machine-Learning/master/KNN-data.csv")
myData = pd.read_csv(source_url, header = 0)
myData.drop(["id", "Unnamed: 32"], axis = 1, inplace = True)
#print(myData.head())

M = myData[myData.diagnosis == "M"]
B = myData[myData.diagnosis == "B"]
#print(M.info())
#print(B.info())

plot.scatter(M.radius_mean, M.area_mean, color = "red", label = "bad", alpha = 0.4)
plot.scatter(B.radius_mean, B.area_mean, color  = "blue", label = "good")
plot.xlabel("radius_mean")
plot.ylabel("area_mean")
plot.legend()
plot.show()

plot.scatter(M.radius_mean, M.texture_mean, color = "red", label = "bad", alpha = 0.4)
plot.scatter(B.radius_mean, B.texture_mean, color  = "blue", label = "good")
plot.xlabel("radius_mean")
plot.ylabel("texture_mean")
plot.legend()
plot.show()

myData.diagnosis = [1 if each == "M" else 0 for each in myData.diagnosis]
label = myData.diagnosis.values
att = myData.drop(["diagnosis"], axis = 1)

att = (att - np.min(att)) / (np.max(att) - np.min(att))

att_train, att_test, label_train, label_test = train_test_split(att, label, test_size = 0.3, random_state = 42)

my_model = KNeighborsClassifier(n_neighbors = 3)
my_model.fit(att_train, label_train)
label_predict = my_model.predict(att_test)
#print(label_predict)
print("{} KNN Score: {}".format(3, my_model.score(att_test, label_test)))

k_score_list = []
for each in range(1, 100):
    my_model2 = KNeighborsClassifier(n_neighbors = each)
    my_model2.fit(att_train, label_train)
    k_score_list.append(my_model2.score(att_test, label_test))

max_k_score = max(k_score_list)

for i in range(len(k_score_list)):
    if max_k_score == k_score_list[i]:
        max_k_value = i + 1
print(k_score_list)
print("The k value giving the best score is {}".format(max_k_value))

plot.plot(range(1,100), k_score_list)
plot.xlabel("k values")
plot.ylabel("k scores")
plot.show()
