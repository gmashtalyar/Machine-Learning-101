import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pandas as pd
import numpy as np


data = pd.read_csv("car.data")
print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
safety = le.fit_transform(list(data["safety"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
cls = le.fit_transform(list(data["class"]))
print(buying)

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, Y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

print(x_train, y_train)