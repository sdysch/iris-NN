# === imports ===
import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data/iris.csv")
#print(data.describe())

# === plotting histograms ===
data_setosa     = data.loc[data["Species"] == "Iris-setosa"]
data_versicolor = data.loc[data["Species"] == "Iris-versicolor"]
data_virginica  = data.loc[data["Species"] == "Iris-virginica"]

fig, axs = plt.subplots(2, 2)

# sepal length
axs[0, 0].hist(data_setosa["SepalLengthCm"].to_numpy(),     bins = 10, histtype = "step", facecolor = "blue",   label = "Iris-setosa")
axs[0, 0].hist(data_versicolor["SepalLengthCm"].to_numpy(), bins = 10, histtype = "step", facecolor = "red",    label = "Iris-versicolor")
axs[0, 0].hist(data_virginica["SepalLengthCm"].to_numpy(),  bins = 10, histtype = "step", facecolor = "green",  label = "Iris-virginica")

axs[0, 0].set_xlabel("Sepal length [cm]")
axs[0, 0].set_ylabel("Events / bin")

# sepal width
axs[0, 1].hist(data_setosa["SepalWidthCm"].to_numpy(),     bins = 10, histtype = "step", facecolor = "blue",   label = "Iris-setosa")
axs[0, 1].hist(data_versicolor["SepalWidthCm"].to_numpy(), bins = 10, histtype = "step", facecolor = "red",    label = "Iris-versicolor")
axs[0, 1].hist(data_virginica["SepalWidthCm"].to_numpy(),  bins = 10, histtype = "step", facecolor = "green",  label = "Iris-virginica")

axs[0, 1].set_xlabel("Sepal width [cm]")
axs[0, 1].set_ylabel("Events / bin")

# petal length
axs[1, 0].hist(data_setosa["PetalLengthCm"].to_numpy(),     bins = 10, histtype = "step", facecolor = "blue",   label = "Iris-setosa")
axs[1, 0].hist(data_versicolor["PetalLengthCm"].to_numpy(), bins = 10, histtype = "step", facecolor = "red",    label = "Iris-versicolor")
axs[1, 0].hist(data_virginica["PetalLengthCm"].to_numpy(),  bins = 10, histtype = "step", facecolor = "green",  label = "Iris-virginica")

axs[1, 0].set_xlabel("Petal length [cm]")
axs[1, 0].set_ylabel("Events / bin")

# petal width
axs[1, 1].hist(data_setosa["PetalWidthCm"].to_numpy(),     bins = 10, histtype = "step", facecolor = "blue",   label = "Iris-setosa")
axs[1, 1].hist(data_versicolor["PetalWidthCm"].to_numpy(), bins = 10, histtype = "step", facecolor = "red",    label = "Iris-versicolor")
axs[1, 1].hist(data_virginica["PetalWidthCm"].to_numpy(),  bins = 10, histtype = "step", facecolor = "green",  label = "Iris-virginica")

axs[1, 1].set_xlabel("Petal width [cm]")
axs[1, 1].set_ylabel("Events / bin")

fig.tight_layout()
plt.legend(loc = "upper right")
#plt.show()

fig.savefig("hist.pdf")
fig.savefig("hist.png")

# === neural network ===

# === preprocessing ===
# convert each species into a unique label
data.loc[data["Species"] == "Iris-setosa", "Species"]     = 0
data.loc[data["Species"] == "Iris-versicolor", "Species"] = 1
data.loc[data["Species"] == "Iris-virginica", "Species"]  = 2

# random permutations
data = data.iloc[np.random.permutation(len(data))]

# convert into arrays of variables and normalize
from sklearn.preprocessing import normalize
X = data.iloc[:,1:5].values
y = data.iloc[:,5].values
X_normalized = normalize(X, axis = 0)

# split into training, testing
#fraction = 0.75
fraction = 0.8
totalLength = len(data)
trainLength = int(fraction * totalLength)
testLength  = int((1. - fraction) * totalLength)

X_train = X_normalized[:trainLength]
X_test  = X_normalized[trainLength:]
y_train = y[:trainLength]
y_test  = y[trainLength:]

# === define model ===
from keras.models import Sequential 
from keras.layers import Dense,Activation,Dropout 
from keras.layers.normalization import BatchNormalization 
from keras.utils import np_utils

"""
[0]--->[1 0 0]
[1]--->[0 1 0]
[2]--->[0 0 1]
"""

y_train = np_utils.to_categorical(y_train, num_classes = 3)
y_test  = np_utils.to_categorical(y_test,  num_classes = 3)

model = Sequential()
model.add(Dense(1000, input_dim = 4, activation = "relu"))
model.add(Dense(500,  activation = "relu"))
model.add(Dense(300,  activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(3, activation = "softmax"))
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["accuracy"])

model.summary()

model.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size = 20, epochs = 10, verbose = 1)

prediction = model.predict(X_test)
length = len(prediction)
y_label = np.argmax(y_test, axis = 1)
predict_label = np.argmax(prediction, axis = 1)

accuracy = np.sum(y_label == predict_label) / length
print(f"Accuracy of the dataset {accuracy}")
