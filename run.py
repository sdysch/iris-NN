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
plt.show()

fig.savefig("hist.pdf")
fig.savefig("hist.png")
