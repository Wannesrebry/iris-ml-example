import pandas as pd
import io
import requests
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/iris.csv", na_values=['NA', '?']
)
# Convert data :
X = df[['sepal_l', 'sepal_w', 'petal_l', 'petal_w']]
dummies = pd.get_dummies(df['species'])
species = dummies.columns
Y = dummies.values
print(X)
print(species)
print("shape of Y: ", Y.shape[1])
print("shape of X: ", X.shape[1])
# Build neural network :

model = Sequential()
model.add(Dense(18, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(18, activation='relu'))
# output layer
model.add(Dense(Y.shape[1], activation='softmax'))
# compile:
model.compile(loss="categorical_crossentropy", optimizer='adam')
model.fit(X, Y, epochs=100, batch_size=50, verbose=2)
# predict data:
pred = model.predict(X)
print("Predicted shape is {}".format(pred.shape))

np.set_printoptions(suppress=True)
print(Y[0:10])
predict_classes = np.argmax(pred, axis=1)
expected_classes = np.argmax(Y, axis=1)
print(f"Predictions: {predict_classes}")
print(f"Expected: {expected_classes}")
print(species[predict_classes[1:10]])
correct = accuracy_score(expected_classes, predict_classes)
print(f"Accuracy: {correct}")
