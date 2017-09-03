import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #import and manage datasets

# import datasets
dataset = pd.read_csv('./Data/Churn_Modelling.csv')
X = dataset.iloc[ :, 3 : 13].values
y = dataset.iloc[ :, 13].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X1 = LabelEncoder()
X[ :, 1] = labelEncoder_X1.fit_transform(X[ : , 1])
labelEncoder_X2 = LabelEncoder()
X[ :, 2] = labelEncoder_X2.fit_transform(X[ : , 2])
# Create dummy variables
oneHotEncoder = OneHotEncoder(categorical_features = [1])
X = oneHotEncoder.fit_transform(X).toarray()
# to reduce the number of affecting dummy variables and get out of dummy variav=ble trap
X = X[:, 1:] 

#Splitting the dataset into test and training set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Imprt keras Library nd packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialise the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit he ANN to Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predict the Test set results
y_pred = classifier.predict(X_test)
y+pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)