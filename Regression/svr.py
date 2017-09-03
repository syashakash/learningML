# Polynomial Regression
import pandas as pd #import and manage datasets
import matplotlib.pyplot as plt
import numpy as np
# import datasets
dataset = pd.read_csv('./Data/Position_Salaries.csv')
X = dataset.iloc[ :, 1 : 2].values
y = dataset.iloc[ : , 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() 
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# Create your regressor
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#Predict a new result with Regression
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the SVR results
plt.scatter(X, y, color = 'green')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

