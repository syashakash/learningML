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

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualising the Linear Regression resuts
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()
"""
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = np.reshape(len(X_grid), 1)
"""
# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'green')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

# Prdict a new result with Linear Regression
print(lin_reg.predict(6.5))

#Predict a new result with Polynomial Regression
print(lin_reg2.predict(poly_reg.fit_transform(6.5)))