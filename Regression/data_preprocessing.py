import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #import and manage datasets

# import datasets
dataset = pd.read_csv('./Data/Data.csv')
X = dataset.iloc[ :, : -1].values
y = dataset.iloc[ : , 3].values
#print(X)

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[ : , 1 : 3])
X[ : , 1 : 3] = imputer.transform(X[ : , 1 : 3])
"""
for i in X:
    print(i)
"""
#print(y)
#Encoding data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[ : , 0] = labelEncoder_X.fit_transform(X[ : , 0])
#print(X[ : , 0])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()
#print(X)
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)
# segret=gate data into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
