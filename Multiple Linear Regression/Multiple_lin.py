
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
dataset = pd.read_csv('your dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X[: , 3] = labelencoder_X.fit_transform(X[: , 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

X = X[: , 1:]

from sklearn.model_selection import train_test_split
X_train , X_test , y_train, y_test = train_test_split(X , y , test_size = 0.2)

from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(X_train , y_train)
y_pred = regress.predict(X_test)
#plt.plot(X_train, y_train)'''

import statsmodels.api as sm
X = np.append(arr = np.ones((50 ,1 )).astype(int) , values = X  , axis = 1)

X_opt = X[: , [0, 1, 2, 3, 4, 5]]
regress_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regress_OLS.summary()

X_opt = X[: , [0, 1, 3, 4, 5]]
regress_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regress_OLS.summary()

X_opt = X[: , [0, 3, 4, 5]]
regress_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regress_OLS.summary()

X_opt = X[: , [0, 3, 5]]
regress_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regress_OLS.summary()

'''X_train , X_test , y_train, y_test = train_test_split(X_opt , y , test_size = 0.2)
from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(X_train , y_train)
y_pred = regress.predict(X_test)'''
