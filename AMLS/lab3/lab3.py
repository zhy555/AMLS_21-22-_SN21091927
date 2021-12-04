import matplotlib.pyplot as plt
import numpy as np
from pandas import *
import pandas as pd
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

boston = load_boston()
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df['Price'] = boston.target
newX = boston_df.drop('Price', axis=1)
newY = boston_df['Price']
boston_df.head()

X_train, X_test, y_train, y_test = train_test_split(newX, newY, test_size=0.3, random_state=3)
print('train set: {}  | test set: {}'.format(round(len(y_train) / (len(newX) * 1.0), 2),
                                             round(len(y_test) / (len(newX) * 1.0), 2)))


def ridgeRegr(X_train, y_train, X_test):
    # Create linear regression object with a ridge coefficient 0.1
    ridge_regr_model = Ridge(alpha=0.1, fit_intercept=True)
    ridge_regr_model.fit(X_train, y_train)  # Fit Ridge regression model

    Y_pred = ridge_regr_model.predict(X_test)
    # print (Y_pred)
    return Y_pred

Y_pred = ridgeRegr(X_train, y_train, X_test)

plt.scatter(y_test, Y_pred)
plt.xlabel("Actual prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()  # Ideally, the scatter plot should create a linear line. Since the model does not fit 100%, the scatter plot is not creating a linear line.

r_alphas = 10**np.linspace(10,-2,100)*0.5
print(r_alphas)

def ridgeRegrCVPredict(X_train, y_train, r_alphas, X_test):
    ridgecv = RidgeCV(alphas=r_alphas, fit_intercept=True)
    ridgecv.fit(X_train, y_train)
    print('Best alpha value: ' + str(ridgecv.alpha_))

    Y_pred_cv = ridgecv.predict(X_test)
    # alternatively you could:
    # ridge = Ridge(alpha = ridgecv.alpha_)
    # ridge.fit(X_train, y_train)
    # Y_pred_cv=ridge.predict(X_test)
    return Y_pred_cv

Y_pred_cv =  ridgeRegrCVPredict(X_train, y_train, r_alphas,X_test)
mse_cv=mean_squared_error(y_test,Y_pred_cv)
print('Mean Squared Error (MSE) on test set (built-in cross-validation): '+str(mse_cv))

plt.scatter(y_test, Y_pred_cv)
plt.xlabel("Actual prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()