import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

houseprice = pandas.read_csv('./regression_data.csv')

houseprice = houseprice[["Price (Older)", 'Price (New)']]

X = houseprice[['Price (Older)']]
Y = houseprice[['Price (New)']]

xTrain, xTest, yTrain, yTest = train_test_split(X, Y)

def linearRegrPredict(xTrain, yTrain, xTest):
    # Create linear regression object
    regr = LinearRegression()
    # Train the model using the training sets
    regr.fit(xTrain, yTrain)
    # Make predictions using the testing set
    y_pred = regr.predict(xTest)
    # print("Accuracy Score:",regr.score(xTest,yTest))
    return y_pred


y_pred = linearRegrPredict(xTrain, yTrain, xTest)

# Plot testing set predictions
plt.scatter(xTest, yTest)
plt.plot(xTest, y_pred, 'r-')
plt.show()

xTrain1 = np.array(xTrain.values).flatten()
xTest1 = np.array(xTest.values).flatten()
yTrain1 = np.array(yTrain.values).flatten()
yTest1 = np.array(yTest.values).flatten()

def paramEstimates(xTrain, yTrain):
    beta = np.sum(np.multiply(xTrain, (np.add(yTrain, -np.mean(yTrain))))) / np.sum(
        np.multiply(xTrain, (np.add(xTrain, - np.mean(xTrain)))))

    alpha = np.mean(yTrain) - beta * np.mean(xTrain)

    return alpha, beta


def linearRegrNEWPredict(xTrain, yTrain, xTest):
    alpha, beta = paramEstimates(xTrain, yTrain)
    print(alpha)
    print(beta)
    y_pred1 = np.add(alpha, np.multiply(beta, xTest))

    return y_pred1


y_pred1 = linearRegrNEWPredict(xTrain1, yTrain1, xTest1)

# Plot testing set predictions
plt.scatter(xTest, yTest)
plt.plot(xTest1, y_pred1, 'r-')
plt.show()