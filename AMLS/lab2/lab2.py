import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pandas.read_csv('./multi_regr_data.csv')

X = dataset[list(dataset.columns)[:-1]]
Y = dataset[list(dataset.columns)[-1]]
xtrain,xtest,ytrain,ytest = train_test_split(X,Y)


def multilinearRegrPredict(xtrain, ytrain, xtest):
    # Create linear regression object
    reg = LinearRegression()
    # Train the model using the training sets
    reg.fit(xtrain, ytrain)

    print(reg.coef_)
    # Make predictions using the testing set
    y_pred = reg.predict(xtest)
    # See how good it works in test data,
    # we print out one of the true target and its estimate
    print('For the true target: ', list(ytest)[-1])
    print('We predict as: ', list(y_pred)[-1])  # print out the
    print("Overall Accuracy Score from library implementation:", reg.score(xtest,ytest))  # .score(Predicted value, Y axis of Test data) methods returns the Accuracy Score or how much percentage the predicted value and the actual value matches

    return y_pred

y_pred = multilinearRegrPredict(xtrain, ytrain, xtest)