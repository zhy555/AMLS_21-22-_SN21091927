import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pandas.read_csv('./multi_regr_data.csv')
print(dataset)