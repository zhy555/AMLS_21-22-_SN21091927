import os
import cv2 as cv
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from skimage.io import imread
from skimage.feature import greycomatrix, greycoprops
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')

feature_data = pd.read_csv("feature.csv")
X = feature_data[['Contrast','Energy','Homogeneity','Correlation','Dissimilarity','ASM','Area','Perimeter','Epsilon','IsConvex']]
Y = feature_data[['Class']]

def Random_Forest(X,Y,oob_score):
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y)

    forest = RandomForestClassifier(n_estimators=90,max_depth=90,min_samples_split = 3,oob_score = True)
    forest.fit(xTrain,yTrain)
    yPred = forest.predict(xTest)
    yTest = yTest.values.flatten()

    right = 0
    wrong = 0
    for index in range(len(yTest)):
        if (yTest[index] == 3 and yPred[index] == 3) or (yTest[index] != 3 and yPred[index] != 3):
            right += 1
        else:
            wrong += 1
    return (right/(right+wrong))

epoch = 100
paras = [0,1]
lst_res = []

for para in paras:
    res = 0
    for epoch_time in range(epoch):
        res += Random_Forest(X, Y,para)
        print(epoch_time)
    res = res / epoch
    lst_res.append(res)
print(lst_res)
