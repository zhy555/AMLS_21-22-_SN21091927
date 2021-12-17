import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

feature_data = pd.read_csv("feature.csv")
X = feature_data[['Contrast','Energy','Homogeneity','Correlation','Dissimilarity','ASM','Area','Perimeter','Epsilon','IsConvex']]
Y = feature_data[['Class']]

def Random_Forest(X,Y):
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y)

    forest = RandomForestClassifier(n_estimators=90,max_depth=90,min_samples_split = 3,oob_score = True)
    forest.fit(xTrain,yTrain)
    yPred = forest.predict(xTest)
    yTest = yTest.values.flatten()

    right = 0
    wrong = 0
    for index in range(len(yTest)):
        if yTest[index] == yPred[index]:
            right += 1
        else:
            wrong += 1

    def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
        plt.imshow(cm, interpolation='nearest', cmap='YlGnBu')
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    #cm = confusion_matrix(y_true = yTest, y_pred = yPred)
    #cm_plot_labels = ["meningioma_tumor","glioma_tumor",'pituitary_tumor',"no_tumor"]
    #plot_confusion_matrix(cm, cm_plot_labels, title='Confusion matrix')

    return (right/(right+wrong))

epoch = 100
res = 0

for epoch_time in range(epoch):
    res += Random_Forest(X, Y)
    print(epoch_time)
res = res / epoch