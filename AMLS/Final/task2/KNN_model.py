import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import itertools

feature_data = pd.read_csv("feature.csv")
X = feature_data[['Contrast','Energy','Homogeneity','Correlation','Dissimilarity','ASM','Area','Perimeter','Epsilon','IsConvex']]
Y = feature_data[['Class']]


def Knn_model(X,Y):
    X = X.values.tolist()
    Y = Y.values.flatten()

    xTrain, xTest, yTrain, yTest = train_test_split(X, Y)
    knn = KNeighborsClassifier()
    knn.n_neighbors = 1
    knn.p = 1
    knn.leaf_size = 6
    knn.fit(xTrain,yTrain)
    yPred = knn.predict(xTest)

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
    res += Knn_model(X, Y)
    print(epoch_time)
res = res / epoch
print(res)