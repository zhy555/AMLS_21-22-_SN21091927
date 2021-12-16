import csv
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split

def createfolder(path):
    isExists = os.path.exists(path)
    if isExists:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)
    isExists = os.path.exists(path)

with open('./label.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
data = np.array(rows)
class1 = 'meningioma_tumor'
class2 = 'glioma_tumor'
class3 = 'pituitary_tumor'
class4 = 'no_tumor'
class_output = [[],[],[],[]]

for item in data:
    if item[1] == 'label':
        continue
    if item[1] == class1:
        class_output[0].append(item[0])
    elif item[1] == class2:
        class_output[1].append(item[0])
    elif item[1] == class3:
        class_output[2].append(item[0])
    elif item[1] == class4:
        class_output[3].append(item[0])
    else:
        print('error in class_output',item)

createfolder('./train/' + class1)
createfolder('./test/' + class1)
createfolder('./train/' + class2)
createfolder('./test/' + class2)
createfolder('./train/' + class3)
createfolder('./test/' + class3)
createfolder('./train/' + class4)
createfolder('./test/' + class4)

class1_Train, class1_Test = train_test_split(class_output[0])
class2_Train, class2_Test = train_test_split(class_output[1])
class3_Train, class3_Test = train_test_split(class_output[2])
class4_Train, class4_Test = train_test_split(class_output[3])


for item in class_output[0] + class_output[1] + class_output[2] + class_output[3]:
    source_path = './image/' + item
    if item in class_output[0]:
        if item in class1_Train:
            des_path = './train/' + class1 + '/' + item
        else:
            des_path = './test/' + class1 + '/' + item
    elif item in class_output[1]:
        if item in class2_Train:
            des_path = './train/' + class2 + '/'+ item
        else:
            des_path = './test/' + class2 + '/'+ item
    elif item in class_output[2]:
        if item in class3_Train:
            des_path = './train/' + class3 + '/'+ item
        else:
            des_path = './test/' + class3 + '/'+ item
    elif item in class_output[3]:
        if item in class4_Train:
            des_path = './train/' + class4 + '/'+ item
        else:
            des_path = './test/' + class4 + '/'+ item
    shutil.copy(source_path,des_path)

print('Train Test split Over!')