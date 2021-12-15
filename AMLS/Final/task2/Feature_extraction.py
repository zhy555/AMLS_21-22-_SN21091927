from skimage.io import imread
from skimage.feature import greycomatrix, greycoprops
from sklearn import preprocessing
import numpy as np
import pandas as pd
import cv2 as cv
import os
import csv
import warnings
warnings.filterwarnings('ignore')

input_data_dir = './image'

pics = os.listdir(input_data_dir)
no_samples = range(len(pics))

no_cols = 11 # 10 features and 1 class
cols =np.asarray(['Contrast','Energy','Homogeneity','Correlation','Dissimilarity','ASM','Area','Perimeter','Epsilon','IsConvex','Class'])

label_class = {
  "meningioma_tumor": 0,
  "glioma_tumor": 1,
  'pituitary_tumor':2,
  "no_tumor": 3
}

features = np.ones([len(no_samples),no_cols])

# read the image label from label.csv
with open('./label.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
rows = rows[1:]
image_label = {}
for item in rows:
    image_label[item[0]] = label_class[item[1]]

i = 0
for pic in pics:
    img = imread(input_data_dir + '/' + pic)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    ret, thresh = cv.threshold(img, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, 1, 2)
    cnt = contours[0]

    area = cv.contourArea(cnt)
    perimeter = cv.arcLength(cnt, True)
    epsilon = 0.1 * cv.arcLength(cnt, True)
    k = cv.isContourConvex(cnt)

    S = preprocessing.MinMaxScaler((0, 11)).fit_transform(img).astype(int)

    g = greycomatrix(S, distances=[1], angles=[0], levels=256, symmetric=False, normed=False)

    contrast = greycoprops(g, 'contrast')
    energy = greycoprops(g, 'energy')
    homogeneity = greycoprops(g, 'homogeneity')
    correlation = greycoprops(g, 'correlation')
    dissimilarity = greycoprops(g, 'dissimilarity')
    ASM = greycoprops(g, 'ASM')

    f_arr = np.asarray(
        [contrast[0][0], energy[0][0], homogeneity[0][0], correlation[0][0], dissimilarity[0][0], ASM[0][0], area,
         perimeter, epsilon, k, image_label[pic]], dtype='object')
    features[i] = f_arr
    i += 1

df = pd.DataFrame(features, columns=cols)
df['Class'] = np.int64(df['Class'])
df.to_csv("feature.csv",index= False)
print("Feature extract over!")