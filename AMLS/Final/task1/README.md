# UCL_AMLS
# SN 21091927

This is the Readme file for the taskA, here is the instructions about how to run the code.

KNN model:

libraries: skimage, sklearn, numpy, pandas, opencv-python, os, csv, warnings, matplotlib, itertools

1. run the Feature_extraction.py file. It will output a csv file feature.csv. You do not need to do this if there is a existing feature.csv
2. run the KNN_model.py. 


RF model:

libraries: skimage, sklearn, numpy, pandas, opencv-python, os, csv, warnings, matplotlib, itertools

1. run the Feature_extraction.py file. It will output a csv file feature.csv. You do not need to do this if there is a existing feature.csv
2. run the RF_model.py. 


CNN model:

libraries: csv, numpy, os, shutil, sklearn, datetimes, tensorflow-gpu, matplotlib, keras, keras.processing

P.S. the version of tensorflow-gpu is 2.6.0, the version of keras is 2.6.0, and the CUDA version is 11.2, while the GPU is 3060ti and Nivida driver is in version 496.76, the cuDNN version is 8.1.0, the Python version is 3.9.7. This is a workable environment The code may not work if there is a version mismatch among Nivida driver, tensorflow-gpu, CUDA and cuDNN. I spend a lot of time solving the environment issue, and if there is a environment problem when you run the code, please contact me.

1. run the image_info_read.py file. It will output two folders called train and test. You do not need to do this if there are existing folders
2. run the CNN_model.py. 

<image> folder is the MRI dataset which is provided by the teacher, no change has been made
<label.csv> is the MRI dataset labels which is provided by the teacher, no change has been made
<Record> folder records the procedure I adjust the parameters of CNN models
<test> and <train> folder are produced by the Feature_extraction.py, which contains testset and trainset which is splitted from <image>
<CNN_model.py> is the code of CNN model
<Feature_extraction.py> is the code to extract features from <iamge>
<feature.csv> is the extracted data of MRI, produced by <Feature_extraction.py>
<image_info_read.py> is used to split the image in <image> into <train> and <test>
<KNN_model.py> is the code of KNN model
<RF_model.py> is the code of RF model
  
  
If there is any problem running my code, please contact me without hesitation
