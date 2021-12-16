import winsound
import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

#get system time
t_sys = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

#plot-function

def plot_acc(log,t_sys):
    plt.plot(log['accuracy'])
    plt.plot(log['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./model_acc_' + t_sys + '.jpg')
    plt.show()

def plot_loss(log,t_sys):
    plt.plot(log['loss'])
    plt.plot(log['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('./model_loss_' + t_sys + '.jpg')
    plt.show()

# Part 1 - Data Preprocessing

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('./train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('./test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layershui
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Step 6 - Dropout
#cnn.add(tf.keras.layers.Dropout(0.3))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 100)

#plot
data_log = cnn.history.history
np.save('./model_log_' + t_sys,data_log)
plot_acc(data_log,t_sys)
plot_loss(data_log,t_sys)

#make sound
#winsound.Beep(400,1000)

# Part 4 - Making a single prediction

#import numpy as np
#from keras.preprocessing import image
#test_image = image.load_img('dataset/single_prediction/yes_or_no3.jpg', target_size = (64, 64))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis = 0)
#result = cnn.predict(test_image)
#training_set.class_indices
#if result[0][0] == 1:
#    prediction = 'yes'
#else:
#    prediction = 'no'
#print(prediction)