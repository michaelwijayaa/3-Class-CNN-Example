import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout 
from tflearn.layers.conv import conv_2d,max_pool_2d 
from tflearn.layers.estimator import regression 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import os 
from random import shuffle 
from google_images_download import google_images_download
from PIL import Image
from tqdm import tqdm
import matplotlib.image as mpimg

#=====================PARAMETERS=========================#
IMG_SIZE = 100
learning_rate = 4e-4
N_EPOCH = 10
#==================================Get Label from Image==================================#
def label_img(img):

    word_label = img.split('_')[0]
    # print(word_label)
    if word_label == "car":
        return [1,0,0]

    elif word_label == "bike":
        return [0,1,0]

    elif word_label == "truck":
        return[0,0,1]

#==================================Create Training Set==================================#
def create_train_set():
    
    training_data = []
    TRAIN_DIR_bike = "E:/mike/SKRIPSI/belajarCNN/cnn_custom_dataset/bike"
    TRAIN_DIR_car = "E:/mike/SKRIPSI/belajarCNN/cnn_custom_dataset/car"
    TRAIN_DIR_truck = "E:/mike/SKRIPSI/belajarCNN/cnn_custom_dataset/truck"
    train_url = [TRAIN_DIR_bike, TRAIN_DIR_car, TRAIN_DIR_truck]
    for i in train_url:
        for image in tqdm(os.listdir(i)):
            label = label_img(image)
            # print(label)
            path = os.path.join(i,image)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data.append([ np.array(img), np.array(label)])
            # print(training_data)
            shuffle(training_data)
            shuffle(training_data)
            shuffle(training_data)
            shuffle(training_data)
            np.save('training_data.npy', training_data)
    return training_data

#==================================Create Test Set==================================#
def create_test_set():
    testing_data = []
    TEST_DIR_bike = "E:/mike/SKRIPSI/belajarCNN/cnn_custom_dataset/biketest"
    TEST_DIR_car = "E:/mike/SKRIPSI/belajarCNN/cnn_custom_dataset/cartest"
    TEST_DIR_truck = "E:/mike/SKRIPSI/belajarCNN/cnn_custom_dataset/trucktest"
    test_url = [TEST_DIR_bike, TEST_DIR_car, TEST_DIR_truck]
    for i in test_url:
        for image in tqdm(os.listdir(i)):
            label = label_img(image)
            # print(label)
            path = os.path.join(i,image)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            testing_data.append([ np.array(img),np.array(label)])
            # print(testing_data)
            shuffle(testing_data)
            shuffle(testing_data)
            shuffle(testing_data)
            shuffle(testing_data)
            np.save('testing_data.npy',testing_data)
    return testing_data

MODEL_NAME = "3class-{}-{}-{}.model".format(learning_rate,'6-conv-basic',N_EPOCH)

#=====================CREATE OR LOAD TRAINING DATA=========================#
# train_data = create_train_set()
# test_data = create_test_set()
train_data = np.load('training_data.npy')
test_data = np.load('testing_data.npy')

#=====================CONV NET HERE=========================#
convnet = input_data(shape = [None,IMG_SIZE,IMG_SIZE,1], name='inputs')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# convnet = fully_connected(convnet,512,activation='relu')
convnet = fully_connected(convnet, 1024, activation='relu')
# convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet,3,activation='softmax')
convnet = regression(convnet,optimizer='adam', name='targets', learning_rate=learning_rate, loss='binary_crossentropy', metric = 'accuracy')

model = tflearn.DNN(convnet,tensorboard_dir='log')

X = np.array([ i[0] for i in train_data ]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = [ i[1] for i in train_data ]

test_X = np.array([ i[0] for i in test_data ]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [ i[1] for i in test_data ]
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model loaded!')
    # model.summary()
else:
    model.fit({'inputs':X},{'targets':y}, n_epoch=N_EPOCH, validation_set=({'inputs':test_X}, {'targets':test_y}), show_metric=True, snapshot_step=100, run_id=MODEL_NAME, batch_size=10)
    model.save(MODEL_NAME)

# matplotlib inline
fig = plt.figure(figsize=(10,6))

for num, image in enumerate(test_data[16:32]):
    img_num = image[1]
    img_data = image[0]

    y = fig.add_subplot(4,4,num+1)
    orig = img_data
    image = img_data.reshape(IMG_SIZE,IMG_SIZE,1)

    model_out = model.predict([image])[0]
    print(model_out)
    # print(image)
    # y_plot = fig.add_subplot(3,6,num+1)
    # model_out = model.predict([image[0]])[0][0]
    # print(model_out) 
    # if model_out == 1:
    #     label = "FOREST_FIRE"
    # else:
    #     label="NATURAL_VEG" 
    if np.argmax(model_out) == 0: 
        str_label ='Car'
    elif np.argmax(model_out) == 1: 
        str_label = 'Bike'
    elif np.argmax(model_out) == 2:
        str_label = 'Truck'
    else:
        str_label = 'i dunno'
        print(np.argmax(model_out))
    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_yaxis().set_visible(False)
    y.axes.get_xaxis().set_visible(False)
# y.imshow(image[0])
# plt.title(label)
# y.axis('off')
plt.show()