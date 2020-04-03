#FOR RENAME ALL IMAGE

# import os
# os.getcwd()
# collection = "E:/mike/SKRIPSI/belajarCNN/cnn_custom_dataset/truck"
# for i, filename in enumerate(os.listdir(collection)):
#     os.rename("E:/mike/SKRIPSI/belajarCNN/cnn_custom_dataset/truck/" + filename, "E:/mike/SKRIPSI/belajarCNN/cnn_custom_dataset/truck/{}_{}.png".format('truck',str(i)))

import cv2 
import glob

imagepath = "E:/mike/SKRIPSI/belajarCNN/cnn_custom_dataset/bike"
imgs_names = glob.glob(imagepath+'/*.jpg')
for imgname in imgs_names:
    img = cv2.imread(imgname)
    if img is None:
     print(imgname)