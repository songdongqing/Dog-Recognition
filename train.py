import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import Dense,AveragePooling2D, Flatten
from keras.models import Sequential,Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")   # 关闭警告

# 出现了一个问题，loss越变越大，以至于放不下了，结果为nan
# 结果居然是学习率太高，为0.01.。。简直了

learning_rate = 0.0001
train_path = 'E:/wangyyun_data/07-data/train/'
test_path = 'E:/wangyyun_data/07-data/test/'
img_width = 150
img_height = 150
batch_size = 32
epoch = 20

vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
print("模型加载成功！", vgg16)
vgg16.summary()

output = vgg16.get_layer("block5_pool").output   # shape=(None, 7, 7, 512)
output = Flatten(name='flatten')(output)
output = Dense(10, activation='softmax', name='predictions')(output)

model = Model(vgg16.input, output)
optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

# this is the augmentation configuration we will use for validation:
# only rescaling
val_datagen = ImageDataGenerator(
        rescale=1./255,)

# directory: path to the target directory. It should contain one subdirectory per class.
# Any PNG, JPG, BMP, PPM or TIF images inside each of the subdirectories directory tree
# will be included in the generator.
train_generator = train_datagen.flow_from_directory(
        train_path,
        batch_size = batch_size,
        target_size=(img_height,img_width),
        shuffle = True)

validation_generator = val_datagen.flow_from_directory(
        test_path,
        batch_size=batch_size,
        target_size=(img_height,img_width),
        shuffle = True)

print(len(train_generator))   # 73
print(len(validation_generator))  # 19
label = train_generator.class_indices
print("标签为：", label)

model.fit_generator(
        train_generator,
        steps_per_epoch = len(train_generator),
        epochs = epoch,
        validation_data = validation_generator,
        validation_steps=len(validation_generator))

model.save("vgg16_model.h5")
print("训练完成！")