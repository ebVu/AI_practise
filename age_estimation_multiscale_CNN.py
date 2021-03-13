import csv
import pandas as pd
import random 
import tensorflow as tf
import dlib
import os
import cv2 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, concatenate
from tensorflow.keras.models import Model
import numpy as np
###################################################################
#Define global variable
###################################################################
maximum_age = 75
minimum_age = 0
ds_ratio = 1
train_ds_ratio = 0.8
valid_ds_ratio = 0.1
test_ds_ratio = 0.1
race = {"white":0, "black":1, "asian":2, "indian":3, "others":4}
database_path = "/media/sf_RZV/DeepLearning/group_practise/myworking/AI_practise/workspace/database/UTKFace"
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(predictor_path)
patches = {1: [37, 38], 2: [45, 46], 3: [38,39], 4: [44, 45], 5: [39, 40],
        6: [43, 44], 7: [37, 42], 8: [46, 47], 9: [40, 41], 10: [43, 48],
        11: [37, 38, 39, 40, 41, 42], 12: [43, 44, 45, 46, 47, 48],
        13: [32,33], 14: [35, 36], 15: [49, 50, 60, 61],
        16: [54, 55, 56, 65], 17: [28, 29, 30], 18: [29, 30, 31],
        19: [51, 52, 53, 62, 63, 64], 20: [57, 58, 59, 66, 67, 68],
        21: [8, 9, 10]}
middle_landmark_patches = [17, 18, 19, 20, 21]
left_landmark_patches =  [1, 3, 5, 7, 9, 11, 13, 15] + middle_landmark_patches    
right_landmark_patches = [2, 4, 6, 8, 10, 12, 14, 16] + middle_landmark_patches
patches_scale = []
patches_scale.append(list(range(1, 22))) 
patches_scale.append([11, 12, 15, 16, 17, 18, 20, 21])
patches_scale.append([11, 12, 17, 19])
patches_scale.append([18])
remove_list = []
left_input=[]
right_input=[]
image_input = []

class Input_Image:
    def __init__(self, scale, patch_id, img):
        self.scale = scale
        self.patch_id = patch_id
        self.img = np.reshape(img, (img.shape[0], img.shape[1], 1))


####################################################################
# Define function
####################################################################

def get_image_label(ds):
    bits = tf.io.read_file(database_path + ds.Filename)
    image = tf.image.decode_jpeg(bits)
    label = tf.strings.to_number(ds.Age, out_type=tf.dtypes.int32)
    return image, label
def create_face_patches(file_name):
    print("Processing file: {}".format(file_name))
    err_flag = False
    img = dlib.load_rgb_image(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    file_name_ = os.path.splitext(os.path.basename(file_name))[0]
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = face_detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = face_predictor(img, d)
        l0 = abs(shape.part(27).y - shape.part(51).y) # distance between point 28 and 52 as reference
        l = [70, 60, 42, 24]
        ratio = []
        pad = [24, 24, 24, 24]
        for l_ in l:
            ratio.append(l_/l0)
        # resize image to 4 image with 4 scales
        for i in range(4):
            resize_img = cv2.resize(img, (round(img.shape[1] * ratio[i]) , round(img.shape[0] * ratio[i])))

            for pid in patches_scale[i]:
                xpatch_points = [round(ratio[i] * shape.part(x-1).x) for x in patches[pid]]
                ypatch_points = [round(ratio[i] * shape.part(x-1).y) for x in patches[pid]]
        
                maxx = max(xpatch_points)
                minx = min(xpatch_points)
                maxy = max(ypatch_points)
                miny = min(ypatch_points) 
                if ((minx < 0) or (miny < 0)):
                    print("train image error should be removed")
                    remove_list.append(file_name_)
                    break
                mid_pointx = round((maxx + minx)/2)
                mid_pointy = round((maxy + miny)/2)

                crop_image = resize_img[mid_pointy-pad[i]:mid_pointy +pad[i],mid_pointx-pad[i]:mid_pointx +pad[i]]
                if (crop_image.shape[0] < 48) or (crop_image.shape[1] < 48):
                    print(file_name_,": crop image fail")
                    print("scale", i,  "pid:", pid, " shape[0]: ", crop_image.shape[0], " shape[1]: ", crop_image.shape[1])
                    remove_list.append(file_name_)
                    err_flag = True
                    break
                if not (os.path.isdir('debug/scale(' + str(i) + ')')):
                    os.mkdir('debug/scale(' + str(i) + ')')
                if not (os.path.isdir('debug/scale(' + str(i) + ')/patch(' + str(pid) + ')')):
                    os.mkdir('debug/scale(' + str(i) + ')/patch(' + str(pid) + ')')
                # right land mark patches are mirrored
                img_input_i = Input_Image(i, pid, crop_image)
                image_input.append(img_input_i)       
                dlib.save_image(crop_image, 'debug/scale(' + str(i) + ')/patch(' + str(pid) + ')' + '/' + file_name_ + '.jpg')
            if err_flag != False:
                break
    return err_flag
def draw_landmark(file_name, start_point, end_point):
    print("Processing file: {}".format(file_name))
    img = dlib.load_rgb_image(file_name)
    dets = face_detector(img, 1)
    for k, d in enumerate(dets):
        shape = face_predictor(img, d)
        for i in range(start_point, end_point):
            cv2.circle(img, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1)
            # Displaying the image 
        #cv2.imshow('image', img) 
    file_name_ = os.path.splitext(os.path.basename(file_name))[0]
    dlib.save_image(img, 'debug/' + file_name_ + "_debug.jpg")

def sub_cnn(img_input):
    x = Conv2D(
        filters = 16,
        kernel_size = (7, 7),
        strides =(1, 1),
        padding = 'same',
        activation = 'relu')(img_input)
    x = MaxPooling2D (
        pool_size=(5, 5), 
        strides=(3, 3), 
        padding='same', 
    )(x) #### which is a cross channel unit
    x = Conv2D(
        filters = 16,
        kernel_size = (3, 3),
        strides =(1, 1),
        padding = 'same',
        activation = 'relu'
    )(x)
    return x

def build_cnn():

    sub_cnn_arr = []
    img_input_arr = []
    for i in range (23):
        img_input_arr.append(Input(shape=(48, 48, 1)))
        sub_cnn_arr.append(sub_cnn(img_input_arr[i]))
    fcn = concatenate(sub_cnn_arr)
    out = Flatten()(fcn)
    out = Dense(368, activation='relu') (out)
    out = Dense(maximum_age - minimum_age + 1, activation='softmax')(out)

    model = Model(img_input_arr, out)
    print(model.summary())
    model.compile('sgd', 'categorical_crossentropy', ['accuracy'])
    return model

####################################################################

database = pd.read_csv('UTKFace.csv')
database = database.dropna()
dlib_predictor_path = "shape_predictor_68_face_landmarks.dat"
age_selection_1 = (database.Age.astype(int) < maximum_age)
age_selection_2 = (database.Age.astype(int) >= minimum_age)
gender_selection = (database.Race.astype(int) == race["asian"])

data_filters = age_selection_1 & age_selection_2 & gender_selection
database = database[data_filters]
data_set = database.sample(frac=ds_ratio)

#get train, validation, test dataset
train_ds = data_set.sample(frac=train_ds_ratio)
database_ex_train = database.drop(train_ds.index)
valid_ds = database_ex_train.sample(frac= (valid_ds_ratio/(valid_ds_ratio + test_ds_ratio)))
test_ds = database_ex_train.drop(valid_ds.index)

# build network
model = build_cnn()

#debug
for img_count in range (0, 100):
    err = create_face_patches(train_ds['Filename'].values[img_count])
    if (err == True):
        print("error create face patches")
        continue
    #prepare left input
    for i in range(4): 
        for pid in patches_scale[i]:
            image_input_id = Input_Image(image_input[0].img.shape[0], image_input[0].img.shape[1], \
            np.zeros((image_input[0].img.shape[0], image_input[0].img.shape[1], 1)))            
            if pid in left_landmark_patches:
                for image_input_id in image_input:
                    if (image_input_id.patch_id == pid) and (image_input_id.scale == i):
                        break
                image_input_id.img = np.reshape(image_input_id.img, (1, 48, 48, 1))
                left_input.append(np.asarray(image_input_id.img))

            if pid in right_landmark_patches:
                for image_input_id in image_input:
                    if (image_input_id.patch_id == pid)  and (image_input_id.scale == i):
                        break
                image_input_id.img = np.reshape(image_input_id.img, (1, 48, 48, 1))
                right_input.append(np.asarray(image_input_id.img))

    #print('lengh of left input', len(left_input))
    #print('lengh of right input', len(right_input))

    X = left_input
    y = np.zeros((1, maximum_age - minimum_age + 1))
    y[0][train_ds['Age'].values[img_count]] = 1

    model.fit(X, y, batch_size = 1)
    print("predict")
    print(model.predict(X))
    left_input = []
    right_input = []
print("number image need to be removed\n", len(remove_list))



