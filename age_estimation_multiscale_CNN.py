import csv
import pandas as pd
import random 
import tensorflow as tf
import dlib
import os
import cv2 
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
database_path = "/media/sf_RZV/DeepLearning/group_practise/database/UTKFace"
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
patches_scale = []
patches_scale.append(list(range(1, 22)))
patches_scale.append([11, 12, 15, 16, 17, 18, 20, 21])
patches_scale.append([11, 12, 17, 19])
patches_scale.append([18])
remove_list = []
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
    img = dlib.load_rgb_image(file_name)
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
        l = [70, 36, 42, 24]
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
                if (crop_image.shape[0] == 0) or (crop_image.shape[1] == 0):
                    print("crop image fail")
                    remove_list.append(file_name_)
                    break
                if not (os.path.isdir('debug/scale(' + str(i) + ')')):
                    os.mkdir('debug/scale(' + str(i) + ')')
                if not (os.path.isdir('debug/scale(' + str(i) + ')/patch(' + str(pid) + ')')):
                    os.mkdir('debug/scale(' + str(i) + ')/patch(' + str(pid) + ')')
                dlib.save_image(crop_image, 'debug/scale(' + str(i) + ')/patch(' + str(pid) + ')' + '/' + file_name_ + '.jpg')

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
####################################################################

database = pd.read_csv('UTKFace.csv')
dlib_predictor_path = "shape_predictor_68_face_landmarks.dat"
age_selection_1 = (database.Age < maximum_age)
age_selection_2 = (database.Age >= minimum_age)
gender_selection = (database.Race == race["asian"])

data_filters = age_selection_1 & age_selection_2 & gender_selection
database = database[data_filters]
data_set = database.sample(frac=ds_ratio)

#get train, validation, test dataset
train_ds = data_set.sample(frac=train_ds_ratio)
database_ex_train = database.drop(train_ds.index)
valid_ds = database_ex_train.sample(frac= (valid_ds_ratio/(valid_ds_ratio + test_ds_ratio)))
test_ds = database_ex_train.drop(valid_ds.index)

#debug
for i in range (0, 1000):
    create_face_patches(train_ds['Filename'].values[i])
    #draw_landmark(test_ds['Filename'].values[i], 1, 68)
print("number image need to be removed\n", len(remove_list))





