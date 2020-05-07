import cv2
from matplotlib import pyplot as plt
# import mtcnn
# from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np
import os
import pickle
from collections import defaultdict
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
import tensorflow.keras.backend as K

def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)


STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4K": (3840, 2160)
}


def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS['480p']
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    change_res(cap, width, height)
    return width, height


VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID')
}


def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

def load_vgg_model():
    #Define VGG_FACE_MODEL architecture
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.load_weights('extract/extract/vgg_face_weights.h5')
    return model

def predict_face(face_roi):
        face_roi=img_to_array(face_roi)
        face_roi=np.expand_dims(face_roi,axis=0)
        face_roi=preprocess_input(face_roi)
        img_encode=vgg_face(face_roi)
        embed = K.eval(img_encode)
        person = classifier_model.predict(embed)
        # print(np)
        # print(person_rep[np.argmax(person)])
        return (person_rep[np.argmax(person)],person[0][np.argmax(np.argmax(person))])

if __name__ == "__main__":
    model = load_vgg_model()
    vgg_face=Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)
    classifier_model=tf.keras.models.load_model('extract/extract/face_classifier_v1_model.h5')
    with open('extract/extract/person_rep.pickle','rb') as f:
        person_rep = pickle.load(f)
    # print(person_rep)
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('extract/extract\cascades\data\haarcascade_frontalface_default.xml')
    
    filename = 'video.mp4'
    frames_per_seconds = 24.0
    my_res = '480p'

    cap = cv2.VideoCapture(0)
    dims = get_dims(cap, res=my_res)
    # video_type_cv2 = get_video_type(filename)
    # out = cv2.VideoWriter(filename,video_type_cv2,frames_per_seconds,dims)
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter('output/output.avi',fourcc, 20.0, (640,480))
    max_conf = 0
    name1 = ''
    while(True):
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        
        faces_points = face_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=5)
        for index,face_point in enumerate(faces_points):
            x,y,w,h = face_point
            face_roi = cv2.resize(frame[y:y+h,x:x+w],(224,224))
            name ,conf= predict_face(face_roi)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            color_font = (255, 255, 255)
            stroke = 2
            cv2.putText(frame,name,(x,y+10),font,1,color_font)
            color = (255, 0, 0)
            stroke = 2
            end_cord_x = x+w
            end_cord_y = y+h
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)

        out.write(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()