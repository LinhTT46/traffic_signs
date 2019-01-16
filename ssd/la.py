import cv2

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Lambda, Dropout

# ======================= NECESSARY IMPORT FOR SSD ============================
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
#======================================================================================

# ======================= THIS IS GLOBAL VARIABLES FOR SSD ============================
img_height = 480 # Height of the input images
img_width = 640 # Width of the input images
img_channels = 3 # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 3 # Number of positive classes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.8, 1.0, 1.25] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = True # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size
classes = ['neg','left','right','stop']

model = build_model(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=None,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=intensity_mean,
                    divide_by_stddev=intensity_range)

# 2: Optional: Load some weights

model.load_weights('ssd7_epoch-04_loss-0.2610_val_loss-0.6570.h5', by_name=True)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
def get_ypred_decoded(r_img):
    y_pred = model.predict(r_img)
    #y_pred = model.predict(r_img)
    y_pred_decoded = decode_detections(y_pred,
                                       confidence_thresh=0.5,
                                       iou_threshold=0.1,
                                       top_k=200,
                                       normalize_coords=normalize_coords,
                                       img_height=img_height,
                                       img_width=img_width)

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    
    return y_pred_decoded

def preprocessing( cv_img):
    return cv2.resize(cv_img, (img_width, img_height))
    
cap = cv2.VideoCapture('sample_x4.avi')
while(cap.isOpened()):
    ret, frame = cap.read()
 
    img = preprocessing(frame)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    classs = 0
    y_pred_decoded = get_ypred_decoded(rgb_img.reshape(1, img_height, img_width, 3))
    
    for box in y_pred_decoded[0]:
        xmin = int(box[-4])
        ymin = int(box[-3])
        xmax = int(box[-2])
        ymax = int(box[-1])
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        classs = int(box[0])
        if classs == 1:
            lala = "turn left"
        elif classs == 2:
            lala = "turn right"

        cv2.putText(img, lala, (20,20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        #draw bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax,ymax), (0,0,255), 2)
        cv2.putText(img, str(label), (xmin,ymin-2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)

    cv2.imshow('traffic_sign_test', img)
    k = cv2.waitKey(1)
    if  k & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
