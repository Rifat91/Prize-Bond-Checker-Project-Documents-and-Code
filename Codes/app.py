
import numpy as np
import os
import tensorflow as tf
import pytesseract 
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
import cv2
import gradio as gr
import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
from core.config import cfg
from statistics import mean
import pandas as pd

#Diseable cuda
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def load_config():


    STRIDES = np.array(cfg.YOLO.STRIDES)

    ANCHORS = get_anchors(cfg.YOLO.ANCHORS)

        
    XYSCALE = cfg.YOLO.XYSCALE 
    NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))

    return STRIDES, ANCHORS, NUM_CLASS, XYSCALE

def get_anchors(anchors_path, tiny=False):
    anchors = np.array(anchors_path)
    if tiny:
        return anchors.reshape(2, 3, 2)
    else:
        return anchors.reshape(3, 3, 2)
    
# function for cropping each detection and saving as new image
def crop_objects(img, data, allowed_classes):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    #create dictionary to hold count of objects for image name
    counts = dict()
    x=[]
    for i in range(num_objects):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            # get box coords
            xmin, ymin, xmax, ymax = boxes[i]
            cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]    

        else:
            continue
    return cropped_img

def Gen_Text(images):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = load_config()
    input_size = 320

    saved_model_loaded = tf.saved_model.load('./checkpoints/custom-320/', tags=[tag_constants.SERVING])
    # images = './test_img/1.jpg'


    original_image = images
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.


    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)



    pred_bbox = infer(batch_data)

    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]


        
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold= 0.50
    )



    original_h, original_w, _ = original_image.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)



    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]



    class_names = read_class_names(cfg.YOLO.CLASSES)


    allowed_classes = ['number']

    mf = crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, allowed_classes)


    # img = cv2.imread('tst8.jpg')
    # gray = cv2.cvtColor(mf, cv2.COLOR_BGR2GRAY)
    txt = pytesseract.image_to_string(mf, lang='ben')
    df = pd.read_csv('winner.csv')
    for key, number in df.iterrows():
        if number.Number in txt:
           txt = number.Number
           txt = txt+'\nCongratulations, you are winner!'
           break
    if not len(txt)>=33:
        txt = txt+'\nSorry, try again next time!'

    return mf, txt



if __name__ == '__main__':
    iface = gr.Interface(fn = Gen_Text, 
                        inputs=gr.inputs.Image(), 
                        outputs=["image","text"])
    iface.launch(share=True)
    














