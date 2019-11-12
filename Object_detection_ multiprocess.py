import multiprocessing
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

def cam1(pathh):
    cred = credentials.Certificate("C:/Users/gokul/Desktop/schoolai-firebase-adminsdk-78lrk-064b6e6416.json")
    firebase_admin.initialize_app(cred,{
        'databaseURL': 'https://schoolai.firebaseio.com'
    })
    print(datetime.now())

    MODEL_NAME = 'inference_graph'
    VIDEO_NAME = 'kathir6.mp4'
    CWD_PATH = os.getcwd()

    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

    PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

    NUM_CLASSES = 2

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    ref = db.reference('KATHIR/CAM1')
    video = cv2.VideoCapture(pathh)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
    x=0
    while(video.isOpened()):
        x=x+1
        ret, frame = video.read()
        if(ret==True and x==2):
            x=0
            frame_expanded = np.expand_dims(frame, axis=0)

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.20)
            now = datetime.now()
            ims=cv2.resize(frame,(960,540))
            cv2.imshow('Student detector CAM1', ims)
            if (scores[0,0]>0.65):
                curdate=str(datetime.now()).split(" ")
                hrmin=curdate[1].split(":")
                users_ref = ref.child(curdate[0]+'/'+hrmin[0]+':'+hrmin[1])
                users_ref.set({'Flag':1})
        if cv2.waitKey(1) == ord('q'):
            video.release()
            cv2.destroyAllWindows()

def cam2():
    cred = credentials.Certificate("C:/Users/gokul/Desktop/schoolai-firebase-adminsdk-78lrk-064b6e6416.json")
    firebase_admin.initialize_app(cred,{
        'databaseURL': 'https://schoolai.firebaseio.com'
    })
    print(datetime.now())

    MODEL_NAME = 'inference_graph'
    VIDEO_NAME = 'kathir7.mp4'
    CWD_PATH = os.getcwd()

    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

    PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

    NUM_CLASSES = 2

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    ref = db.reference('KATHIR/CAM1')
    video = cv2.VideoCapture('C:/tensorflow1/models/research/object_detection/kathir7.mp4')
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
    x=0
    while(video.isOpened()):
        x=x+1
        ret, frame = video.read()
        if(ret==True and x==2):
            x=0
            frame_expanded = np.expand_dims(frame, axis=0)

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.60)
            now = datetime.now() 
            ims=cv2.resize(frame,(960,540))
            cv2.imshow('Student detector CAM2', ims)
            if (scores[0,0]>0.65):
                curdate=str(datetime.now()).split(" ")
                hrmin=curdate[1].split(":")
                users_ref = ref.child(curdate[0]+'/'+hrmin[0]+':'+hrmin[1])
                users_ref.set({'Flag':1})
        if cv2.waitKey(1) == ord('q'):
            video.release()
            cv2.destroyAllWindows()

if __name__ == "__main__": 

    p1 = multiprocessing.Process(target=cam1,args=("C:/tensorflow1/models/research/object_detection/kathir7.mp4",)) 
    p2 = multiprocessing.Process(target=cam2) 
   
    p1.start() 
 
    p2.start() 
  
  
    p1.join() 
   
    p2.join() 
  
   
    print("Done!") 
