#!/usr/bin/env python3

import rospy
import base64
import cv2
from face_plate_msgs.msg import Face_pic, Face_compare
from pathlib import Path
import numpy as np
import dlib
import argparse
from contextlib import contextmanager
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from src.factory import get_model
import tensorflow as tf
import queue
import threading
import time

tf_config = tf.compat.v1.ConfigProto()
#tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4 # 分配40%
tf_config.gpu_options.allow_growth = True # 自适应
session = tf.compat.v1.Session(config = tf_config)

que = queue.Queue()

# --------------------------------------------------------
##
# \概要:    切割图像
#
# \参数:    frame
#
# \返回:    切割后图像帧
# --------------------------------------------------------
def Image_intercept(frame):
    sp = frame.shape
    height = sp[0]
    weight = sp[1]

    rect_frame = frame[int(height / 4) : int(height / 4 * 3), int(weight / 9) : int(weight / 9 * 8)]

    return rect_frame

# --------------------------------------------------------
##
# \概要:    cv2 转 base64
#
# \参数:    image
#
# \返回:    base64_str
# --------------------------------------------------------
def cv2base64(image):
    base64_str = cv2.imencode('.jpg', image)[1]
    base64_str = str(base64.b64encode(base64_str))[2 : -1]

    return base64_str

# --------------------------------------------------------
##
# \概要:    发布消息
#
# \参数:    cp_frame, frame_face, age, sex
#
# \返回:
# --------------------------------------------------------
def Pub_face_pic_message(cp_frame, frame_face, age, sex, pub_):
    print("222222222222222")
    face_pic_msg = Face_pic()
    face_pic_msg.vin = "as00032";
    face_pic_msg.deviceId = "032人脸";
    face_pic_msg.pictureType = 1;
    face_pic_msg.sex = sex;
    face_pic_msg.age = age;
    face_pic_msg.facialExpression = 0;
    face_pic_msg.race = 1;
    face_pic_msg.hat = 1;
    face_pic_msg.bmask = 1;
    face_pic_msg.eyeglass = 1;

    t = time.time()
    face_pic_msg.capTime = int(round(t * 1000))

    frame_face = np.ascontiguousarray(frame_face)
    cp_frame = np.ascontiguousarray(cp_frame)
    face_pic_msg.facePicture = cv2base64(frame_face)
    face_pic_msg.faceScenePicture = cv2base64(cp_frame)

    pub_.publish(face_pic_msg)

# --------------------------------------------------------
##
# \概要:    接收线程
#
# \参数:
#
# \返回:
# --------------------------------------------------------
def Receive(video_, rate_r):
    print('Start Receive')
    cap = cv2.VideoCapture(video_)

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            print("receive success")
            que.put(frame)
            # Throw out the old image in the queue
            if que.qsize() > 1:
                que.get()
        else:
            rospy.loginfo("Capturing image failed.")

        rate_r.sleep()

# --------------------------------------------------------
##
# \概要:    发布线程
#
# \参数:
#
# \返回:
# --------------------------------------------------------
def Display(model, detector, img_size, pub_, rate_d):
    print('Start Display')
    while not rospy.is_shutdown():
        if que.empty() != True:
            frame = que.get()

            frame_intercept = Image_intercept(frame)
            sp = frame_intercept.shape
            img_h = sp[0]
            img_w = sp[1]

            # detect faces using dlib detector
            detected = detector(frame_intercept, 1)
            faces = np.empty((len(detected), img_size, img_size, 3))

            if len(detected) > 0:
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - 0.4 * w), 0)
                    yw1 = max(int(y1 - 0.4 * h), 0)
                    xw2 = min(int(x2 + 0.4 * w), img_w - 1)
                    yw2 = min(int(y2 + 0.4 * h), img_h - 1)
                    cv2.rectangle(frame_intercept, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    print(yw2 - yw1)
                    print(xw2 - xw1)
                    # cv2.rectangle(frame_intercept, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                    faces[i] = cv2.resize(frame_intercept[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))
                # predict ages and genders of the detected faces
                results = model.predict(faces)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()

                # draw results
                for i, d in enumerate(detected):
                    print(predicted_ages[i])
                    print(predicted_genders[i][0])
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 + (0.5 * w) - (413 / 2)), 0)
                    yw1 = max(int(y1 + (0.5 * h) - (626 / 2)), 0)
                    xw2 = min(int(x2 - (0.5 * w) + (413 / 2)), img_w - 1)
                    yw2 = min(int(y2 - (0.5 * h) + (626 / 2)), img_h - 1)

                    rect_frame = frame_intercept[yw1 : yw2, xw1 : xw2]
                    age = int(predicted_ages[i])
                    sex = 1 if predicted_genders[i][0] < 0.5 else 2

                    Pub_face_pic_message(frame, rect_frame, age, sex, pub_)
            print('** publishing webcam_frame ***')
        else:
            print("que is empty")

        rate_d.sleep()

def main():
    rospy.init_node('deepface')
    rate_r = rospy.Rate(15)
    rate_d = rospy.Rate(1)
    print("11111111111111")

    video_sub_ = 'H.265/ch1/main/av_stream'
    image_path_ = rospy.get_param('image_path')
    weight_file_ = rospy.get_param('weight_file')
    video_ = rospy.get_param('~video')
    cam_ = rospy.get_param('~cam')

    video_ = video_ + video_sub_

    model_name, img_size = Path(weight_file_).stem.split("_")[:2]
    img_size = int(float(img_size))
    detector = dlib.get_frontal_face_detector()
    cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
    model = get_model(cfg)
    model.load_weights(weight_file_)

    #rospy.Subscriber("/face_pic_msg", Face_pic, Face_pic_callback, queue_size = 1, buff_size = 2**24)
    #Sub_video_fun()

    pub_ = rospy.Publisher("/face_attribute_msg", Face_pic, queue_size = 1)

    pthread_1 = threading.Thread(target = Receive, args = (video_, rate_r))
    pthread_2 = threading.Thread(target = Display, args = (model, detector, img_size, pub_, rate_d))
    pthread_1.start()
    pthread_2.start()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
