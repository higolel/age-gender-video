#!/usr/bin/env python3

import rospy
import base64
import cv2
from face_plate_msgs.msg import Face_pic, Face_compare

from deepface import DeepFace

def Face_pic_callback(msg):
    face_pic_msg = Face_pic()
    face_pic_msg = msg
    #print(face_pic_msg.vin)
    pic_image = base64.b64decode(face_pic_msg.facePicture)

    pic_write = open(image_path_, 'wb')
    pic_write.write(pic_image)
    pic_write.close()

    obj = DeepFace.analyze(image_path_, actions = ['age', 'gender', 'race', 'emotion'])
    print(obj["age"]," years old ",obj["dominant_race"]," ",obj["dominant_emotion"]," ", obj["gender"])
    face_pic_msg.age = int(obj["age"])
    if obj["gender"] == "Man":
        face_pic_msg.sex = 1
    elif obj["gender"] == "Woman":
        face_pic_msg.sex = 2
    else:
        face_pic_msg.sex = 0

    if obj["dominant_race"] == "asian":
        face_pic_msg.race = 1
    elif obj["race"] == "black":
        face_pic_msg.race = 2
    elif obj["race"] == "white":
        face_pic_msg.race = 3
    elif obj["race"] == "middle eastern":
        face_pic_msg.race = 4
    elif obj["race"] == "indian":
        face_pic_msg.race = 5
    elif obj["race"] == "latino":
        face_pic_msg.race = 6
    else:
        face_pic_msg.race = 0

    if obj["dominant_emotion"] == "angry":
        face_pic_msg.facialExpression = 1
    elif obj["dominant_emotion"] == "fear":
        face_pic_msg.facialExpression = 2
    elif obj["dominant_emotion"] == "neutral":
        face_pic_msg.facialExpression = 3
    elif obj["dominant_emotion"] == "sad":
        face_pic_msg.facialExpression = 4
    elif obj["dominant_emotion"] == "disgust":
        face_pic_msg.facialExpression = 5
    elif obj["dominant_emotion"] == "happy":
        face_pic_msg.facialExpression = 6
    elif obj["dominant_emotion"] == "surprise":
        face_pic_msg.facialExpression = 7
    else:
        face_pic_msg.facialExpression = 0

    pub_.publish(face_pic_msg)



if __name__ == "__main__":
    rospy.init_node('deepface')
    print("11111111111111")
    image_path_ = rospy.get_param('image_path')
    rospy.Subscriber("/face_pic_msg", Face_pic, Face_pic_callback, queue_size = 1, buff_size = 2**24)
    pub_ = rospy.Publisher("face_attribute_msg", Face_pic, queue_size = 1)

    rospy.spin()


