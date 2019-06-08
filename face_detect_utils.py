import cv2
import dlib
from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
import time
import load
# from main import recogPic
picWidth= 128
picHigh = 128

recogPic = [
    {1:"梁朝伟",2:"刘德华",3:"马云",4:"郭冬临",5:"暴走的小吉吉",6:"绝世的陈逗逗",7:"古天乐",8:"赵丽颖",
     9:"邓超",10:"孙俪",11:"岳云鹏",12:"沈腾",13:"何炅",14:"邓紫棋",15:"李荣浩",
     16:"陈赫",17:"钟汉良",18:"刘涛",19:"冯提莫",20:"王力宏",21:"吴亦凡",
     22:"张杰",23:"张家辉",24:"佟丽娅"},
    {1:"钟侠骄",2:"汪林",3:"邓杰",4:"叶港培",5:"李博录",6:"高小霞",}
]
# pos  0  明星脸， 1  自己，   2 都挺好
pos =  1

dlib_face_detector = dlib.get_frontal_face_detector()

def detect_and_sub(img, model_path):
    sp = dlib.shape_predictor(model_path)
    dets = dlib_face_detector(img, 1)
    faces = dlib.full_object_detections()
    i = 0
    if len(dets) != 0:
        for detection in dets:
            faces.append(sp(img, detection))
        images = dlib.get_face_chips(img, faces, size=128)
        return images
    return None

def detect_and_label(img1):
    dets = dlib_face_detector(img1, 1)
   # print(dets,type(dets))
    if len(dets) != 0:
        for detection in dets:
            cv2.rectangle(img1,
                          (detection.left(), detection.top()),
                          (detection.right(), detection.bottom()),
                          (0, 255, 0),
                          2)

        # // 获取感兴趣区域
        img = img1[(detection.top()+2):(detection.bottom()-2), (detection.left()+2):(detection.right()-2)]

        img = cv2.resize(img, (128, 128))


        time.sleep(0.001)
        return img
        # return rgb_image
def detect_and_label_1(img1):
    dets = dlib_face_detector(img1, 1)
    faces  = list() # 把所有的人脸都加到里面去，
    # 返回的数据集合
    imgset = list()
    if len(dets) != 0:
        for detection in dets:
            #faces.append(detection)
             cv2.rectangle(img1,
                          (detection.left(), detection.top()),
                          (detection.right(), detection.bottom()),
                          (0, 255, 0),
                          2)

        img = img1[(detection.top() + 2):(detection.bottom() - 2), (detection.left() + 2):(detection.right() - 2)]
# 添加文字
        rgb_ = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        ttfont = ImageFont.truetype("simhei.ttf", 30)
        draw = ImageDraw.Draw(rgb_)

 #在线识别
        imread = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imread = imread / 255.
        imread = cv2.resize(imread, (picWidth, picHigh))
        current_image = np.reshape(imread, (1, picWidth, picHigh, 1))
        result = load.predict(current_image)

# 添加文字
        draw.text((detection.left(), detection.top() + 10), recogPic[pos][int(result)], fill=(25, 0, 255), font=ttfont)
        rgb_ = cv2.cvtColor(np.array(rgb_), cv2.COLOR_RGB2BGR)

        time.sleep(0.001)
        return rgb_
    else: # 如果没有人脸，那么就显示原来的额
        return img1
#
# imread = cv2.imread("20_1.bmp")
#
# detect_and_label(imread)
#  cv2.imwrite("20_2.bmp", img)
