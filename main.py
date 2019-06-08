import sys
import cv2
import time
import face_detect_utils
import numpy as np
import load
from PIL import ImageFont, ImageDraw, Image
#from helloui import Ui_MainWindow
from graph_120 import Ui_MainWindow
import dlib
import numpy as np


# 未知的;   上线值
upper_bound  = 0.50

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
dlib_face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

pascurrent_image = None

picWidth = 128
picHigh = 128
flag =0

recogPic = [
    {1:"梁朝伟",2:"刘德华",3:"马云",4:"郭冬临",5:"暴走的小吉吉",6:"绝世的陈逗逗",7:"古天乐",8:"赵丽颖",
     9:"邓超",10:"孙俪",11:"岳云鹏",12:"沈腾",13:"何炅",14:"邓紫棋",15:"李荣浩",
     16:"陈赫",17:"钟汉良",18:"刘涛",19:"冯提莫",20:"王力宏",21:"吴亦凡",
     22:"张杰",23:"张家辉",24:"佟丽娅"},
    {0: '李梦旋', 1: '蒋畅', 2: '钟侠骄', 3: '陈洪', 4: '李廷川', 5: '陈毅', 6: '李任宁', 7: '汪林', 8: '张城阳',
     9: '冯松', 10: '胡稳', 11: '林雪梅', 12: '熊英英', 13: '高小霞', 14: '邓杰', 15: '叶港培', 16: '李博录',
     17: '刘天龙', 18: '刘泽城', 19: '柏现迪',20: '王倩倩'}
    ,
    {0: "石天冬", 1: "苏大强", 2: "苏明成", 3: "苏明玉", 4: "苏明哲", 5: "吴非",6: "朱丽" },
]
# pos  0  明星脸， 1  自己，   2 都挺好
pos =  1
w = 621
h = 561
# 这个窗口继承了用QtDesignner 绘制的窗口
class Window(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)
        self.th = None
        self.current_image = None

    def openStream(self):
        self.th = Thread_120(self)
        self.th.changePixmap.connect(self.set_video_image)
        self.th.start()

    def snap(self):
        ret, frame = self.th.cap.read()
        if ret:
            global flag
            flag = 1
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_image = rgb_image
            global pascurrent_image
           # pascurrent_image = rgb_image

            pascurrent_image = face_detect_utils.detect_and_label(frame)
#            print(pascurrent_image.shape)
            cv2.imwrite('pasPic.bmp',pascurrent_image)
            convert_to_QtFormat = QtGui.QImage(rgb_image.data, rgb_image.shape[1],
                                               rgb_image.shape[0],
                                               QImage.Format_RGB888)
            p = convert_to_QtFormat.scaled(320, 240, Qt.KeepAspectRatio)

            self.set_image(self.picLabel, p)

    def openImage(self):
        img_name, img_type = QFileDialog.getOpenFileName(self, "选择图片", "", " *.bmp;;*.jpg;;*.png;;*.jpeg")
        print(img_name, img_type)
        imread = cv2.imread(img_name)
        imread = cv2.cvtColor(imread, cv2.COLOR_BGR2GRAY)
        imread = imread / 255.
        imread = cv2.resize(imread, (picWidth, picHigh))
        self.current_image = np.reshape(imread, (1, picWidth, picHigh, 1))
        # 利用qlabel显示图片
        # 适应设计label时的大小
        png = QtGui.QPixmap(img_name).scaled(self.picLabel.width(), self.picLabel.height())
        # global w ,h
        # w = self.picLabel.width()
        # h = self.imgLabel.height()
      #  print(w,h)
        self.picLabel.setPixmap(png)
        print("openImage")

    def recognize(self):
        global flag
        global pascurrent_image
        if flag ==1:
            flag =0
            imread = cv2.imread('pasPic.bmp')
            imread = cv2.cvtColor(imread, cv2.COLOR_BGR2GRAY)
            imread = imread / 255.
            imread = cv2.resize(imread, (picWidth, picHigh))
            self.current_image = np.reshape(imread, (1, picWidth, picHigh, 1))

      #  self.current_image = np.array(self.picLabel.text())
        result,gailv = load.predict(self.current_image)
        # gailv =list(gailv.numpy())
        # print(type(gailv))
        # print(gailv)
        if gailv[0][int(result)]<upper_bound: # 表示无法识别
            print('Unknown')
            self.lineEdit.setText('Unknown')
        else:
            print(recogPic[pos][int(result)])
            self.lineEdit.setText(recogPic[pos][int(result)])

    def set_video_image(self, image):
        self.set_image(self.imgLabel, image)

    def set_image(self, label, image):
        label.setPixmap(QPixmap.fromImage(image))

# 播放视频线程
class Thread_120(QThread):
    def __init__(self, other):
        super(Thread_120, self).__init__()
        self.cap = None
        self.pause = False

    changePixmap = pyqtSignal(QtGui.QImage)

    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3,540) #  设置分辨率
        self.cap.set(4,540)
        while self.cap.isOpened():
            if 1 - self.pause:
                ret, frame = self.cap.read()
                if ret:
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 在这里可以对每帧图像进行处理
                   # rgb_image = face_detect_utils.detect_and_label_1(rgb_image)
                    # les = len(rgb_image)
                    # -------------------------------------------------------
                    dets = dlib_face_detector(rgb_image, 1)
                    faces = list()  # 把所有的人脸都加到里面去，
                    # 返回的数据集合
                    imgset = list()
                    if len(dets) != 0:
                        for detection in dets:
                            # faces.append(detection)
                            cv2.rectangle(rgb_image,
                                          (detection.left(), detection.top()),
                                          (detection.right(), detection.bottom()),
                                          (0, 255, 0),
                                          2)

                            img = rgb_image[(detection.top() + 2):(detection.bottom() - 2),
                                  (detection.left() + 2):(detection.right() - 2)]
                            # 添加文字
                            rgb_ = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
                            ttfont = ImageFont.truetype("simhei.ttf", 30)
                            draw = ImageDraw.Draw(rgb_)

                            # 在线识别
                            #try:
                            imread = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                           # except cv2.error:
                            imread = imread / 255.
                            imread = cv2.resize(imread, (picWidth, picHigh))
                            current_image = np.reshape(imread, (1, picWidth, picHigh, 1))
                            result,gravit = load.predict(current_image)

                            if gravit[0][int(result)] < upper_bound:  # 表示无法识别
                               s ='Unknown'
                                #self.lineEdit.setText('Unknown')
                            else:
                               s =recogPic[pos][int(result)]
                                # 添加文字
                            draw.text((detection.left(), detection.top() + 10), s,
                                          fill=(25, 25, 255), font=ttfont)
                            rgb_ = cv2.cvtColor(np.array(rgb_), cv2.COLOR_RGB2BGR)
                            #-------------------------------------


                         #   img = cv2.imread('p1.jpg')
                            faces = dlib_face_detector(rgb_, 1)
                            if (len(faces) > 0):
                                for k, d in enumerate(faces):
                                    cv2.rectangle(rgb_, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255))
                                    shape = landmark_predictor(rgb_, d)
                                    for i in range(68):
                                        cv2.circle(rgb_, (shape.part(i).x, shape.part(i).y), 5, (0, 255, 0), -1, 8)
                                        cv2.putText(rgb_, str(i), (shape.part(i).x, shape.part(i).y),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 2555, 255))
                          #  cv2.imshow('Frame', rgb_)
                          #  cv2.waitKey(0)

                            #------------------------------------


                            try:
                                convert_to_QtFormat = QtGui.QImage(rgb_.data, rgb_.shape[1],
                                                                   rgb_.shape[0],
                                                                   QImage.Format_RGB888)
                            except AttributeError:
                                p = convert_to_QtFormat.scaled(w, h, Qt.KeepAspectRatio)
                                #  self.changePixmap.emit(p)
                            else:
                                p = convert_to_QtFormat.scaled(w, h, Qt.KeepAspectRatio)
                            self.changePixmap.emit(p)
                            time.sleep(0.001)

                    else:  # 如果没有人脸，那么就显示原来的额
                        try:
                            convert_to_QtFormat = QtGui.QImage(rgb_image.data, rgb_image.shape[1],
                                                               rgb_image.shape[0],
                                                               QImage.Format_RGB888)
                        except AttributeError:
                            p = convert_to_QtFormat.scaled(w, h, Qt.KeepAspectRatio)
                        else:
                            p = convert_to_QtFormat.scaled(w, h, Qt.KeepAspectRatio)

                        self.changePixmap.emit(p)
                    # 0------------------------------------------------------
                else:
                    break
                time.sleep(0.01)

if __name__ == '__main__':


    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

