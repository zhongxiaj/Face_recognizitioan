import cv2
import time
import face_detect_utils
import dlib
dlib_face_detector = dlib.get_frontal_face_detector()


from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
picPath = r"E:\untitled3\venv\Digital\dataset\FaceDataSets\ceshi\train"

class getPic_120():
    def __init__(self):
        self.num = None
        with open('num.txt',"r") as f :
            self.num = int(f.read())
        print(self.num)
        self.n = 1

        cap = cv2.VideoCapture(0)
        while (1):
         # get a frame
            ret, frame = cap.read()
            # show a frame
            image = face_detect_utils.detect_and_label(frame)
         #  cv2.imshow("capture", frame)

            if len(dlib_face_detector(frame, 1))!=0:
                cv2.imwrite((picPath + "\\" + str(self.num) + '_' + str(self.n) + ".bmp"), image)
                print((picPath + "\\" + str(self.num) + '_' + str(self.n) + ".bmp"))
                self.n+=1
            if self.n == 100:
                cap.release()
                cv2.destroyAllWindows()
                break
        self.num +=1
        with open('num.txt', "w") as f:
            f.write(str(self.num))

if __name__ == '__main__':

    ca = getPic_120()
