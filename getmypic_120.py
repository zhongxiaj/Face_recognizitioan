
import dlib
import face_detect_utils
# pylint: disable=invalid-name12
path  = r"E:\untitled3\venv\Digital\dataset\FaceDataSets"
import os
import random
import numpy as np
import cv2
dlib_face_detector = dlib.get_frontal_face_detector()

def createdir_120(*args):
    ''' create dir'''
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)

IMGSIZE = 128

def getpaddingSize_120(shape):
    ''' get size to make image to be a square rect '''
    h, w = shape
    longest = max(h, w)
    result = (np.array([longest]*4, int) - np.array([h, h, w, w], int)) // 2
    return result.tolist()

def dealwithimage_120(img, h=64, w=64):
    ''' dealwithimage '''
    #img = cv2.imread(imgpath)
    top, bottom, left, right = getpaddingSize(img.shape[0:2])
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.resize(img, (h, w))
    return img

def relight_120(imgsrc, alpha=1, bias=0):
    '''relight'''
    imgsrc = imgsrc.astype(float)
    imgsrc = imgsrc * alpha + bias
    imgsrc[imgsrc < 0] = 0
    imgsrc[imgsrc > 255] = 255
    imgsrc = imgsrc.astype(np.uint8)
    return imgsrc

def getfacefromcamera_120(outdir,name,laber):
    createdir_120(outdir)
    camera = cv2.VideoCapture(0)
    haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    n = 1
    while 1:
        if (n <= 50):
            print('It`s processing %s image.' % n)
            # 读帧-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
            success, img = camera.read()
        #-------------------------------------

            #  cv2.imshow("capture", frame)

            # if len(dlib_face_detector(frame, 1)) != 0:
            #     cv2.imwrite((picPath + "\\" + str(self.num) + '_' + str(self.n) + ".bmp"), image)
            #     print((picPath + "\\" + str(self.num) + '_' + str(self.n) + ".bmp"))
            #
            #--------------------------------


            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(gray_img, 1.3, 5)
           # faces = face_detect_utils.detect_and_label(img)
            for f_x, f_y, f_w, f_h in faces:
                #could deal with face to train12
                if len(dlib_face_detector(img, 1)) != 0 :
                   # img = face_detect_utils.detect_and_label(img)
                    face = img[f_y:f_y + f_h, f_x:f_x + f_w]
                    face = cv2.resize(face, (IMGSIZE, IMGSIZE))
                    face = relight_120(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                    cv2.imwrite(os.path.join(outdir, laber+'_'+name+str(n)+'.bmp'), face)
                    cv2.putText(img, 'text', (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  #显示名字
                    img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
                    n+=1
            cv2.imshow('img', img)
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break
        else:
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    name = input('please input your name: ')
    laber=input('please input your number: ')
    getfacefromcamera_120(os.path.join(path, name),name,laber)