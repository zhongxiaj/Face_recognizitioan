# #
# # change the name  of Pic
# import  os
# import cv2 as  cv
# trianpath =r"E:\untitled3\venv\Digital\dataset\FaceDataSets\test"
# trainPic =[]
# kepa = r"E:\untitled3\venv\Digital\dataset\FaceDataSets\test"
# for filename in os.listdir(trianpath):
#     s = ''
#     s += trianpath + "\\" + filename
#     #pic = cv.imread(s)  # the gray
#     trainPic.append(s)
#
# print(trainPic)
# les = len(trainPic)
# print(les)
# # print(trainPic[0].split('.')[0])
# # print()
# # for i in range(les):
# #  img = cv.imread(trainPic[i])
# #  # # cv.imwrite(kepa+'\\'+"6"+"_"+str(i+1)+'.bmp',img)
# #  # print(trainPic[i].split('\\')[-1].split('.')[0])
# #
# #  s = kepa+'\\'+(trainPic[i].split('\\')[-1].split('.')[0])+'.bmp'
# #
# #  # print(s)
#
#  # cv.imwrite(s,img)
#
#
# 对静态人脸图像文件进行68个特征点的标定


#
#
# #   对人脸特征点的读取

import cv2
import dlib
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
img = cv2.imread('p1.jpg')
faces = detector(img,1)
if (len(faces) > 0):
    for k,d in enumerate(faces):
        cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(255,255,255))
        shape = landmark_predictor(img,d)
        for i in range(68):
            cv2.circle(img, (shape.part(i).x, shape.part(i).y),5,(0,255,0), -1, 8)
            cv2.putText(img,str(i),(shape.part(i).x,shape.part(i).y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,2555,255))
cv2.imshow('Frame',img)
cv2.waitKey(0)

