import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
#方法1：使用os.listdir
import numpy as np
import os
import cv2 as cv
batchsz = 32


#  明星脸
# trianpath = r"E:\untitled3\venv\Digital\dataset\FaceDataSets\DataSet2-FaceOfStar\train"
# testpath = r"E:\untitled3\venv\Digital\dataset\FaceDataSets\DataSet2-FaceOfStar\test"#

# 自己和同学
# trianpath = r"E:\untitled3\venv\Digital\dataset\FaceDataSets\ceshi\train"
# testpath = r"E:\untitled3\venv\Digital\dataset\FaceDataSets\ceshi\train"
trianpath = r"E:\untitled3\venv\Digital\dataset\FaceDataSets\train"
testpath = r"E:\untitled3\venv\Digital\dataset\FaceDataSets\test"

# 都挺好
# trianpath = r"E:\untitled3\venv\Digital\dataset\FaceDataSets\DataSet1-FaceOfAllIsWell\train"
# testpath = r"E:\untitled3\venv\Digital\dataset\FaceDataSets\DataSet1-FaceOfAllIsWell\test"
#

#过程
# keep the pic path
#keep pic name  train
trainLabel = list()
trainPic = list()
outputs = 21
# test
Epochs = 10

testLabel = list()
testPic= list()
# x = list()
# y =  list()

def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1,128, 128, 1])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=outputs)  #  最大元素为 depth - 1
    return x,y


#得到训练集和测试集. 同时对训练，保存结果
def get_thePic_120():
    global trainPic, trainLabel
    global testPic,testLabel
    # 遍历文件夹
    s = ''
    # read the image  and  get the  label     read  the  train

    for filename in os.listdir(trianpath):
        # path.append(filename)
        s = ''
        s += trianpath + "\\" + filename

        pic = cv.imread(s, 0)  # the gray
        pic = cv.resize(pic, (128, 128))


        trainPic.append(pic)
        trainLabel.append(int(filename.split('_')[0]))

    trainPic = np.array(trainPic)
    trainLabel = np.array(trainLabel)

    x, y = preprocess(trainPic, trainLabel)
    # test
    # read the image  and  get the  label     read  the
    # print('datasets:',  y.shape, x.min(), x.max())
    for filename in os.listdir(testpath):
        s = ''
        s += testpath + "\\" + filename
        pic = cv.imread(s, 0)  # the gray
        pic = cv.resize(pic, (128, 128))

        testPic.append(pic)
        testLabel.append(int(filename.split('_')[0]))

    testPic = np.array(testPic)
    testLabel = np.array(testLabel)

    x_val, y_val = preprocess(testPic, testLabel)
    # 网络搭建
    conv_layers = (
        # unit 1
        # layers.Conv2D(8, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(16, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        layers.Dropout(0.5),  # 防止过拟合

        # unit 2
        layers.Conv2D(32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),  # 16
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        layers.Dropout(0.5),  # 防止过拟合

        # unit 3
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),  # 16
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        layers.Dropout(0.5),  # 防止过拟合
        # # unit 4
        # layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),  # 16
        # layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # 展开为向量
        tf.keras.layers.Flatten(),

        layers.Dense(1024, activation=tf.nn.relu),
        layers.Dropout(0.5),  # 防止过拟合
        # layers.Dense(512, activation=tf.nn.relu),
        # 输出层
        layers.Dense(outputs),
        # layers.Dense(outputs, activation=tf.nn.softmax),
    )

    net = Sequential(conv_layers)
    # 输入数据为(x, 28, 28, 1)
    net.build(input_shape=(None, 128, 128, 1))

    net.summary()

    net.compile(optimizer=optimizers.Adam(lr=0.001),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # 训练
    net.fit(x, y, epochs=Epochs, batch_size=batchsz, validation_data=(x_val, y_val), validation_freq=2)  # 8
    # 评测
    # net.evaluate(ds_val)
    # 保存模型
    net.save('model/model.h5')
    # 保存参数
    net.save_weights('model/weights.ckpt')


if __name__ == '__main__':
    get_thePic_120()
    pass
