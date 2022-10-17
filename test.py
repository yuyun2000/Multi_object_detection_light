import tensorflow as tf
import cv2
import numpy as np

def my_softmax(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x)

model = tf.keras.models.load_model("./beizi64m.h5")

imagegt = cv2.imread('./data/val/1661327423827.jpg')
imagegt = cv2.resize(imagegt,(128,128))
image = imagegt.astype(np.float32)
img = image / 255
img = img.reshape(1,128,128,3)
out = model(img,training=True)
out = np.array(tf.reshape(out[0:1,:,:,],(64,64,2)))
for i in range(64):
    for j in range(64):
        out[i][j]=my_softmax(out[i][j])
print(out)
for i in range(64):
    for j in range(64):
        if out[i][j][1] >0.995:
            # out[i][j] =1
            cv2.rectangle(imagegt,(j*2,i*2),(j*2+2,i*2+2),(0,0,255),1)
        else:
            out[i][j] =0
# print(out)
imagegt = cv2.resize(imagegt,(256,256))
cv2.imshow('1',imagegt)
cv2.waitKey(0)
cv2.destroyAllWindows()





# vid = cv2.VideoCapture('./data/zhituan/val/img/t1.mp4')
# fourcc = cv2.VideoWriter_fourcc(*'I420')
# outv = cv2.VideoWriter('output.avi',fourcc,20,(256,256))
# while True:
#     flag,img = vid.read(0)
#     if not flag:
#         break
#     img0 = cv2.resize(img,(128,128))
#     img = img0.astype(np.float32)
#     img = img / 255
#     img = img.reshape(1, 128, 128, 3)
#     out = model(img, training=True)
#     # print(out)
#     out = np.array(tf.reshape(out[0:1,:,:,],(64,64,2)))
#     for i in range(64):
#         for j in range(64):
#             out[i][j]=my_softmax(out[i][j])
#     # print(out)
#     for i in range(64):
#         for j in range(64):
#             if out[i][j][1] > 0.996:
#                 # out[i][j] = 1
#                 cv2.rectangle(img0, (j * 2, i * 2), (j * 2 + 2, i * 2 + 2), (0, 0, 255), 2)
#             else:
#                 out[i][j] = 0
#     # print(out)
#     img0 = cv2.resize(img0, (256, 256))
#     outv.write(img0)
#     cv2.imshow('1', img0)
#     if ord('q') == cv2.waitKey(1):
#         break
# vid.release()
# outv.release()
# #销毁所有的数据
# cv2.destroyAllWindows()