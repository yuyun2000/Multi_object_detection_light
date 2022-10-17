import cv2
import os

# list = os.listdir('./data/train/img')
# print(list)
# for i in range(len(list)):
#     img = cv2.imread('./data/train/img/%s'%(list[i]))
#     img = cv2.resize(img,(128,128))
#     cv2.imwrite('./data/train/img/%s'%(list[i]),img)

img = cv2.imread('./bee.jpg' )
img = cv2.resize(img, (160, 160))
cv2.imwrite('./bee160.jpg',img)