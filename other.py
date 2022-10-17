
import os
import xml.dom.minidom
import numpy as np


def readxml(filename):
    '''
    返回中心
    '''
    DOMTree = xml.dom.minidom.parse(filename)
    data = DOMTree.documentElement
    dataall = data.getElementsByTagName('object')
    object_num = len(dataall)
    xl = np.zeros((object_num))
    yl = np.zeros((object_num))
    xminl = np.zeros((object_num))
    yminl = np.zeros((object_num))
    xmaxl = np.zeros((object_num))
    ymaxl = np.zeros((object_num))
    for i in range(object_num):
        data = dataall[i]
        # style = xml中的大类 ; typename = 细分属性; typevalue = 细分属性的值; valuename = xml文件，需要获取的值的tag;
        def get_data_vaule(style, typename, typevalue, valuename):
            nodelist = data.getElementsByTagName(style)  # 根据标签的名字获得节点列表
            for node in nodelist:
                if typevalue == node.getAttribute(typename):
                    node_name = node.getElementsByTagName(valuename)
                    value = node_name[0].childNodes[0].nodeValue
                    return value
            return

        class_name = get_data_vaule('object', "", "", 'name')
        # print('class_name:', class_name)

        xmin = get_data_vaule('bndbox', "", "", 'xmin')
        # print('xmin:', xmin)
        ymin = get_data_vaule('bndbox', "", "", 'ymin')
        # print('ymin:', ymin)
        xmax = get_data_vaule('bndbox', "", "", 'xmax')
        # print('xmax:', xmax)
        ymax = get_data_vaule('bndbox', "", "", 'ymax')
        # print('ymax:', ymax)

        xl[i] = (int(xmax)+int(xmin))/2
        yl[i] = (int(ymax)+int(ymin))/2
        xminl[i] = int(xmin)
        xmaxl[i] = int(xmax)
        yminl[i] = int(ymin)
        ymaxl[i] = int(ymax)
    # print(xminl)
    return xl,yl,xminl,xmaxl,yminl,ymaxl

def make_label(filename):
    '''
    制作标签，8*8的特征图，二分类，标签形状为8*8*1 有物体为1，其余都为0
    '''
    x, y,xmin,xmax,ymin,ymax = readxml(filename)
    len = x.shape[0]
    label = np.zeros((32, 32, 3))#0是否有物体12质心的xy偏移34物体类别（二分类先只用一位）
    label[:,:,0:1] = 0

    for l in range(len):
        x1 = int(xmin[l] /4)
        x2 = int(xmax[l] / 4)
        y1 = int(ymin[l] / 4)
        y2 = int(ymax[l] / 4)
        # print(xmin,xmax,ymin,ymax)
        # print(x1,x2,y1,y2)
        x3 = int(x[l] / 4)
        y3 = int(y[l] / 4)
        xoffset = (x[l]%4)/4
        yoffset = (y[l] %4 )/ 4

        label[y3][x3][1] = xoffset
        label[y3][x3][2] = yoffset

        print('质心',x[l], y[l])
        print('offset',xoffset,yoffset)
        print('所在格',x3,y3)

        for i in range(64):
            if i >= y1 and i <= y2:
                for j in range(32):
                    if j >= x1 and j <= x2:
                        label[i][j][1] = 1
                        # label[y3][x3][0] = 1
    return label

if __name__ == '__main__':
    import cv2

    s = 1661224972270
    label = make_label('./data/train/label/%s.xml' % s)
    # print(label.reshape(16, 16))
    imagegt = cv2.imread('./data/train/img/%s.jpg' % s)
    out = label[:,:,1:2]
    offset = label[:,:,1:]
    # print(offset)
    for i in range(64):
        for j in range(64):
            if out[i][j] > 0:
                out[i][j] = 1
                # print(i, j)
                cv2.rectangle(imagegt, (j * 2, i * 2), (j * 2 + 2, i * 2 + 2), (0, 0, 255), 1)
            else:
                out[i][j] = 0
    # print(out)

    x, y, xmin, xmax, ymin, ymax = readxml('./data/train/label/%s.xml' % s)
    for i in range(len(x)):
        cv2.circle(imagegt,(int(x[i]),int(y[i])),1,(0,255,0))

    imagegt = cv2.resize(imagegt, (256, 256))
    cv2.imshow('1', imagegt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

