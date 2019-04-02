# coding:utf-8

from __future__ import print_function

import numpy as np
import pylab
import matplotlib.pyplot as plt
import cv2
from cv2 import dnn
import time


cm_path = 'C:\\Users\\admin\\Desktop\\'



if __name__ == "__main__":

    fn = r'C:\Users\admin\Desktop\ShanghaiTech_Crowd_Counting_Dataset\part_B_final\test_data\images\IMG_191.jpg'

    im_ori = cv2.imread(fn)
    plt.figure(1)
    plt.imshow(im_ori)
    plt.axis('off')
    pylab.show()


    blob = dnn.blobFromImage(im_ori, 1, (1280, 720), (0, 0, 0), True)

    print("Input:", blob.shape, blob.dtype)

    net = dnn.readNetFromCaffe(cm_path + 'B_testdemo.prototxt', cm_path + 'B2_iter_93000.caffemodel')

    t = time.time()
    net.setInput(blob)
    density = net.forward()
    elapsed = time.time() - t

    print('inference image: %.4f seconds.' % elapsed)

    density = density/1000.0

    print("Output:", density.shape, density.dtype)

    person_num = np.sum(density[:])
    print("number: ",person_num)

    plt.figure(1)
    plt.imshow(density[0, 0, :, :])
    plt.axis('off')
    pylab.show()
