#!/usr/bin/env python
from __future__ import division
import sys, os
import caffe
import numpy as np
from PIL import Image
from PIL import ImageDraw
import scipy.io
import string
import cv2
import matplotlib.pyplot as plt
from os import listdir

face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
face_cascade.load('./cascades/haarcascade_frontalface_default.xml')


def on_mouse(event, x, y, flags, param):
    param.mouse_cb(event, x, y, flags)


def nothing(x):
    pass


COLOR_NG = (0, 0, 0)
COLOR_FG = (0, 255, 0)


class interactiveFaceDenoising():
    def __init__(self):
        self.winname = "InteractiveFaceDenoising"
        self.img = np.zeros((0))
        # self.mask = np.zeros((0))
        self.left_mouse_down = False
        self.right_mouse_down = False
        # self.cur_mouse = (-1, -1)
        cv2.namedWindow(self.winname)
        cv2.setMouseCallback(self.winname, on_mouse, self)
        # cv2.createTrackbar('brush size', self.winname, self.radius, self.max_radius, nothing)

    def mouse_cb(self, event, x, y, flags):
        global x1, y1, x2, y2
        # self.cur_mouse = (x, y)

        if self.img.size > 0:
            if flags:
                if event == cv2.EVENT_LBUTTONDOWN:
                    x1, y1 = x, y
                    # print(event)
                    # print('flags:', flags)
                # if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
                #     x2, y2 = x, y
                #     cv2.rectangle(self.img, (x1, y1), (x2, y2), COLOR_NG, thickness=1)
                if event == cv2.EVENT_LBUTTONUP:
                    x2, y2 = x, y
                    cv2.rectangle(self.img, (x1, y1), (x2, y2), COLOR_NG, thickness=1)

    def process(self, img):
        self.img = np.copy(img)
        while True:
            show_img = self.img
            cv2.imshow(self.winname, show_img)
            key = cv2.waitKey(100)
            if key == ord('c'):
                self.img = np.copy(img)
            elif key == ord('q') or key == 27 or key == ord('s') or key == ord('p') or key == ord('n') or key == 10:
                break
            elif key == ord('a') or key == 32:
                cv2.putText(show_img, 'denoising...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(self.winname, show_img)
                cv2.waitKey(1)
                img_denoise = img[y1:y2, x1:x2]
                img[y1:y2, x1:x2] = cv2.fastNlMeansDenoising(img_denoise, dst=None, h=10, templateWindowSize=5,
                                                             searchWindowSize=13)
                self.img = np.copy(img)
        return key


def face_parsing(data_path):
    # load image names
    global imi_face
    imNames = [line.rstrip() for line in listdir(data_path)]
    imNames.sort()

    # segment and measure performance
    for imIdx, imName in enumerate(imNames):
        if imName[-3:] == 'jpg':
            imName = imName[:-4]
        print(imName)

        # prepare the image
        imi = cv2.imread(data_path + imName + '.jpg')
        m = imi.shape[0]
        n = imi.shape[1]
        cv2.cvtColor(imi, cv2.COLOR_BGR2RGB)
        # imi = cv2.resize(imi, (370, 450))
        # imi = imi[100:-1, :]
        im = np.array(imi, dtype=np.float32)
        if len(imi.shape) == 2:
            im = np.reshape(im, im.shape + (1,))
            im = np.concatenate((im, im, im), axis=2)
        # im = im[:, :, ::-1]  # RGB to BGR
        imi = cv2.resize(imi, (350, 400))
        imi_org = np.zeros((imi.shape[0], imi.shape[1]))
        faces = face_cascade.detectMultiScale(imi, 1.3, 5)
        for (x, y, w, h) in faces:
            # im_face = im[y - 10: y + h + 10, x - 10: x + w + 10]
            # h = h + 20
            imi_face = imi[y: -1, x: x + w]
            break
        # img2_face = cv2.resize(img2_face, (img_size, img_size))

        # trained with different means (accidently)
        lanIm = imi_face - np.array((122.67, 104.00, 116.67))
        segIm = imi_face - np.array((87.86, 101.92, 133.01))
        # lanIm = im_face
        # segIm = im_face
        lanIm = lanIm.transpose((2, 0, 1))
        segIm = segIm.transpose((2, 0, 1))

        # Forward through the landmark network
        net1.blobs['data'].reshape(1, *lanIm.shape)
        net1.blobs['data'].data[0, :, :, :] = lanIm
        net1.forward()
        H = net1.blobs['score'].data[0]

        # Do some recovery of the points
        C = np.zeros(H.shape, 'uint8')  # cleaned up heatmaps
        C = np.pad(C, ((0, 0), (120, 120), (120, 120)), 'constant')
        print(C.shape)
        Q = np.zeros((68, 2), 'float')  # detected landmarks
        for k in range(0, 68):
            ij = np.unravel_index(H[k, :, :].argmax(), H[k, :, :].shape)
            Q[k, 0] = ij[0]
            Q[k, 1] = ij[1]
            C[k, ij[0] + 120 - 100:ij[0] + 120 + 101, ij[1] + 120 - 100:ij[1] + 120 + 101] = f
        C = C[:, 120:-120, 120:-120] * 0.5

        # Forward through the segmentation network
        D = np.concatenate((segIm, C))
        net2.blobs['data'].reshape(1, *D.shape)
        net2.blobs['data'].data[0, :, :, :] = D
        net2.forward()
        S = net2.blobs['score'].data[0].argmax(axis=0)
        imi_org[y: -1, x: x + w] = S
        imi_org = cv2.resize(imi_org, (n, m))
        imi_seg = Image.fromarray(imi_org.astype(np.uint8))
        imi_seg.putpalette(palette)
        cv2.imwrite('face_parsing/'+imName+'.jpg', imi_org)
        # cv2.imshow('face_landmarks', imi)
        # cv2.imshow('face_pasing', S * 255.0 / 7.0)
        # cv2.waitKey(0)
        # fig = plt.figure()
        # ax1 = fig.add_subplot(121)
        # ax2 = fig.add_subplot(122)
        # ax1.imshow(imi)
        # ax1.scatter(Q[:, 1]+x, Q[:, 0]+y, c='r', s=10)
        # ax2.imshow(imi_seg)
        # plt.show()


if __name__ == '__main__':
    # Comment this out to run the code on the CPU
    caffe.set_mode_gpu()
    caffe.set_device(0)

    # Load both networks
    net1 = caffe.Net('net_landmarks.prototxt',
                     'params_landmarks.caffemodel', caffe.TEST)
    net2 = caffe.Net('net_segmentation.prototxt',
                     'params_segmentation.caffemodel', caffe.TEST)

    data_path = "data/cuhk/train/photo/"

    # Generate the colourmap for the segmentation mask
    palette = np.zeros((255, 3))
    palette[0, :] = [0, 0, 0]  # Background
    palette[1, :] = [255, 184, 153]  # Skin
    palette[2, :] = [112, 65, 57]  # Eyebrows
    palette[3, :] = [51, 153, 255]  # Eyes
    palette[4, :] = [219, 144, 101]  # Nose
    palette[5, :] = [135, 4, 0]  # Upper lip
    palette[6, :] = [67, 0, 0]  # Mouth
    palette[7, :] = [135, 4, 0]  # Lower lip
    palette = palette.astype('uint8').tostring()

    # We have a Gaussian to recover the output slightly - better results
    f = scipy.io.loadmat('gaus.mat')['f']
    face_parsing(data_path)
    # ifd = interactiveFaceDenoising()
    # key = ifd.process(img)
