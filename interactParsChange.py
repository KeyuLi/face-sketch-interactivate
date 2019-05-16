from __future__ import division
import sys, os
import numpy as np
from PIL import Image
from PIL import ImageDraw
import scipy.io
import string
import cv2
import matplotlib.pyplot as plt
from os import listdir
from skimage.filters import gaussian
import torch
from torchvision.models import vgg16
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from networks import define_G
from loss import *
from get_data_list import get_data_list

vgg = vgg16(pretrained=True)
vgg_percept1 = nn.Sequential(*list(vgg.features)[:31]).eval().cuda()
for param in vgg_percept1.parameters():
    param.requires_grad = False
netG_A = define_G(3, 3, 64, 'batch', False, [0]).cuda()

transform_list = [transforms.ToTensor()]
transform = transforms.Compose(transform_list)

# face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
# face_cascade.load('./cascades/haarcascade_frontalface_default.xml')
kernel = np.array([[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1],
                   [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1], [-1, -1, -1, -1, -1]]) / 8.0


def usm(img):
    gauss_out = gaussian(img, sigma=2, multichannel=None)
    gauss_out = gauss_out * 255.0
    alpha = 0.7
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = img_out * 255.0
    return img_out


def on_mouse(event, x, y, flags, param):
    param.mouse_cb(event, x, y, flags)


def nothing(x):
    pass


COLOR_NG = (0, 0, 0)
COLOR_FG = (0, 255, 0)


class interactiveFaceChanging():
    def __init__(self):

        self.winname = "InteractiveFaceChanging"
        self.img_face = np.zeros((0))
        self.img_parsing = np.zeros((0))

        # self.mask = np.zeros((0))
        self.left_mouse_down = False
        self.right_mouse_down = False
        # self.cur_mouse = (-1, -1)
        cv2.namedWindow(self.winname)
        cv2.setMouseCallback(self.winname, on_mouse, self)
        # cv2.createTrackbar('brush size', self.winname, self.radius, self.max_radius, nothing)

    def mouse_cb(self, event, x, y, flags):
        global x1, y1, x2, y2, flag
        # self.cur_mouse = (x, y)

        if self.img_face.size > 0:
            if flags:
                if flags == (cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_LBUTTONDOWN) and event == cv2.EVENT_LBUTTONDOWN:
                    x1, y1 = x, y
                    flag = 0
                if flags == (cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_LBUTTONDOWN) and event == cv2.EVENT_LBUTTONDOWN:
                    x1, y1 = x, y
                    flag = 1
                    # print(event)
                    # print('flags:', flags)
                # if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
                #     x2, y2 = x, y
                #     cv2.rectangle(self.img, (x1, y1), (x2, y2), COLOR_NG, thickness=1)
                if event == cv2.EVENT_LBUTTONUP:
                    x2, y2 = x, y
                    cv2.rectangle(self.img_face, (x1, y1), (x2, y2), (0, 0, 0), thickness=1)

    def process(self, img_face, img_parsing, img_percept, photo_list, sketch_list):
        netG_A = torch.load("checkpoint/cuhk/cycle_gan/netG_A_model_epoch_300.pth").cuda()
        self.img_parsing = np.copy(img_parsing)
        self.img_face = np.copy(img_face)
        while True:
            show_img = self.img_face
            cv2.imshow(self.winname, show_img)
            key = cv2.waitKey(100)
            if key == ord('c'):
                self.img_face = np.copy(img_face)
            elif key == ord('q') or key == 27 or key == ord('s') or key == ord('p') or key == ord('n') or key == 10:
                break
            elif key == ord('a') or key == 32:
                cv2.putText(show_img, 'Changing...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(self.winname, show_img)
                cv2.waitKey(1)
                # img_style = np.copy(face_style)[y1:y2, x1:x2]
                # img_style = cv2.resize(img_style, (256, 256))
                # cv2.imshow('img_style', img_style)
                # cv2.waitKey(0)
                # img_style = np.transpose(img_style, (2, 0, 1))
                # img_style = img_style.reshape(
                #     (1, img_style.shape[0], img_style.shape[1], img_style.shape[2]))
                # img_style = torch.cuda.FloatTensor(img_style)

                img_change = np.copy(img_percept)[y1:y2, x1:x2]
                img_original = np.copy(img_face)[y1:y2, x1:x2]
                from neighbor_select_simple import neighbor_select_simple
                candidates = neighbor_select_simple(img_change, photo_list, sketch_list, y1, y2, x1, x2)
                from NeuralStyleTransfer import style_transfer
                # output = style_transfer(candidates, 500, (128, 128))
                # cv2.imshow('img_output', output)
                # cv2.waitKey()
                m, n, _ = img_change.shape
                # output = cv2.resize(output, (n, m))

                # img_change = cv2.resize(img_change, (256, 256))
                # cv2.imshow('img_denosie', img_change)
                # cv2.waitKey(0)
                # img_change = np.transpose(img_change, (2, 0, 1))
                # img_change = img_change.reshape(
                #     (1, img_change.shape[0], img_change.shape[1], img_change.shape[2]))
                # img_change = torch.cuda.FloatTensor(img_change)
                img_parse = img_parsing[y1:y2, x1:x2]
                if flag == 0:
                    output = style_transfer(candidates, 300, (128, 128))
                    output = cv2.resize(output, (n, m))
                    cv2.imshow('img_output', output)
                    cv2.waitKey()
                    img_com = np.zeros((img_parse.shape[0], img_parse.shape[1], img_parse.shape[2]))
                    img_gro = np.zeros((img_parse.shape[0], img_parse.shape[1], img_parse.shape[2]))
                    img_skin = np.zeros((img_parse.shape[0], img_parse.shape[1], img_parse.shape[2]))
                    # img_com present the component of the face
                    # img_skin ---------------------skin
                    # img_gro   ---------------------background
                    for i in range(img_parse.shape[0]):
                        for j in range(img_parse.shape[1]):
                            if (img_parse[i][j] == 0).all():
                                img_com[i][j] = 0
                                img_gro[i][j] = 1
                                img_skin[i][j] = 0
                            if (img_parse[i][j] == 1).all():
                                img_com[i][j] = 0
                                img_gro[i][j] = 0
                                img_skin[i][j] = 1
                            if (img_parse[i][j] != 0).all() and (img_parse[i][j] != 1).all():
                                img_com[i][j] = 1
                                img_gro[i][j] = 0
                                img_skin[i][j] = 0

                    # img_percept = transform(img_percept).cuda()
                    # for i in range(10):
                    # print(vgg_percept1(img_style)[0, :, :, :].shape)
                    # print(gram_matrix(vgg_percept1(img_style)[0, :, :, :]).shape)
                    # percept_loss = mseLoss(gram_matrix(vgg_percept1(netG_A(img_change))[0, :, :, :]),
                    #                        gram_matrix(vgg_percept1(img_style)[0, :, :, :]))
                    # print('epoch:{:.1f}, percept_loss:{:.4f}'.format(i, percept_loss))
                    # content_loss = mseLoss(netG_A(img_change), img_change)
                    # print('epoch:{:.1f},content_loss:{:.4f}'.format(i, content_loss))
                    # loss = 0.4*percept_loss + 0.6*content_loss
                    # optimzer_G.zero_grad()
                    # loss.backward()
                    # optimzer_G.step()

                    # img_newpercept = netG_A(img_change).cpu().data.numpy().reshape((3, 256, 256))
                    # img_newpercept = cv2.resize(np.transpose(img_newpercept, (1, 2, 0)), (n, m))
                    # img_newpercept = (cv2.cvtColor(img_newpercept, cv2.COLOR_BGR2GRAY) * 255.0).astype(np.uint8)
                    # cv2.imshow('img_newpercept', img_newpercept)
                    # cv2.waitKey(0)

                    # img_face[y1:y2, x1:x2] = cv2.fastNlMeansChanging(img_original, dst=None, h=10,
                    # templateWindowSize=5, searchWindowSize=13) * img_skin + img_original * img_gro + output * img_com
                    img_face[y1:y2, x1:x2] = 0.6*img_face[y1:y2, x1:x2] + 0.4*output
                    self.img_face = np.copy(img_face)
                if flag == 1:
                    img_com = np.copy(img_parse)
                    img_gro = np.copy(img_parse)
                    img_skin = np.copy(img_parse)
                    for i in range(img_parse.shape[0]):
                        for j in range(img_parse.shape[1]):
                            if (img_parse[i][j] == 0).all():
                                img_com[i][j] = 0
                                img_gro[i][j] = 1
                                img_skin[i][j] = 0
                            if (img_parse[i][j] == 1).all():
                                img_com[i][j] = 0
                                img_gro[i][j] = 0
                                img_skin[i][j] = 1
                            else:
                                img_com[i][j] = 1
                                img_gro[i][j] = 0
                                img_skin[i][j] = 0
                    img_face[y1:y2, x1:x2] = cv2.fastNlMeansChanging(img_original, dst=None, h=10,
                                                                      templateWindowSize=5, searchWindowSize=13)
                    # img_face[y1:y2, x1:x2] = output
                    self.img_face = np.copy(img_face)
        return key


def interactive(face_path, parsing_path, photo_list, sketch_list):
    imgNames = os.listdir(face_path)
    imgNames.sort()
    for imgName in imgNames:
        img_percept = cv2.imread(face_path + imgName)
        img_parsing = cv2.imread(parsing_path + imgName)
        # img_percept = cv2.resize(img_percept, (256, 256))
        # img_parsing = cv2.resize(img_parsing, (256, 256))
        # if len(img_face.shape) == 3:
        #     img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
        # if len(img_parsing.shape) == 3:
        #     img_parsing = cv2.cvtColor(img_parsing, cv2.COLOR_BGR2GRAY)
        img_face = cv2.resize(img_percept, (256, 256)).transpose((2, 0, 1))
        img_face = img_face.reshape(
            (1, img_face.shape[0], img_face.shape[1], img_face.shape[2]))
        img_face = torch.cuda.FloatTensor(img_face)
        img_face = np.transpose(netG_A(img_face).cpu().data.numpy().reshape((3, 256, 256)), (1, 2, 0))
        img_face = (img_face.copy() * 255).astype(np.uint8)
        img_face = cv2.resize(img_face, (200, 250))
        # print(img_face.shape, img_parsing.shape)
        itd = interactiveFaceChanging()
        itd.process(img_face, img_parsing, img_percept, photo_list, sketch_list)


if __name__ == '__main__':
    mseLoss = nn.MSELoss()
    face_path = 'data/cuhk/test/photo/'
    parsing_path = 'face_parsing/'
    netG_A = torch.load("checkpoint/cuhk/cycle_gan/netG_A_model_epoch_300.pth").cuda()
    optimzer_G = optim.Adam(netG_A.parameters(), lr=1e-3, betas=(0.5, 0.999))
    face_style = cv2.resize(cv2.imread('1.jpg'), (200, 250))
    data_dir = 'data/cuhk/train/'
    photo_list, sketch_list = get_data_list(data_dir)
    interactive(face_path, parsing_path, photo_list, sketch_list)
