import numpy as np
from PIL import Image
import cv2

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = cv2.imread(filepath)
    # img = Image.open(filepath).convert('RGB')
    # img = img.resize((256, 256), Image.BICUBIC)
    img = cv2.resize(img, (256, 256))
    return img

# --------------------------------------------------
def load_img1(filepath):
    img = cv2.imread(filepath)
    img = img[25:225, 23:178]
    img = cv2.resize(img, (256, 256))
    return img

def load_img2(filepath):
    img = cv2.imread(filepath)
    # img = cv2.imread(filepath)
    img = img[25:225, 23:178]
    img = cv2.resize(img, (256, 256))
    # img = img.reshape(188, 143, 1)
    # img = cv2.resize(img, (256, 256))
    # cv2.imshow('img', img)
    # cv2.waitKey()
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = cv2.resize(image_numpy, (200, 250))
    cv2.imwrite(filename, image_pil)
    # image_pil = Image.fromarray(image_numpy)
    # image_pil = image_pil.resize((200, 250), Image.BICUBIC)
    # image_pil.save(filename)
    print("Image saved as {}".format(filename))


