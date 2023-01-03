from PIL import Image
import numpy as np
import os
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import cv2 
import random
from math import log

def getImageMatrix(imageName):
    im = Image.open(imageName) 
    pix = im.load()
    color = 1
    if type(pix[0,0]) == int:
      color = 0
    image_size = im.size 
    image_matrix = []
    for width in range(int(image_size[0])):
        row = []
        for height in range(int(image_size[1])):
                row.append((pix[width,height]))
        image_matrix.append(row)
    return image_matrix, image_size[0], image_size[1],color


def getImageMatrix_gray(imageName):
    im = Image.open(imageName).convert('LA')
    pix = im.load()
    image_size = im.size 
    image_matrix = []
    for width in range(int(image_size[0])):
        row = []
        for height in range(int(image_size[1])):
                row.append((pix[width,height]))
        image_matrix.append(row)
    return image_matrix, image_size[0], image_size[1]

def ArnoldCatTransform(img, num):
    rows, cols, ch = img.shape
    n = rows
    img_arnold = np.zeros([rows, cols, ch])
    for x in range(0, rows):
        for y in range(0, cols):
            img_arnold[x][y] = img[(x+y)%n][(x+2*y)%n]  
    return img_arnold    

def ArnoldCatEncryption(imageName, key):
    img = cv2.imread(imageName)
    for i in range (0,key):
        img = ArnoldCatTransform(img, i)
    # cv2.imwrite(imageName.split('.')[0] + "_ArnoldcatEnc.png", img)
    return img

def ArnoldCatDecryption(imageName, key):
    img = cv2.imread(imageName)
    rows, cols, ch = img.shape
    dimension = rows
    decrypt_it = dimension
    if (dimension%2==0) and 5**int(round(log(dimension/2,5))) == int(dimension/2):
        decrypt_it = 3*dimension
    elif 5**int(round(log(dimension,5))) == int(dimension):
        decrypt_it = 2*dimension
    elif (dimension%6==0) and  5**int(round(log(dimension/6,5))) == int(dimension/6):
        decrypt_it = 2*dimension
    else:
        decrypt_it = int(12*dimension/7)
    for i in range(key,decrypt_it):
        img = ArnoldCatTransform(img, i)
    # cv2.imwrite(imageName.split('_')[0] + "_ArnoldcatDec.png",img)
    return img

image = "original"
ext = ".png"
key = 5

# img = cv2.imread(image + ext)
# print(img)
# cv2.imshow( "FRAME", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)


ArnoldCatEncryptionIm = ArnoldCatEncryption(image + ext, key)
cv2.imwrite("encrypted.png",ArnoldCatEncryptionIm)
# pil_im1 = Image.open("encrypted.png", 'r')
# imshow(np.asarray(pil_im1))


ArnoldCatDecryptionIm = ArnoldCatDecryption("encrypted.png", key)
cv2.imwrite("Decrypted.png",ArnoldCatDecryptionIm)
# pil_im2 = Image.open("Decrypted.png", 'r')
# imshow(np.asarray(pil_im2))

# print(readEncrypted)
# cv2.imshow( readEncrypted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# image = "original"
# ext = ".png"
# img = cv2.imread(image + ext,1) 
# pil_im = Image.open(image + ext, 'r')
# imshow(np.asarray(pil_im))

# plt.figure(figsize=(14,6))
# histogram_blue = cv2.calcHist([img],[0],None,[256],[0,256])
# plt.plot(histogram_blue, color='blue') 
# histogram_green = cv2.calcHist([img],[1],None,[256],[0,256]) 
# plt.plot(histogram_green, color='green') 
# histogram_red = cv2.calcHist([img],[2],None,[256],[0,256]) 
# plt.plot(histogram_red, color='red')
# plt.title('Intensity Histogram - Original Image', fontsize=20)
# plt.xlabel('pixel values', fontsize=16)
# plt.ylabel('pixel count', fontsize=16) 
# plt.show()

# image = "encrypted"
# ext = ".png"
# img = cv2.imread(image + ext,1) 
# pil_im = Image.open(image + ext, 'r')
# imshow(np.asarray(pil_im))

# plt.figure(figsize=(14,6))
# histogram_blue = cv2.calcHist([img],[0],None,[256],[0,256])
# plt.plot(histogram_blue, color='blue') 
# histogram_green = cv2.calcHist([img],[1],None,[256],[0,256]) 
# plt.plot(histogram_green, color='green') 
# histogram_red = cv2.calcHist([img],[2],None,[256],[0,256]) 
# plt.plot(histogram_red, color='red') 
# plt.title('Intensity Histogram - Arnold Cat Encrypted', fontsize=20)
# plt.xlabel('pixel values', fontsize=16)
# plt.ylabel('pixel count', fontsize=16) 
# plt.show()


# image = "Decrypted"
# ext = ".png"
# img = cv2.imread(image + ext,1) 
# pil_im = Image.open(image + ext, 'r')
# imshow(np.asarray(pil_im))

# plt.figure(figsize=(14,6))
# histogram_blue = cv2.calcHist([img],[0],None,[256],[0,256])
# plt.plot(histogram_blue, color='blue') 
# histogram_green = cv2.calcHist([img],[1],None,[256],[0,256]) 
# plt.plot(histogram_green, color='green') 
# histogram_red = cv2.calcHist([img],[2],None,[256],[0,256]) 
# plt.plot(histogram_red, color='red') 
# plt.title('Intensity Histogram - Arnold Cat Encrypted', fontsize=20)
# plt.xlabel('pixel values', fontsize=16)
# plt.ylabel('pixel count', fontsize=16) 
# plt.show()