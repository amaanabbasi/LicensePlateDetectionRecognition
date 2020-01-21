from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import scipy.ndimage
import numpy as np
import cv2
import os

def binarization_(gray_img):
    ret,thresh1 = cv2.threshold(gray_img,127,255,0)
    return thresh1

def complemented_img(img):
    img_comp = cv2.bitwise_not(img)
    # plt.imshow(img_comp, cmap="gray")
    return img_comp

def histogram_of_pixel_projection(img):
    """
    This method is responsible for licence plate segmentation with histogram of pixel projection approach
    :param img: input image
    :return: list of image, each one contain a digit
    """
    # list that will contains all digits
    character_list_image = list()

    # img = crop(img)
    
    # Compliment
    img = complemented_img(img)
    
    # Add black border to the image
    BLACK = [0, 0, 0]
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=BLACK)

    # change to gray
    # import pdb; pdb.set_trace()
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     cv2.imshow("gray", gray)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    else:
        gray = img.copy()
    # Change to numpy array format

    # Binarization
    nb = binarization_(gray)
    
    nb = np.array(nb)
    
    # compute the sommation
    x_sum = cv2.reduce(nb, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    y_sum = cv2.reduce(nb, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

    # rotate the vector x_sum
    x_sum = x_sum.transpose()

    # get height and weight
    x = gray.shape[1]
    y = gray.shape[0]

    # division the result by height and weight
    x_sum = x_sum / y
    y_sum = y_sum / x

    # x_arr and y_arr are two vector weight and height to plot histogram projection properly
    x_arr = np.arange(x)
    y_arr = np.arange(y)

    # convert x_sum to numpy array
    z = np.array(x_sum)

    # convert y_arr to numpy array
    w = np.array(y_sum)

    # convert to zero small details
    z[z < 15] = 0
    z[z > 15] = 1

    # convert to zero small details and 1 for needed details
    w[w < 20] = 0
    w[w > 20] = 1

    # vertical segmentation
    test = z.transpose() * nb

    # horizontal segmentation
    test = w * test

    
    # plot histogram projection result using pyplot
    horizontal = plt.plot(w, y_arr)
    vertical = plt.plot(x_arr ,z)

    # plt.show(horizontal)
    # plt.show(vertical)

    f = 0
    ff = z[0]
    t1 = list()
    t2 = list()
    for i in range(z.size):
        if z[i] != ff:
            f += 1
            ff = z[i]
            t1.append(i)
    rect_h = np.array(t1)

    f = 0
    ff = w[0]
    for i in range(w.size):
        if w[i] != ff:
            f += 1
            ff = w[i]
            t2.append(i)
    rect_v = np.array(t2)

    # import pdb; pdb.set_trace()
    # take the appropriate height
    rectv = []
    rectv.append(rect_v[0])
    rectv.append(rect_v[1])
    max = int(rect_v[1]) - int(rect_v[0])
    for i in range(len(rect_v) - 1):
        diff2 = int(rect_v[i + 1]) - int(rect_v[i])

        if diff2 > max:
            rectv[0] = rect_v[i]
            rectv[1] = rect_v[i + 1]
            max = diff2

    
    # extract caracter
    for i in range(len(rect_h) - 1):

        # eliminate slice that can't be a digit, a digit must have width bigger then 8
        diff1 = int(rect_h[i + 1]) - int(rect_h[i])

        if (diff1 > 5) and (z[rect_h[i]] == 1):
            # cutting nb (image) and adding each slice to the list character_list_image
            character_list_image.append(nb[int(rectv[0]):int(rectv[1]), rect_h[i]:rect_h[i + 1]])

            # draw rectangle on digits
            cv2.rectangle(img, (rect_h[i], rectv[0]), (rect_h[i + 1], rectv[1]), (0, 255, 0), 1)

    # return segmentation result


    return character_list_image, img

def segment_characters(img, image_path):
    character_list_image, image = histogram_of_pixel_projection(img)
    image_name = image_path.split('.')
    # plt.imshow(image)
    # plt.savefig(os.path.join('results' , image_name[0] + "-segmented."+ image_name[1]))
    for i in range(len(character_list_image)):
        cv2.imwrite(os.path.join('results','segmented', str(i) + "segmented."+ image_name[1]), character_list_image[i])
    cv2.imwrite(os.path.join('results' , image_name[0] + "-segmented."+ image_name[1]), image)
    return character_list_image