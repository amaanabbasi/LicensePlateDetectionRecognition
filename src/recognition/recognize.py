from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt


from keras.models import model_from_json


def preprocess_img(img, flag=0):
    """ 
    Takes in a character image, convert to gray, 28x28, add dimensions acc to
    model input.
    """
    # print(img.shape)

    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img= np.array(img)
    img = cv2.resize(img, (28, 28))
    if flag:
            img[img < 15] = 0
            img[img > 15] = 1
    # img = np.expand_dims(img, axis=0)
    # img = np.expand_dims(img, axis=4)

    return img


def recognize_chracters(segmented_characters, t_name, t_value):
    t = {}
    detected_plate = []

    j = 0
    for segmented in segmented_characters:
        segmented = preprocess_img(segmented, 1)
        cv2.imwrite("segmented/" + str(j) + ".jpg", segmented)
        j += 1
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        for i in range(len(t_name)):
            template = t_value[i]
            template = preprocess_img(template, 0)
            diff = (template - segmented).mean()
            t[t_name[i]] = diff
        detected_plate.append(min(t, key=t.get))

	
    return detected_plate
