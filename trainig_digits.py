import cv2 as cv
import numpy as np
import glob
from sklearn.model_selection import train_test_split


def preprocessing(infoPath):
    data = []
    labels = []

    for i, item in enumerate(glob.glob(infoPath)):
        img = cv.imread(item, 0)
        # gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (32, 32)).flatten()
        img = img / 255

        data.append(img)
        label = item.split('\\')[-2]

        labels.append(label)

        if i % 100 == 0:
            print('[INFO] {}/2000 processed'.format(i))

    data = np.array(data)

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42)

    print(x_train.shape)
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = preprocessing(
    'crack_captcha\\captcha\\*\\*')
