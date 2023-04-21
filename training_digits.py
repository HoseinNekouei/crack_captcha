import cv2 as cv
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import models, layers
import matplotlib.pyplot as plt

BATCH_SIZE = 32
EPOCHS = 10


def load_data(infoPath):
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

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, x_test, y_train, y_test


def neural_network():
    # make sequential network
    net = models.Sequential([
                            layers.Dense(64, activation='relu'),
                            layers.Dense(32, activation='relu'),
                            layers.Dense(9, activation='softmax')
                            ])

    net.summary

    # Determination of parameters
    net.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics='accuracy')

    H = net.fit(x_train, y_train, batch_size=BATCH_SIZE,
                epochs=EPOCHS, validation_data=(x_test, y_test))

    loss, accuracy = net.evaluate(x_test, y_test)
    print('loss: {:.2f} , accuracy: {:.2f}'.format(loss, accuracy))

    net.save('crack_captcha\\model.h5')
    return H


def show_results():
    plt.style.use('ggplot')
    plt.plot(np.arange(EPOCHS),H.history['loss'],label='Train Loss')
    plt.plot(np.arange(EPOCHS),H.history['val_loss'], label='Test Loss')
    plt.plot(np.arange(EPOCHS),H.history['accuracy'],label= 'Train Accuracy')
    plt.plot(np.arange(EPOCHS),H.history['val_accuracy'],label='val_accuracy')

    plt.legend()

    plt.xlabel('EPOTCHS')
    plt.ylabel('Loss/Accuracy')
    plt.title('Traing Digit Model')
    plt.show()


x_train, x_test, y_train, y_test = load_data(
    'crack_captcha\\captcha\\*\\*')
H = neural_network()
show_results()