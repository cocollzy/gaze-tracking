import configparser
import numpy as np
# from glob import glob
from os import listdir
import cv2
from sklearn.model_selection import train_test_split
# from model import model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error


from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Concatenate
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
# from keras import backend as K

def get_model(input_shape=(40, 200, 3)):

    X_input = Input(input_shape)
    X = Conv2D(20, (5, 5), activation='relu')(X_input)
    X = MaxPooling2D((2,2))(X)
    X = Flatten()(X)
    X = Dropout(0.1)(X)
    X = Dense(2)(X)

    model = Model(inputs=[X_input], outputs=[X])
    return model



def get_model2(input_shape=(40, 80, 3)):

    X_left_input = Input(input_shape)
    X_right_input = Input(input_shape)

    X_l = Conv2D(8, (5, 5), activation='relu')(X_left_input)
    X_l = MaxPooling2D((2,2))(X_l)
    X_l = Flatten()(X_l)
    X_l = Dropout(0.1)(X_l)

    X_r = Conv2D(8, (5, 5), activation='relu')(X_right_input)
    X_r = MaxPooling2D((2,2))(X_r)
    X_r = Flatten()(X_r)
    X_r = Dropout(0.1)(X_r)

    X = Concatenate()([X_l, X_r])
    X = Dense(10)(X)
    X = Dropout(0.1)(X)
    X = Dense(2)(X)

    model = Model(inputs=[X_left_input, X_right_input], outputs=[X])
    return model

def get_model_sharedLayer(input_shape=(40, 80, 3)):
    X_left_input = Input(input_shape)
    X_right_input = Input(input_shape)

    conv1 = Conv2D(8, (5, 5), activation='relu')
    conv2 = Conv2D(20, (5, 5), activation='relu')
#     conv3 = Conv2D(32, (3, 3), activation='relu')
    dropout = Dropout(0.1)

    X_l = conv2(X_left_input)
    X_r = conv2(X_right_input)
    X_l = MaxPooling2D((2,2))(X_l)
    X_r = MaxPooling2D((2,2))(X_r)
#     X_l = conv2(X_l)
#     X_r = conv2(X_r)
#     X_l = conv3(X_l)
#     X_r = conv3(X_r)

    X_l = Flatten()(X_l)
#     X_l = dropout(X_l)

    X_r = Flatten()(X_r)
#     X_r = dropout(X_r)

    X = Concatenate()([X_l, X_r])
    X = Dense(2)(X)

    model = Model(inputs=[X_left_input, X_right_input], outputs=[X])

    return model


def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]


config = configparser.ConfigParser()
config.read('./config.ini')
input_dim = eval(config['GAZE']['Input_dim'])

data_dir = "/Users/liuziyiliu/Desktop/gaze_tracking/data"
eye_left_files = mylistdir(data_dir + '/left')
eye_right_files = mylistdir(data_dir + '/right')
model_fname = config['GAZE']['Model_fname']
gray = True
if gray:
    eye_left_images = np.zeros([len(eye_left_files), input_dim[0], input_dim[1], 1], dtype=np.int)
    eye_right_images = np.zeros([len(eye_left_files), input_dim[0], input_dim[1], 1], dtype=np.int)
    labels = np.zeros([len(eye_left_files), 2])
    for i, data_file in enumerate(eye_left_files):
        print(data_dir + '/left/' + data_file)
        img_l = cv2.imread(data_dir + '/left/' + data_file)
        img_r = cv2.imread(data_dir + '/right/' + data_file)
        eye_left_images[i,:,:,0] = np.array(cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY))
        eye_right_images[i,:,:,0] = np.array(cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY))

        label = data_file.split('-')[-1].split('.')[0].split('_')
        labels[i] = np.array((label[0], label[1]))
        print(label[0], label[1])


x_left_train, x_left_val, x_right_train, x_right_val, y_train, y_val = train_test_split(eye_left_images,
                                                                                         eye_right_images,
                                                                                         labels, test_size=0.1, shuffle=True)

model = get_model_sharedLayer((40, 80, 1))


model.compile(optimizer=Adam(0.005), loss='mse',  metrics=['accuracy'])
# training

msave = ModelCheckpoint(model_fname, save_best_only=True)

history = model.fit([x_left_train, x_right_train], y_train,
                    validation_data=([x_left_val, x_right_val], y_val),
                    epochs=100,
                    callbacks = [msave],
                    batch_size=4,
                    verbose=1)

# evaluate the model
_, train_acc = model.evaluate([x_left_train, x_right_train], y_train, verbose=0)
_, test_acc = model.evaluate([x_left_val, x_right_val], y_val, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
