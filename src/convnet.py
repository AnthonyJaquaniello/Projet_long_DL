import keras
import numpy as np

from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Flatten, Dropout
from keras import Input, Model
from keras.layers import add, Activation
from keras.layers import Conv3D, MaxPool3D
from keras.callbacks.callbacks import EarlyStopping, Callback
from keras.models import Sequential
from os import listdir

PATH_TRAIN = "../data/x_train"
PATH_TEST = "../data/x_test"

def load_x(path):
    X = [np.load("{}/{}".format(path,pocket))
         for pocket in listdir(path)]
    X = [np.squeeze(array) for array in X]
    X = np.array(X)
    X = np.moveaxis(X, 1, -1)
    return X

def load_y(path, nucleotid, heme, steroid, control):
    Y = []
    for pocket in listdir(path):
        if pocket in nucleotid:
            Y.append(1)
        elif pocket in heme:
            Y.append(2)
        elif pocket in steroid:
            Y.append(3)
        elif pocket in control:
            Y.append(4)
    Y  = np.array(Y)
    return Y

def one_hot_encoding(y):
    classes = LabelEncoder()
    integer_encoding = classes.fit_transform(y)
    one_hot_Y = keras.utils.to_categorical(integer_encoding)
    return one_hot_Y

def list_generator(file):
    with open(file, "r") as filin:
        liste = ["{}.npy".format(line[:-1]) for line in filin]
    return liste
    
#X_train est une liste d'array de dimensions (1,14,32,32,32)
#Il est constitué des 346 premières poches
nucleotid = list_generator("nucleotide.list.txt")
heme = list_generator("heme.list.txt")
steroid = list_generator("steroid.list.txt")
control = list_generator("control.list.txt")
X_train = load_x(PATH_TRAIN)
X_test = load_x(PATH_TEST)
Y_train = load_y(PATH_TRAIN, nucleotid, heme, steroid, control)
Y_test = load_y(PATH_TEST, nucleotid, heme, steroid, control)
one_hot_Y_train = one_hot_encoding(Y_train)
one_hot_Y_test = one_hot_encoding(Y_test)
print(X_train.shape)
print(X_test.shape)
print(one_hot_Y_train.shape)
print(one_hot_Y_test.shape)

model = Sequential()
model.add(Conv3D(filters = 24, kernel_size = 12, strides=1, padding= "same", activation = "relu", kernel_initializer="he_normal"))
#model.add(Conv3D(filters=64, kernel_size = 26, strides=1, padding= "same", activation = "relu", kernel_initializer="he_normal"))
#model.add(Dropout(0.4))
model.add(MaxPool3D(pool_size = 2, strides = 1, padding = "same"))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(units = 50, activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(units = 4, activation = "softmax"))

cb = Callback()
critor = EarlyStopping(monitor = "val_loss", patience = 5)
cb_list = [critor]

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])
model.fit(X_train[0:75,:,:,:,:],one_hot_Y_train[0:75,:],epochs=5, batch_size = 15,
          validation_data = (X_test[0:35,:,:,:,:],one_hot_Y_test[0:35,:]), callbacks = cb_list)

evaluation = model.evaluate(X_train, one_hot_Y_train)
print(evaluation)
