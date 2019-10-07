import keras
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.layers import Dense, Flatten, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks.callbacks import EarlyStopping, Callback
from keras.layers import add, Activation
from keras.layers import Conv3D, MaxPool3D
from keras.models import Sequential
from os import listdir

PATH_TRAIN = "../data/x_train"
PATH_TEST = "../data/x_test"

def load_x(path, control):
    X = [np.load("{}/{}".format(path,pocket))
         for pocket in listdir(path) 
         if pocket not in control]
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
            continue 
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
X_train = load_x(PATH_TRAIN, control)
X_test = load_x(PATH_TEST, control)
Y_train = load_y(PATH_TRAIN, nucleotid, heme, steroid, control)
Y_test = load_y(PATH_TEST, nucleotid, heme, steroid, control)
one_hot_Y_train = one_hot_encoding(Y_train)
one_hot_Y_test = one_hot_encoding(Y_test)
print(X_train.shape)
print(X_test.shape)
print(one_hot_Y_train.shape)
print(one_hot_Y_test.shape)

def model_one():
    model = Sequential()
    model.add(Conv3D(filters = 21, kernel_size = 9, strides=1, padding= "same", activation = "relu", kernel_initializer="he_normal"))
    #model.add(Conv3D(filters=64, kernel_size = 26, strides=1, padding= "same", activation = "relu", kernel_initializer="he_normal"))
    model.add(Dropout(0.4))
    model.add(MaxPool3D(pool_size = 2, strides = 1, padding = "same"))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(units = 50, activation = "relu"))
    model.add(Dropout(0.4))
    model.add(Dense(units = 3, activation = "softmax"))
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])
    return model
    
cb = Callback()
critor = EarlyStopping(monitor = "val_loss", patience = 2)
cb_list = [critor]
my_model = model_one()

my_model.fit(X_train[0:75:,:,:,:],one_hot_Y_train[0:75,:],epochs=10, batch_size = 15,
             validation_data = (X_test[0:25,:,:,:,:],one_hot_Y_test[0:25,:]), callbacks = cb_list)
          
evaluation = my_model.evaluate(X_train, one_hot_Y_train)
print(evaluation)

tp = 0
fp = 0
tn = 0
fn = 0

for i in range(predictions.shape[0]):
    maxi = max(predictions[i,:])
    if maxi == predictions[i, 0]:
        classe = 0
    elif maxi == predictions[i,1]:
        classe = 1
    elif maxi == predictions[i,2]:
        classe = 2
        
    if (one_hot_Y_test[i, 0] == 1.0) and (classe == 0):
        tp += 1
    elif (one_hot_Y_test[i, 1] == 1.0) and (classe == 1):
        tp += 1
    elif (one_hot_Y_test[i, 0] == 0.0) and (classe == 0):
        fp += 1
    elif (one_hot_Y_test[i, 1] == 0.0) and (classe == 1):
        fp += 1
    elif (one_hot_Y_test[i, 2] == 1.0) and (classe == 2):
        tn += 1
    elif (one_hot_Y_test[i, 2] == 0.0) and (classe == 2):
        fn += 1
        
from math import sqrt

print("TP:{:.2f}%".format(tp*100/len(predictions)))
print("FP:{:.2f}%".format(fp*100/len(predictions)))
print("TN:{:.2f}".format(tn*100/len(predictions)))
print("FN:{:.2f}".format(fn*100/len(predictions)))
print("ACC = {:.2f}%".format((tp+tn)*100/(tp+tn+fp+fn)))
print("PPV = {:.2f}%".format(tp*100/(tp+fp)))
print("TNR = {:.2f}%".format(tn*100/(tn+fp)))
print("TPR = {:.2f}%".format(tp*100/(tp+fn)))
print("FPR = {:.2f}%".format(fp*100/(fp+tn)))
#print("MCC = {:.2f}".format(((tn*tp)-(fp*fn))/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))
