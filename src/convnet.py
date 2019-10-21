#!/usr/bin/env python
# coding: utf-8

# ## Classification des poches protéiques en fonction du type de druggabilité, par un CNN

# ### 1) Préparation des données

# In[1]:


import keras
import numpy as np
from random import sample
from random import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.layers import Dense, Flatten, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks.callbacks import EarlyStopping, Callback, ModelCheckpoint
from keras.layers import add, Activation
from keras.layers import Conv3D, MaxPool3D
from keras.models import Sequential, load_model
from os import listdir


# In[2]:


PATH_DATA = "/media/anthony/POULOP/deepdrug3d_voxel_data"


# In[3]:


def equilibrator_samplor(path, nucleotid, heme, control, steroid, k):
    all_pocket = listdir(path)
    ech = sample(nucleotid, k) + sample(heme, k) + sample(control, k)
    shuffle(ech)
    chosen_pocket = [pocket for pocket in all_pocket if pocket in ech]
    return chosen_pocket

def remove_list(chosen_pocket, nucleotid, heme, control, steroid):
    for pocket in chosen_pocket:
        if pocket in nucleotid:
            nucleotid.remove(pocket)
        elif pocket in heme:
            heme.remove(pocket)
        elif pocket in control:
            control.remove(pocket)
        elif pocket in steroid:
            steroid.remove(pocket)

def load_x(path, chosen_pocket):
    try:
        X = [np.load("{}/{}".format(path, pocket))
             for pocket in chosen_pocket]
    except ValueError:
        print(pocket)
    X = [np.squeeze(array) for array in X]
    X = np.array(X)
    X = np.moveaxis(X, 1, -1)
    return X

def load_y(chosen_pocket, nucleotid, heme, control, steroid):
    Y = []
    for pocket in chosen_pocket:
        if pocket in nucleotid:
            Y.append(1)
        elif pocket in heme:
            Y.append(2)
        elif pocket in steroid:
            Y.append(4)
        elif pocket in control:
            Y.append(3)
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


# In[44]:


nucleotid = list_generator("nucleotide.list.txt")
heme = list_generator("heme.list.txt")
steroid = list_generator("steroid.list.txt")
control = list_generator("control.list.txt")


# In[5]:


print(len(nucleotid))
print(len(heme))
print(len(control))
print(len(steroid))


# In[79]:


train_pocket = equilibrator_samplor(PATH_DATA, nucleotid, heme, control, steroid, 75)
X_train = load_x(PATH_DATA, train_pocket)
Y_train = load_y(train_pocket, nucleotid, heme, control, steroid)
one_hot_Y_train = one_hot_encoding(Y_train)

remove_list(train_pocket, nucleotid, heme, control, steroid)
print(len(nucleotid))
print(len(heme))
print(len(control))
print(len(steroid))


# In[80]:


test_pocket = equilibrator_samplor(PATH_DATA, nucleotid, heme, control, steroid, 25)
X_test = load_x(PATH_DATA, test_pocket)
Y_test = load_y(test_pocket, nucleotid, heme, control, steroid)
one_hot_Y_test = one_hot_encoding(Y_test)
print(X_train.shape)
print(X_test.shape)
print(len(Y_train))
print(len(Y_test))
print(one_hot_Y_train.shape)
print(one_hot_Y_test.shape)


# In[81]:



# ### 2) Construction du modèle

# In[84]:


def model_one():
    model = Sequential()
    model.add(Conv3D(filters =32, kernel_size = (14,14,14), data_format="channels_last", strides=1, padding= "same", activation = "relu"))
    model.add(Conv3D(filters = 16, kernel_size = (8,8,8), data_format="channels_last", strides=1, padding= "same", activation = "relu"))
    #model.add(Conv3D(filters = 8, kernel_size = 3, data_format="channels_last", strides=1, padding= "same", activation = "relu", kernel_initializer="he_normal"))
    model.add(Dropout(0.3))
    model.add(MaxPool3D(pool_size = 2, strides = 1, padding = "same"))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units = 75, activation = "relu"))
    model.add(Dropout(0.3))
    model.add(Dense(units = 3, activation = "softmax"))
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])
    return model


# In[ ]:




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


np.random.seed(2000)
critor = EarlyStopping(monitor = "val_loss", patience = 4, mode = "min")
my_model = model_one()

#best_model_path = "../results/my_model"+".h5"
#best_model = ModelCheckpoint(best_model_path, monitor = "val_loss", verbose = 2, save_best_only = True)
#my_best_model = load_model("../results/my_model.h5")

my_model.fit(X_train, one_hot_Y_train, epochs = 15, batch_size = 50,
             validation_split = 0.1, callbacks = [critor])


# # 3) Evaluation du modèle

# In[13]:


evaluation = my_model.evaluate(X_test, one_hot_Y_test)
print(evaluation)


# In[ ]:







