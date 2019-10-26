#Python Deep Learning DeepDrug3D
import keras
import numpy as np
import random

def read_list_file(filepath, group):
    with open(filepath, "r") as filin:
        listin = {}
        for line in filin:
            listin[line[:-1]] = group
    return listin

def split_train_val(all_dic, train_size, test_size):

    if train_size + test_size > len(all_dic):
        print("train_size and test_size are too big, changing to 50/50")
        train_size = len(all_dic) / 2
        test_size = train_size

    X_train_id = random.sample(list(all_dic), train_size)
    X_fullval_id = [item for item in list(all_dic) if item not in X_train_id]
    X_val_id = random.sample(X_fullval_id, test_size)

    X_train = np.zeros((len(X_train_id), 14, 32, 32, 32))
    X_val = np.zeros((len(X_val_id), 14, 32, 32, 32))
    Y_train = np.zeros((len(X_train_id), 3))
    Y_val = np.zeros((len(X_val_id), 3))

    for i in range(len(X_train_id)):
        X_train[i,:,:,:,:] = np.load(
            "../raw_data/deepdrug3d_voxel_data/" + X_train_id[i] + ".npy"
            )
        Y_train[i,:] = all_dic[X_train_id[i]]

    for i in range(len(X_val_id)):
        X_val[i,:,:,:,:] = np.load(
            "../raw_data/deepdrug3d_voxel_data/" + X_val_id[i] + ".npy"
            )
        Y_val[i,:] = all_dic[X_val_id[i]]
    
    return X_train, Y_train, X_val, Y_val


def notdeepdrug3D():
    input_layer = keras.Input(shape=(14, 32, 32, 32))
    conv_1 = keras.layers.Conv3D(
        filters = 32,
        kernel_size = 5,
        activation = "relu", 
        data_format = "channels_first",
        padding = "valid"
        )(input_layer)
    dropout_1 = keras.layers.Dropout(rate=0.2)(conv_1)
    conv_2 = keras.layers.Conv3D(
        filters = 32,
        kernel_size = 3,
        activation = "relu", 
        data_format="channels_first",
        padding="valid"
        )(dropout_1)
    max_pooling_1 = keras.layers.MaxPooling3D(
        pool_size=(2,2,2),
        strides=None,
        padding="valid",
        data_format="channels_first"
        )(conv_2)
    dropout_2 = keras.layers.Dropout(rate=0.4)(max_pooling_1)
    flatten_1 = keras.layers.Flatten()(dropout_2)
    dense_1 = keras.layers.Dense(units=100, activation="relu")(flatten_1)
    output_layer = keras.layers.Dense(units=3, activation="softmax")(dense_1)
    notdeepdrug_model = keras.Model(inputs=input_layer,outputs=output_layer)
    notdeepdrug_model.compile(
            optimizer="adam", 
            loss="categorical_crossentropy",
            metrics=["accuracy"]
            )
    return notdeepdrug_model

def mk_confu_table(predicted, observed):
    nb_class = len(predicted[0])
    confu_table = np.zeros((nb_class, nb_class))
    print(predicted.shape, observed.shape)
    for i in range(len(predicted)):
        predmax = -1
        for j in range(len(nb_class)):
            if predmax < predicted[i,j]:
                predmax = predicted[i,j]
                predicted_class = j
            if observed[i,j] == 1.:
                observed_class = j
        confu_table[predicted_class, observed_class] += 1                    
    return confu_table

def compute_model_metrics(predicted, observed):
    confu_table = mk_confu_table(predicted, observed)
    

control_file = "../raw_data/control.list"
heme_file = "../raw_data/heme.list"
nucleotide_file = "../raw_data/nucleotide.list"
steroid_file = "../raw_data/steroid.list"

control_id = read_list_file(control_file,[0, 0, 1])
heme_id = read_list_file(heme_file, [0, 1, 0])
nucleotide_id = read_list_file(nucleotide_file, [1, 0, 0])
steroid_id = read_list_file(steroid_file, 4)

print(len(control_id), len(heme_id), len(nucleotide_id), len(steroid_id))

all_dic = control_id
all_dic.update(heme_id)
all_dic.update(nucleotide_id)
#all_dic.update(steroid_id)

X_train, Y_train, X_val, Y_val = split_train_val(all_dic, 100 , 50)

notdeepdrug3D_model  = notdeepdrug3D()
model_file = "../results/notdeepdrug_model.h5"
best_model = keras.callbacks.ModelCheckpoint(
        filepath=model_file, 
        monitor = "val_loss",
        verbose = 0, 
        save_best_only=True
        )

history = notdeepdrug3D_model.fit(
    x=X_train, 
    y=Y_train, 
    batch_size=20, 
    epochs=5, 
    validation_split=0.05,
    shuffle=True,
    class_weigth = 
    )
pred_train = notdeepdrug3D_model.predict(X_train)
pred_val = notdeepdrug3D_model.predict(X_val)






