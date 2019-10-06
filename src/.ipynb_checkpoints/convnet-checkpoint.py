{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Classification des poches protéiques en fonction du type de druggabilité, par un CNN"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1) Préparation des données"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "import keras\n",
    "import numpy as np\n",
    "\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.model_selection import KFold\n",
    "from keras.layers import Dense, Flatten, Dropout\n",
    "from keras.wrappers.scikit_learn import KerasClassifier\n",
    "from keras.callbacks.callbacks import EarlyStopping, Callback\n",
    "from keras.layers import add, Activation\n",
    "from keras.layers import Conv3D, MaxPool3D\n",
    "from keras.models import Sequential\n",
    "from os import listdir"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "PATH_TRAIN = \"../data/x_train\"\n",
    "PATH_TEST = \"../data/x_test\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_x(path, control):\n",
    "    X = [np.load(\"{}/{}\".format(path,pocket))\n",
    "         for pocket in listdir(path) \n",
    "         if pocket not in control]\n",
    "    X = [np.squeeze(array) for array in X]\n",
    "    X = np.array(X)\n",
    "    X = np.moveaxis(X, 1, -1)\n",
    "    return X\n",
    "\n",
    "def load_y(path, nucleotid, heme, steroid, control):\n",
    "    Y = []\n",
    "    for pocket in listdir(path):\n",
    "        if pocket in nucleotid:\n",
    "            Y.append(1)\n",
    "        elif pocket in heme:\n",
    "            Y.append(2)\n",
    "        elif pocket in steroid:\n",
    "            Y.append(3)\n",
    "        elif pocket in control:\n",
    "            continue \n",
    "    Y  = np.array(Y)\n",
    "    return Y\n",
    "\n",
    "def one_hot_encoding(y):\n",
    "    classes = LabelEncoder()\n",
    "    integer_encoding = classes.fit_transform(y)\n",
    "    one_hot_Y = keras.utils.to_categorical(integer_encoding)\n",
    "    return one_hot_Y\n",
    "\n",
    "def list_generator(file):\n",
    "    with open(file, \"r\") as filin:\n",
    "        liste = [\"{}.npy\".format(line[:-1]) for line in filin]\n",
    "    return liste"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(127, 32, 32, 32, 14)\n",
      "(73, 32, 32, 32, 14)\n",
      "(127, 3)\n",
      "(73, 3)\n"
     ]
    }
   ],
   "source": [
    "#X_train est une liste d'array de dimensions (1,14,32,32,32)\n",
    "#Il est constitué des 346 premières poches\n",
    "nucleotid = list_generator(\"nucleotide.list.txt\")\n",
    "heme = list_generator(\"heme.list.txt\")\n",
    "steroid = list_generator(\"steroid.list.txt\")\n",
    "control = list_generator(\"control.list.txt\")\n",
    "X_train = load_x(PATH_TRAIN, control)\n",
    "X_test = load_x(PATH_TEST, control)\n",
    "Y_train = load_y(PATH_TRAIN, nucleotid, heme, steroid, control)\n",
    "Y_test = load_y(PATH_TEST, nucleotid, heme, steroid, control)\n",
    "one_hot_Y_train = one_hot_encoding(Y_train)\n",
    "one_hot_Y_test = one_hot_encoding(Y_test)\n",
    "print(X_train.shape)\n",
    "print(X_test.shape)\n",
    "print(one_hot_Y_train.shape)\n",
    "print(one_hot_Y_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Pour les Y va créer un code pour savoir quelle est la classe de druggabilité de la poche:\n",
    "<ul>\n",
    "    <li/> 1 : nucléotide\n",
    "    <li/> 2 : hème\n",
    "    <li/> 3 : stéroïde\n",
    "    <li/> 4 : control"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Attention cependant, j'ai pris les 346 première poches, or peut être qu'il y a un déséquilibre de classe: beaucoup de poches dans une classe et beaucoup dans d'autres. On va regarder. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "nt= 0\n",
    "hem = 0\n",
    "ste = 0\n",
    "ctr = 0\n",
    "\n",
    "for i in range(0,one_hot_Y_train.shape[0]):\n",
    "    if one_hot_Y_train[i,0]:\n",
    "        nt += 1\n",
    "    elif one_hot_Y_train[i,1]:\n",
    "        hem += 1\n",
    "    elif one_hot_Y_train[i,2]:\n",
    "        ste += 1\n",
    "    else:\n",
    "        ctr += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "52\n",
      "68\n",
      "7\n",
      "0\n",
      "52+68+7+0 = 127\n",
      "127\n",
      "1946\n"
     ]
    }
   ],
   "source": [
    "print(nt)\n",
    "print(hem)\n",
    "print(ste)\n",
    "print(ctr)\n",
    "print(\"{}+{}+{}+{} = {}\".format(nt,hem,ste,ctr, nt+hem+ste+ctr))\n",
    "print(len(one_hot_Y_train))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "On s'aperçoit qu'on a très peu de poches druggables par les stéroïdes, c'est une bonne chose car ce sont des faux positifs. Les stéroïdes jouent le rôle de contrôle négatif. A part ça, pas de prédominance particulière d'une classe par rapport à l'autre."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2) Construction du modèle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def model_one():\n",
    "    model = Sequential()\n",
    "    model.add(Conv3D(filters = 21, kernel_size = 9, strides=1, padding= \"same\", activation = \"relu\", kernel_initializer=\"he_normal\"))\n",
    "    #model.add(Conv3D(filters=64, kernel_size = 26, strides=1, padding= \"same\", activation = \"relu\", kernel_initializer=\"he_normal\"))\n",
    "    model.add(Dropout(0.4))\n",
    "    model.add(MaxPool3D(pool_size = 2, strides = 1, padding = \"same\"))\n",
    "    model.add(Dropout(0.4))\n",
    "    model.add(Flatten())\n",
    "    model.add(Dense(units = 50, activation = \"relu\"))\n",
    "    model.add(Dropout(0.4))\n",
    "    model.add(Dense(units = 3, activation = \"softmax\"))\n",
    "    model.compile(optimizer=\"adam\",loss=\"categorical_crossentropy\",metrics=['accuracy'])\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "cb = Callback()\n",
    "critor = EarlyStopping(monitor = \"val_loss\", patience = 2)\n",
    "cb_list = [critor]\n",
    "my_model = model_one()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 75 samples, validate on 25 samples\n",
      "Epoch 1/10\n",
      "75/75 [==============================] - 162s 2s/step - loss: 1.0729 - accuracy: 0.5467 - val_loss: 1.0747 - val_accuracy: 0.2400\n",
      "Epoch 2/10\n",
      "75/75 [==============================] - 154s 2s/step - loss: 1.0661 - accuracy: 0.5467 - val_loss: 1.0700 - val_accuracy: 0.2400\n",
      "Epoch 3/10\n",
      "75/75 [==============================] - 154s 2s/step - loss: 1.0623 - accuracy: 0.5467 - val_loss: 1.0656 - val_accuracy: 0.2400\n",
      "Epoch 4/10\n",
      "75/75 [==============================] - 154s 2s/step - loss: 1.0569 - accuracy: 0.5467 - val_loss: 1.0611 - val_accuracy: 0.2400\n",
      "Epoch 5/10\n",
      "75/75 [==============================] - 146s 2s/step - loss: 1.0507 - accuracy: 0.5467 - val_loss: 1.0567 - val_accuracy: 0.2400\n",
      "Epoch 6/10\n",
      "75/75 [==============================] - 138s 2s/step - loss: 1.0415 - accuracy: 0.5467 - val_loss: 1.0524 - val_accuracy: 0.2400\n",
      "Epoch 7/10\n",
      "75/75 [==============================] - 149s 2s/step - loss: 1.0402 - accuracy: 0.5467 - val_loss: 1.0484 - val_accuracy: 0.2400\n",
      "Epoch 8/10\n",
      "75/75 [==============================] - 147s 2s/step - loss: 1.0321 - accuracy: 0.5467 - val_loss: 1.0445 - val_accuracy: 0.2400\n",
      "Epoch 9/10\n",
      "75/75 [==============================] - 147s 2s/step - loss: 1.0292 - accuracy: 0.5467 - val_loss: 1.0406 - val_accuracy: 0.2400\n",
      "Epoch 10/10\n",
      "75/75 [==============================] - 120s 2s/step - loss: 1.0266 - accuracy: 0.5467 - val_loss: 1.0369 - val_accuracy: 0.2400\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.callbacks.History at 0x7f7a56573b90>"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "my_model.fit(X_train[0:75:,:,:,:],one_hot_Y_train[0:75,:],epochs=10, batch_size = 15,\n",
    "          validation_data = (X_test[0:25,:,:,:,:],one_hot_Y_test[0:25,:]), callbacks = cb_list)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3) Evaluation du modèle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "127/127 [==============================] - 76s 600ms/step\n",
      "[1.0223322872101792, 0.5354330539703369]\n"
     ]
    }
   ],
   "source": [
    "evaluation = my_model.evaluate(X_train, one_hot_Y_train)\n",
    "print(evaluation)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Avec les paramètres suivants:\n",
    "<ul>\n",
    "    <li/> filters = 24\n",
    "    <li/> kernel_size = 12\n",
    "    <li/> pool_size = 2\n",
    "    <li/> units = 50\n",
    "    \n",
    "J'obtient une val loss de 1.29 et une accuracy de 0.37"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "If no scoring is specified, the estimator passed should have a 'score' method. The estimator <keras.engine.sequential.Sequential object at 0x7f7aad3f9590> does not.",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-26-7dafead5d228>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0mtraining\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mKerasClassifier\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbuild_fn\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel_one\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mepochs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m5\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m15\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mverbose\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;31m#kfold = KFold(n_splits = 5, shuffle=True)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mcv_result\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcross_val_score\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmy_model\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtraining\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mX_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mone_hot_Y_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcv\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mkfold\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcv_result\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"%.2f%%(%2d%%)\"\u001b[0m\u001b[0;34m%\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcv_result\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmean\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0;36m100\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcv_result\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstd\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0;36m100\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/miniconda3/envs/dev/lib/python3.7/site-packages/sklearn/model_selection/_validation.py\u001b[0m in \u001b[0;36mcross_val_score\u001b[0;34m(estimator, X, y, groups, scoring, cv, n_jobs, verbose, fit_params, pre_dispatch, error_score)\u001b[0m\n\u001b[1;32m    382\u001b[0m     \"\"\"\n\u001b[1;32m    383\u001b[0m     \u001b[0;31m# To ensure multimetric format is not supported\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 384\u001b[0;31m     \u001b[0mscorer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcheck_scoring\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mestimator\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mscoring\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mscoring\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    385\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    386\u001b[0m     cv_results = cross_validate(estimator=estimator, X=X, y=y, groups=groups,\n",
      "\u001b[0;32m~/miniconda3/envs/dev/lib/python3.7/site-packages/sklearn/metrics/scorer.py\u001b[0m in \u001b[0;36mcheck_scoring\u001b[0;34m(estimator, scoring, allow_none)\u001b[0m\n\u001b[1;32m    293\u001b[0m                 \u001b[0;34m\"If no scoring is specified, the estimator passed should \"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    294\u001b[0m                 \u001b[0;34m\"have a 'score' method. The estimator %r does not.\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 295\u001b[0;31m                 % estimator)\n\u001b[0m\u001b[1;32m    296\u001b[0m     \u001b[0;32melif\u001b[0m \u001b[0misinstance\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mscoring\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mIterable\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    297\u001b[0m         raise ValueError(\"For evaluating multiple scores, use \"\n",
      "\u001b[0;31mTypeError\u001b[0m: If no scoring is specified, the estimator passed should have a 'score' method. The estimator <keras.engine.sequential.Sequential object at 0x7f7aad3f9590> does not."
     ]
    }
   ],
   "source": [
    "training = KerasClassifier(build_fn = model_one(), epochs = 5, batch_size=15, verbose=0)\n",
    "#kfold = KFold(n_splits = 5, shuffle=True)\n",
    "cv_result = cross_val_score(my_model, training, X_train, one_hot_Y_train, cv = kfold)\n",
    "print(cv_result)\n",
    "print(\"%.2f%%(%2d%%)\"%(cv_result.mean()*100, cv_result.std()*100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
