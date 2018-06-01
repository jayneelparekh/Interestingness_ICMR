# File with the function for creating the model
import keras
from keras.layers import Conv2D, Dense, Flatten, Input, Activation, Subtract, Concatenate, Dropout, Add
from keras.models import Model, Sequential
import keras.backend as K

def create_model(arg = 'image'):

    if (arg == 'image'):          # Version 6=9.2 (loss_fn changed)
	inp = Input(shape=(8192,))
	mid = Dropout(0.5)(inp)
	mid = Dense(8192, activation='relu', use_bias='True')(mid)
	mid = Dropout(0.5)(mid)
        mid = Dense(1024, activation='sigmoid', use_bias='True')(mid)
	mid = Dropout(0.5)(mid)
	out = Dense(1, activation='sigmoid', use_bias='True')(mid)
	model = Model(inputs=inp, outputs=out)


    elif (arg == 'video'):          # Version 6=9.2 (loss_fn changed)
	inp = Input(shape=(8192,))
	mid = Dropout(0.75)(inp)
	mid = Dense(8192, activation='relu', use_bias='True')(mid)
	mid = Dropout(0.75)(mid)
        mid = Dense(1024, activation='sigmoid', use_bias='True')(mid)
	mid = Dropout(0.75)(mid)
	out = Dense(1, activation='sigmoid', use_bias='True')(mid)
	model = Model(inputs=inp, outputs=out)


    return model
    
