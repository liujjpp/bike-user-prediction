from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras import backend as K

class FCNet:
    @staticmethod
    def build(dim):
        model = Sequential()
        model.add(Dense(64, input_shape=(dim, )))
        model.add(Activation("relu"))
        model.add(Dense(32))
        model.add(Activation("relu"))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        return model
