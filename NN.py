from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout
from tensorflow.keras import backend as K
import tensorflow as tf
import  getDataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy

def create_siamese_network(input_shape):
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)

    model = tf.keras.Sequential()
    model.add(Conv2D(124, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(124, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(50, activation='relu'))
    encoded1 = model(input1)
    encoded2 = model(input2)

    distance = Lambda(lambda x: K.abs(x[0] - x[1]))([encoded1, encoded2])

    siamese_net = Model(inputs=[input1, input2], outputs=distance)

    return siamese_net

trainDg = getDataset.getSiameseDataset('result.csv', '../VehicleID_V1.0/image/', 128, 64)
valDg = getDataset.getSiameseDataset('val.csv', '../VehicleID_V1.0/image/', 128, 64)

model = create_siamese_network((64, 64) + (3,))

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.0001), loss=binary_crossentropy, metrics=['accuracy'])

# Тренировка модели
model.fit(trainDg, epochs=10, validation_data=valDg)