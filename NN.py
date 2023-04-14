from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout
from tensorflow.keras import backend as K
import tensorflow as tf
import  getDataset
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import binary_crossentropy

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
def contrastive_loss(y, preds, margin=1):
	# explicitly cast the true class label data type to the predicted
	# class label data type (otherwise we run the risk of having two
	# separate data types, causing TensorFlow to error out)
	y = tf.cast(y, preds.dtype)
	# calculate the contrastive loss between the true labels and
	# the predicted labels
	squaredPreds = K.square(preds)
	squaredMargin = K.square(K.maximum(margin - preds, 0))
	loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
	# return the computed contrastive loss to the calling function
	return loss
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

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encoded1, encoded2])

    siamese_net = Model(inputs=[input1, input2], outputs=distance)

    return siamese_net

trainDg = getDataset.getSiameseDataset('result.csv', '../VehicleID_V1.0/image/', 128, 64)
valDg = getDataset.getSiameseDataset('val.csv', '../VehicleID_V1.0/image/', 128, 64)

model = create_siamese_network((64, 64) + (3,))
rms = RMSprop()
# Компиляция модели
model.compile(optimizer="adam", loss=contrastive_loss, metrics=['accuracy'])

# Тренировка модели
model.fit(trainDg, epochs=10, validation_data=valDg)