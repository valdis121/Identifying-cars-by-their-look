import tensorflow as tf
import pandas as pd
from PIL import Image

nameOfCSV = 'result.csv'
pathToImage = '../VehicleID_V1.0/image/'

data = pd.read_csv(nameOfCSV, dtype={'car1': object, 'car2': object})

dataset = []
for i in range(len(data)):
    img_path1 = "{}{}.jpg".format(pathToImage, data.iloc[i]['car1'])
    img_path2 = "{}{}.jpg".format(pathToImage, data.iloc[i]['car2'])
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)
    target = data.iloc[i]['distance']
    sample = {'image1': img1, 'image2': img2, 'target': target}
    dataset.append(sample)
    img1.close()
    img2.close()

def preprocess(sample):
    img1 = tf.convert_to_tensor(sample['image1'])
    img2 = tf.convert_to_tensor(sample['image2'])
    target = tf.constant(sample['target'], dtype=tf.float32)
    return (img1, img2), target

tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
tf_dataset = tf_dataset.map(preprocess)

tf.data.experimental.save(tf_dataset, 'dataset.tfrecord')
