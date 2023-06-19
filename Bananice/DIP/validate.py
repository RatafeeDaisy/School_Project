import numpy as np
from tensorflow import keras

def validate(path):

    flower_dict = {0:'dasiy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}

    img_height = 180
    img_width = 180

    model = keras.models.load_model('./model.h5', compile=True)

    data = keras.preprocessing.image.load_img(path,target_size=(img_height, img_width))
    data = keras.preprocessing.image.img_to_array(data)
    data = np.expand_dims(data, axis=0)
    data = np.vstack([data])
    result = np.argmax(model.predict(data))

    return flower_dict[result]