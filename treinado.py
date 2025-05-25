import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import sys

MODEL_PATH = 'meu_modelo.h5'
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)[0]
    class_idx = int(np.argmax(prediction))
    print(class_idx)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Informe o caminho da imagem.")
    else:
        predict(sys.argv[1])