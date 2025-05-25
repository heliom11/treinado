import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


caminho_dados = 'data'

gerador_treino = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)


treino = gerador_treino.flow_from_directory(
    caminho_dados,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validacao = gerador_treino.flow_from_directory(
    caminho_dados,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)


modelo = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  
])


modelo.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

historico = modelo.fit(
    treino,
    epochs=10,
    validation_data=validacao
)


modelo.save('modelo_edema_pulmonar.h5')

plt.plot(historico.history['accuracy'], label='Treino')
plt.plot(historico.history['val_accuracy'], label='Validação')
plt.title('Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()