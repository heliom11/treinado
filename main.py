import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

CSV_PATH = r"C:\Users\HC\Desktop\ENDEMA_PULMONAR\Data_Entry_2017.csv"
IMG_DIR = r"C:\Users\HC\Desktop\ENDEMA_PULMONAR\images"


df = pd.read_csv(CSV_PATH)
df = df[df["Finding Labels"].str.contains("Pneumonia")]
df["path"] = df["Image Index"].apply(lambda x: os.path.join(IMG_DIR, x))
df["label"] = 1  # pneumonia = 1


def preprocess_image(path, size=(224, 224)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((224, 224, 3))
    img = cv2.resize(img, size)
    img = img / 255.0
    return np.stack([img]*3, axis=-1)

print("🔁 Lendo imagens...")
X = np.array([preprocess_image(p) for p in df["path"][:5000]])
y = np.array([1] * len(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

model.save("modelo_pneumonia_5000.h5")
print("✅ Modelo salvo como modelo_pneumonia_5000.h5")


plt.plot(history.history['accuracy'], label='Acurácia (treino)')
plt.plot(history.history['val_accuracy'], label='Acurácia (validação)')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.title('Desempenho do Modelo')
plt.legend()
plt.show()

# 📊 AVALIAÇÃO
y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)

print("📊 Matriz de confusão:")
print(confusion_matrix(y_test, y_pred_labels))

print("\n📄 Relatório de classificação:")
print(classification_report(y_test, y_pred_labels))