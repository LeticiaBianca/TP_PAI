from keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
from keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

class_names = dataset.class_names
print("Classes:", class_names)

val_dataset = dataset.take(10) 
train_dataset = dataset.skip(10) 

pre_treined_model = MobileNetV2(weights='imagenet', include_top=False)
pre_treined_model.trainable = False

model = models.Sequential([
    pre_treined_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation="softmax") 
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20)

 
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# Avaliar o modelo no conjunto de validação
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Acurácia de validação: {val_accuracy * 100:.2f}%")

# Obter as previsões do modelo
y_true = []
y_pred = []

for images, labels in val_dataset:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

# Calcular a acurácia
accuracy = accuracy_score(y_true, y_pred)
print(f"Acurácia: {accuracy * 100:.2f}%")

# Matriz de confusão
conf_matrix = confusion_matrix(y_true, y_pred)
print("Matriz de Confusão:")
print(conf_matrix)

# Relatório de classificação
class_report = classification_report(y_true, y_pred, target_names=class_names)
print("Relatório de Classificação:")
print(class_report)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

plt.xlabel('Predição')
plt.ylabel('Verdadeiro')
plt.tight_layout()
plt.show()

# Plotar gráfico de acurácia
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.title('Acurácia de Treinamento e Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()