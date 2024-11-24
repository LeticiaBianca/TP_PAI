import os
import customtkinter
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox

# data handler
import cv2
import scipy.io
from PIL import Image
import numpy as np
import xgboost as xg
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# image plots
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import pandas as pd
import warnings

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

roi_data = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
            '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
            '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
            '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
            '40', '41', '42', '43', '44', '45', '46', '47', '48', '49',
            '50', '51', '52', '53', '54']

# processed_data = {} # flatten images
class_data = {} # "saudavel" or "possui esteatose"
caracteristic_data = {} # coordenates, HI, entropy...
accuracy_results = []
sensitivities_results = []
specificities_results = []
aggregated_cm = [[0, 0], [0, 0]]
XGmodels = []

mobileNetModels = []

class_mapping = {
                    "possui esteatose": 0,
                    "saudável": 1
                }

def listar_arquivos(folder = "TP_PAI/dataset"):
    try:      
        arquivos = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        print(f'Total de arquivos: {len(arquivos)}')
        return arquivos
    except: 
        print(f"Erro ao pegar ROIs")
        return []

def create_dataset(image_paths, labels):

    def preprocess_image(filepath, label):
        img = tf.io.read_file("TP_PAI/dataset/" + filepath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.resize(img, (224, 224))
        img /= 255.0
        return img, label

    filepaths = tf.constant(image_paths)
    labels = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
     # Imprimindo o número de elementos antes de aplicar o batch
    dataset_size = sum(1 for _ in dataset)  # Conta o número total de exemplos
    print(f'Total number of examples: {dataset_size}')
    
    dataset = dataset.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
    
def train_mobile_net():
    global mobileNetModels, aggregated_cm, accuracy_results,sensitivities_results,specificities_results
    aggregated_cm = [[0, 0], [0, 0]]
    accuracy_results = []
    class_data = listar_arquivos();
    val_dataset_images = []
    val_dataset_labels = []
    train_dataset_images = []
    train_dataset_labels = []
    if class_data:  
        for test_patient in roi_data:
            for item in class_data:
                if test_patient in item:
                    val_dataset_images.append(item)
                    if int(item.split('_')[1]) <= 16:
                        val_dataset_labels.append(1)
                    else:
                        val_dataset_labels.append(0)
                # test data
                else:
                    train_dataset_images.append(item)
                    if int(item.split('_')[1]) <= 16:
                        train_dataset_labels.append(1)
                    else:
                        train_dataset_labels.append(0)
    
            train_dataset = create_dataset(train_dataset_images, train_dataset_labels)
            val_dataset = create_dataset(val_dataset_images, val_dataset_labels)

            #Load the pre-trained model
            pre_treined_model = MobileNetV2(weights='imagenet', include_top=False)
            pre_treined_model.trainable = False

            #Model training
            mobileNetModel = models.Sequential([
                pre_treined_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(len(class_mapping), activation="softmax") 
            ])

            mobileNetModel.compile(optimizer="adam",
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"])

            mobileNetModel.summary()

            history = mobileNetModel.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=20,
                verbose=1)

            
            mobileNetModel.compile(optimizer="adam",
                        loss="binary_crossentropy",
                        metrics=["accuracy"])

            mobileNetModel.summary()
            mobileNetModels.append(mobileNetModel)
            show_metrics(val_dataset, history, mobileNetModel)
        show_final_metrics()
    

def show_metrics(val_dataset, history, mobileNetModel):
    global aggregated_cm, accuracy_results
    y_true = []
    y_pred = []

    for images, labels in val_dataset:
        predictions = mobileNetModel.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    # Calculate the accuracy
    accuracy = accuracy_score(y_true, y_pred)
    accuracy_results.append(accuracy)
    print(f"Acurácia: {accuracy * 100:.2f}%")

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)   
    aggregated_cm += conf_matrix
    print("Matriz de Confusão:")
    print(conf_matrix)

    # Classification report
    class_report = classification_report(y_true, y_pred, target_names=class_mapping.keys(), labels=[0, 1])
    print("Relatório de Classificação:")
    print(class_report)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.colorbar()
    tick_marks = np.arange(len(class_mapping))
    plt.xticks(tick_marks, class_mapping.keys(), rotation=45)
    plt.yticks(tick_marks, class_mapping.keys())

    plt.xlabel('Predição')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.show()

    # Plot accuracy graph
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title('Acurácia de Treinamento e Validação')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def show_final_metrics():
    global aggregated_cm, accuracy_results
    
    avg_accuracy = np.mean(accuracy_results)
    print(f"Acurácia: {avg_accuracy * 100:.2f}%")


    print("Matriz de Confusão:")
    print(aggregated_cm)


    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(aggregated_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.colorbar()
    tick_marks = np.arange(len(class_mapping))
    plt.xticks(tick_marks, class_mapping.keys(), rotation=45)
    plt.yticks(tick_marks, class_mapping.keys())

    plt.xlabel('Predição')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.show()

if len(mobileNetModels) == 0:
    train_mobile_net()

image_path = filedialog.askopenfilename(filetypes=[("Imagens e MAT", "*.png *.jpg *.mat")])

if image_path:
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Imagem não pode ser carregada.")
            
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    predictions = []
    class_idx = []

    for model in mobileNetModels:
        predictions.append(model.predict(img))
        class_idx.append(np.argmax(predictions))

    def classification_status():
        hi_window = Toplevel(root)  
        hi_window.title("Classificação")
        hi_window.geometry("400x100")
                
        label = Label(hi_window, text=f"Classe prevista: {class_mapping[np.mean(class_idx)]} com probabilidade {np.mean(predictions)[0][np.mean(class_idx)]:.2f}", font=('Arial', 10))
        label.pack(pady=5)
    classification_status()