# Letícia Bianca Oliveira - 
# Raick Miranda Rodrigues Santos - 781755

# GUI Libraries
from tkinter import *
from tkinter import filedialog
import customtkinter

# Data Manipulation
import numpy as np
import scipy.io
import cv2

# Image Plots
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# Global variables
CURRENT_IMAGE_INDEX = 0
IMAGES = None

# Display image along with histogram
def display_image_and_histogram(image, title="Imagem"):
    plt.figure(figsize=(10, 5))

    # Image
    plt.subplot(1, 2, 1)
    if len(image.shape) == 2:  # Grayscale
        plt.imshow(image, cmap='gray')
    else:  # RGB
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)

    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(image.ravel(), bins=256, color='gray', alpha=0.7)
    plt.xlim([0, 255])
    plt.title('Histograma')
    plt.xlabel('Intensidade de Pixel')
    plt.ylabel('Frequência')

    plt.tight_layout()
    plt.show()

# Load PNG/JPG files from computer
def load_image():
    image_path = filedialog.askopenfilename(filetypes=[("Imagens", "*.png;*.jpg")])
    
    if image_path:
        image = cv2.imread(image_path)
        display_image_and_histogram(image, title="Imagem Carregada")

# Load .MAT files and display images
def load_mat_file():
    global IMAGES, CURRENT_IMAGE_INDEX
    mat_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
    
    if mat_path:
        mat = scipy.io.loadmat(mat_path)
        data = mat['data']
        IMAGES = data[0, 0][-1]
        CURRENT_IMAGE_INDEX = 0
        display_image_and_histogram(IMAGES[CURRENT_IMAGE_INDEX], title=f"Imagem {CURRENT_IMAGE_INDEX + 1}")

# ESSA FUNÇÃO É PROVISÓRIA
def next_image():
    global CURRENT_IMAGE_INDEX
    if IMAGES is not None and CURRENT_IMAGE_INDEX < IMAGES.shape[0] - 1:
        CURRENT_IMAGE_INDEX += 1
        display_image_and_histogram(IMAGES[CURRENT_IMAGE_INDEX], title=f"Imagem {CURRENT_IMAGE_INDEX + 1}")
    else:
        print("Não há mais imagens para mostrar.")

# ROI -> Region of Interest
def select_roi():
    image_path = filedialog.askopenfilename(filetypes=[("Imagens", "*.png;*.jpg")])
    
    if image_path:
        image = cv2.imread(image_path)
        roi = cv2.selectROI("Selecione a ROI", image, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        if roi != (0, 0, 0, 0):
            roi_cropped = image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

            display_image_and_histogram(roi_cropped, title="Imagem Recortada (ROI)")

# ------------------------------------ GUI ----------------------------------------

# Init
customtkinter.set_appearance_mode("light")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green
root = customtkinter.CTk()

# Config
root.geometry("600x400")
root.title('Sistema Auxiliar de Diagnóstico da NAFLD')

# Header
label = Label(root, text='Sistema Auxiliar de Diagnóstico da NAFLD', font=('Arial', 14))
label.pack(pady=10)

# Load images (png / jpg) from computer [ BUTTON ]
normal_images_format_button = Button(root, text='Carregar Imagem', command=load_image)
normal_images_format_button.pack(pady=10)

# Load and display .mat file images [ BUTTON ]
mat_file_button = Button(root, text='Carregar Arquivo .mat', command=load_mat_file)
mat_file_button.pack(pady=10)

# Next image [ BUTTON ]
next_image_button = Button(root, text='Próxima Imagem', command=next_image)
next_image_button.pack(pady=10)

# Select ROI [ BUTTON ]
roi_button = Button(root, text='Selecionar ROI', command=select_roi)
roi_button.pack(pady=10)

# GUI Loop
root.mainloop()
