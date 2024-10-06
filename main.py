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

# Images along with histogram
def plot_from_normal_formats(images):
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(images.ravel(), bins=256, color='gray', alpha=0.7)
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
        plot_from_normal_formats(image)

# Image along with histogram for .MAT files
def show_mat_image(index):
    global images
    if images is not None:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(images[index], cmap='gray')
        plt.axis('off')
        plt.title(f'Imagem {index + 1}')

        # Histogram
        plt.subplot(1, 2, 2)
        plt.hist(images[index].ravel(), bins=256, color='gray', alpha=0.7)
        plt.xlim([0, 255])
        plt.ylim([0, 5000])
        plt.title('Histograma')
        plt.xlabel('Intensidade de Pixel')
        plt.ylabel('Frequência')

        plt.tight_layout()
        plt.show()

# Load .MAT files
def load_mat_file():
    global images, current_image_index
    mat_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
    
    if mat_path:
        mat = scipy.io.loadmat(mat_path)
        data = mat['data']
        images = data[0, 0][-1]
        current_image_index = 0 
        show_mat_image(current_image_index)

def next_image():
    global current_image_index
    if images is not None and current_image_index < images.shape[0] - 1:
        current_image_index += 1
        show_mat_image(current_image_index)
    else:
        print("Não há mais imagens para mostrar.")

def select_roi():
    image_path = filedialog.askopenfilename(filetypes=[("Imagens", "*.png;*.jpg")])
    
    if image_path:
        image = cv2.imread(image_path)
        roi = cv2.selectROI("Selecione a ROI", image, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        if roi != (0, 0, 0, 0):
            roi_cropped = image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

            plt.imshow(cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title('Imagem Recortada (ROI)')
            plt.show()
    
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

next_image_button = Button(root, text='Próxima Imagem', command=next_image)
next_image_button.pack(pady=10)

# Select ROI [ BUTTON ]
roi_button = Button(root, text='Selecionar ROI', command=select_roi)
roi_button.pack(pady=10)

# GUI Loop
root.mainloop()