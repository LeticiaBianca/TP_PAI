# Letícia Bianca Oliveira - 776782
# Raick Miranda Rodrigues Santos - 781755
# Nathália Mascarenhas Tenaglia - 766430
# NT = (776782 + 781755 + 766430) mod 4 = 3
# NC = (776782 + 781755 + 766430) mod 2 = 1
# ND = (776782 + 781755 + 766430) mod 5 = 2

# ------------------------------------ LIBRARIES & MODULES --------------------------

# GUI & OS demands
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# image plots
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import pandas as pd
import warnings

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop

import shutil


# ----------------------------- GLOBAL VARIABLES ------------------------------------

CURRENT_IMAGE_INDEX = 0  # paginationimport shutil

IMAGES = None

IsLoadImage = True
patient = 0
ultrasound = 0

roi_data = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
            '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
            '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
            '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
            '40', '41', '42', '43', '44', '45', '46', '47', '48', '49',
            '50', '51', '52', '53', '54']

class_data = {} # "saudavel" or "possui esteatose"
caracteristic_data = {} # coordenates, HI, entropy...
accuracy_results = []
sensitivities_results = []
specificities_results = []
aggregated_cm = [[0, 0], [0, 0]]
XGmodels = []
treinamento_acuracia = []
validacao_acuracia = []

class_mapping = {"possui esteatose": 0, "saudável": 1}
inverse_class_mapping = {v: k for k, v in class_mapping.items()}
image_to_classify = np.array([0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0]) # same as caracteristics

# ----------------------------- FUNCTIONALITIES (Part 1) ----------------------------
def update_excel(file_name, column, value):
   
    file_path = "rois_informations.xlsx"
    file_path = os.path.join(os.getcwd(), file_path)

    df = pd.read_excel(file_path)

    if file_name not in df.iloc[:, 0].values:
        print(f"O arquivo '{file_name}' não foi encontrado.")
        return

    if column not in df.columns:
        print(f"A coluna '{column}' não foi encontrada.")
        return

    idx = df[df.iloc[:, 0] == file_name].index[0]
    \
    try:
        df.at[idx, column] = value

        df.to_excel(file_path, index=False)
        print(f"Valor atualizado na célula correspondente ao arquivo '{file_name}' e coluna '{column}'.")
    except Exception as e:
        print(f"Erro ao salvar o arquivo Excel: {idx}")

def update_csv(file_name, column, value):

    file_path = "rois_informations.csv"
    file_path = os.path.join(os.getcwd(), file_path)
    
    try:
        df = pd.read_csv(file_path, delimiter=";")
    except FileNotFoundError:
        print(f"Arquivo '{file_path}' não foi encontrado.")
        return

    if column not in df.columns:
        print(f"A coluna '{column}' não foi encontrada.")
        return
    
    row = df[df['Nome do arquivo'] == file_name].index[0]

    try:
        df.at[row, column] = value
        df.to_csv(file_path, index=False, sep=";")
        print(f"Valor atualizado na coluna '{column}' para o identificador '{value}'.")
    except Exception as e:
        print(f"Erro ao salvar o arquivo CSV: {e}") 

# plot images with histogram
def display_image_and_histogram(image, title="Imagem", parent=None, isROI=False):
    average_brightness = np.mean(image)
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(10, 5))

    # image settings
    ax_img.imshow(image, cmap='gray')
    ax_img.axis('off')
    ax_img.set_title(title)

    # histogram settings
    n, bins, patches = ax_hist.hist(
        image.ravel(), bins=256, color='gray', alpha=0.7)
    ax_hist.set_xlim([0, 255])
    ax_hist.set_ylim([0, max(n) * 1.1])
    ax_hist.set_title('Histograma')
    ax_hist.set_xlabel('Intensidade de Pixel')
    ax_hist.set_ylabel('Frequência')

    # if not ROI display average brightness line
    if isROI:
        ax_hist.axvline(average_brightness, color='red',
                        linestyle='dashed', linewidth=1)
        ax_hist.text(average_brightness + 5, max(n) * 0.9,
                     f'Avg: {average_brightness:.2f}', color='red')

    plt.tight_layout()

    # add image plots
    if parent is not None:
        for widget in parent.winfo_children():
            widget.destroy()  # remove previous plot

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, anchor="center",
                                    padx=10, pady=10, fill="both", expand=True)

        toolbar_frame = Frame(parent)
        toolbar_frame.pack(side=TOP, anchor="center", padx=10, pady=10)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()


def display_image(image, title="Imagem", parent=None):
    display_image_and_histogram(
        image, title, parent, isROI=False)


def display_roi(roi, title="ROI", parent=None):
    display_image_and_histogram(roi, title, parent, isROI=True)

# load image files from pc
def load_file(isROI=False):
    global IMAGES, CURRENT_IMAGE_INDEX, IsLoadImage
    IsLoadImage = True

    file_path = filedialog.askopenfilename(
        filetypes=[("Imagens e MAT", "*.png *.jpg *.mat")]
    )

    if file_path:  
        # .png,.jpg,.jpeg files
        if file_path.lower().endswith(('*.png', '*.jpg', '*.jpeg')):
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if isROI: 
                display_roi(image, title="Imagem Local", parent=roi_frame)
            else:
                display_image(
                    image, title="Imagem Local", parent=image_frame)

        # .mat files
        elif file_path.lower().endswith('.mat'):
            mat = scipy.io.loadmat(file_path)
            data = mat['data']

            IMAGES = []

            for entry in data[0]:  # images are in the last element of entry
                images_array = entry[-1]

                if images_array is not None:
                    for image in images_array:
                        IMAGES.append(image)

            CURRENT_IMAGE_INDEX = 0  # start at the first image

            if isROI:
                display_roi(IMAGES[CURRENT_IMAGE_INDEX],
                            title=f"Imagem {CURRENT_IMAGE_INDEX}",
                            parent=roi_frame)
            else:
                display_image(IMAGES[CURRENT_IMAGE_INDEX],
                              title=f"Imagem {
                    CURRENT_IMAGE_INDEX + 1}",
                    parent=image_frame)


def destroy_plots():
    for widget in image_frame.winfo_children():
        widget.destroy()

    for widget in roi_liver_frame.winfo_children():
        widget.destroy()

    for widget in roi_kidney_frame.winfo_children():
        widget.destroy()

# ----------------------------- PAGINATION ------------------------------------------

# skip to the next image in the current pagination
def next_image():
    global CURRENT_IMAGE_INDEX

    # if IMAGES is not empty and if there is a next image
    if IMAGES is not None and CURRENT_IMAGE_INDEX < len(IMAGES) - 1:
        CURRENT_IMAGE_INDEX += 1
        destroy_plots()

        if IsLoadImage:  # if regular image display histogram
            display_image_and_histogram(IMAGES[CURRENT_IMAGE_INDEX],
                                        title=f"Imagem {
                                            CURRENT_IMAGE_INDEX}",
                                        parent=image_frame)
        else:  # if ROI display the image in fullsize
            cut_rois(cv2.cvtColor(
                IMAGES[CURRENT_IMAGE_INDEX], cv2.COLOR_GRAY2BGR))
    else:
        print("Não há mais imagens.")


def previous_image():
    global CURRENT_IMAGE_INDEX

    if IMAGES is not None and CURRENT_IMAGE_INDEX > 0:
        CURRENT_IMAGE_INDEX -= 1
        destroy_plots()

        if IsLoadImage:  # if regular image display histogram
            display_image_and_histogram(IMAGES[CURRENT_IMAGE_INDEX],
                                        title=f"Imagem {
                                            CURRENT_IMAGE_INDEX}",
                                        parent=image_frame)
        else:  # if ROI display the image in fullsize
            cut_rois(cv2.cvtColor(
                IMAGES[CURRENT_IMAGE_INDEX], cv2.COLOR_GRAY2BGR))
    else:
        print("Não há mais imagens.")


def reset_pagination():
    global CURRENT_IMAGE_INDEX

    # reset index & destroy plots
    CURRENT_IMAGE_INDEX = 0
    destroy_plots()


def go_to_image():
    global CURRENT_IMAGE_INDEX

    try:
        index = int(index_entry.get())

        # check if index is valid
        if IMAGES is not None and 0 <= index < len(IMAGES):
            CURRENT_IMAGE_INDEX = index
            destroy_plots()

            if IsLoadImage:  # if regular image display histogram
                display_image_and_histogram(IMAGES[CURRENT_IMAGE_INDEX],
                                            title=f"Imagem {
                                                CURRENT_IMAGE_INDEX}",
                                            parent=image_frame)
            else:  # if ROI display the image in fullsize
                cut_rois(cv2.cvtColor(
                    IMAGES[CURRENT_IMAGE_INDEX], cv2.COLOR_GRAY2BGR))
        else:
            print(f"Index inválido. Escolha um valor entre 0 e {
                  len(IMAGES) - 1}.")

    except ValueError:
        print("Insiva um número válido.")


# ----------------------------- FUNCTIONALITIES (Part 2) ----------------------------

# save .mat informations
def save_image(roi_image, classification):
    global patient, ultrasound
    # file name based on patient and ultrasound
    if classification == 0:
        file_name = f"ROI_{str(patient).zfill(2)}_{ultrasound}.jpg"
        file_path = os.path.join(os.getcwd(), file_name)
        image = Image.fromarray(roi_image)
        image.save(file_path, 'JPEG')

        if (ultrasound > 10):
            patient += 1
            ultrasound = 0
        else:
            ultrasound += 1
    else:
        file_name = f"current_patient_roi.jpg"
        file_path = os.path.join(os.getcwd(), file_name)
        # if the file exists, delete it
        if os.path.exists(file_path):
            os.remove(file_path)
        image = Image.fromarray(roi_image)
        image.save(file_path, 'JPEG')

# calc index HI value
def calc_HI(roi_kidney, roi_liver, coord_kidney, coord_liver, classification):
    
    average_kidney = np.mean(roi_kidney)
    average_liver = np.mean(roi_liver)

    hi_ratio = average_liver / average_kidney

    def show_hi_ratio_window(hi_ratio):
        hi_window = Toplevel(root)  
        hi_window.title("HI Ratio")
        hi_window.geometry("300x200")

        label = Label(hi_window, text=f"HI Ratio: {hi_ratio:.2f}\n"
                      f"Coord. Fígado: {coord_liver}\n"
                      f"Coord. Cortex Renal: {coord_kidney}", font=('Arial', 14))
        label.pack(pady=20)
        

    adjusted_roi_liver = roi_liver * hi_ratio

    # round values e convert to uint8
    adjusted_roi_liver = np.round(adjusted_roi_liver).astype(
        np.uint8)

    # values can't be greater than 255
    adjusted_roi_liver = np.clip(adjusted_roi_liver, 0, 255)
    
    if classification == 0:
        update_csv(f"ROI_{str(patient).zfill(2)}_{ultrasound}","Coordenadas Figado", f"{coord_liver[0]}, {coord_liver[1]}")
        update_csv(f"ROI_{str(patient).zfill(2)}_{ultrasound}", "Coordenadas Cortex Renal", f"{coord_kidney[0]}, {coord_kidney[1]}")
        update_csv(f"ROI_{str(patient).zfill(2)}_{ultrasound}", "HI",hi_ratio)
        
        save_image(adjusted_roi_liver, classification)
    else:
        global image_to_classify
        image_to_classify[0] = hi_ratio
        save_image(adjusted_roi_liver, classification)
    
    show_hi_ratio_window(hi_ratio)
    
# ROI -> Region of Interest
def cut_rois(image, classification=0):
    image_copy = image.copy()
    click_count = [0]

    roi_kidney = None
    roi_liver = None
    coord_kidney = None
    coord_liver = None

    fig, ax = plt.subplots(figsize=(10, 5))
    canvas = FigureCanvasTkAgg(fig, master=image_frame)
    canvas.get_tk_widget().pack(side=TOP, anchor="center",
                                padx=10, pady=10, fill="both", expand=True)

    toolbar_frame = Frame(image_frame)
    toolbar_frame.pack(side=TOP, anchor="center", padx=10, pady=10)
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()

    # show original image
    ax.set_title("Clique para selecionar a ROI do Figado")
    ax.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    canvas.draw()
    def click_event(event):
        nonlocal roi_kidney, roi_liver, coord_liver, coord_kidney

        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)

            # force ROI to be 28x28
            if x + 28 <= image.shape[1] and y + 28 <= image.shape[0]:
                roi_cropped = image[y:y + 28, x:x + 28]

                if click_count[0] == 0:
                    # select liver first
                    roi_liver = roi_cropped
                    coord_liver = (x - 14, y - 14)
                    display_roi(
                        roi_liver, title="ROI Figado (28x28)", parent=roi_liver_frame)

                    cv2.rectangle(image_copy, (x - 14, y - 14),
                                  (x + 14, y + 14), (0, 255, 0), 2)

                elif click_count[0] == 1:
                    # select kidney after
                    roi_rim = roi_cropped
                    coord_rim = (x - 14, y - 14)
                    display_roi(roi_rim, title="ROI Rim (28x28)",
                                parent=roi_kidney_frame)

                    cv2.rectangle(image_copy, (x - 14, y - 14),
                                  (x + 14, y + 14), (0, 255, 0), 2)

                    calc_HI(roi_rim, roi_liver, coord_rim, coord_liver, classification)

                # update image with roi draw
                ax.clear()
                ax.set_title("Clique para selecionar a ROI do cortex renal")
                ax.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
                canvas.draw()

                click_count[0] += 1

    # connect click event to canvas
    canvas.mpl_connect('button_press_event', click_event)

def select_rois():
    global IMAGES, CURRENT_IMAGE_INDEX, image_frame, canvas, IsLoadImage

    IsLoadImage = False
    image_path = filedialog.askopenfilename(
        filetypes=[("Imagens e MAT", "*.png *.jpg *.mat")]
    )
    
    if image_path:
        png = image_path.lower().endswith('.png')
        jpg = image_path.lower().endswith('.jpg')
        jpeg = image_path.lower().endswith('.jpeg')
        if png or jpg or jpeg:
            cut_rois(cv2.imread(image_path))
        elif image_path.lower().endswith('.mat'):
            mat = scipy.io.loadmat(image_path)
            data = mat['data']

            IMAGES = []

            for entry in data[0]:
                images_array = entry[-1]

                if images_array is not None:
                    for image in images_array:
                        IMAGES.append(image)

            CURRENT_IMAGE_INDEX = 0  # start at first image
            cut_rois(cv2.cvtColor(
                IMAGES[CURRENT_IMAGE_INDEX], cv2.COLOR_GRAY2BGR))

def calc_glcm(roi, distance):
    angles  = [0, 45, 90, 135, 180, 225, 270, 315]
    glcm = np.zeros((256, 256), dtype=np.float32)

    for angle in angles:
        # calc movement to current algle
        if angle == 0:
            y_offset, x_offset = 0, distance
        elif angle == 45:
            y_offset, x_offset = distance, distance
        elif angle == 90:
            y_offset, x_offset = distance, 0
        elif angle == 135:
            y_offset, x_offset = distance, -distance
        elif angle == 180:
            y_offset, x_offset = 0, -distance
        elif angle == 225:
            y_offset, x_offset = -distance, -distance
        elif angle == 270:
            y_offset, x_offset = -distance, 0
        elif angle == 315:
            y_offset, x_offset = -distance, distance
    
        for y in range(roi.shape[0]):
            for x in range(roi.shape[1]):
                y_neigh = y + y_offset
                x_neigh = x + x_offset
                
                if y_neigh < roi.shape[0] and x_neigh < roi.shape[1]:
                    pixel1 = roi[y, x]
                    pixel2 = roi[y_neigh, x_neigh]
                    glcm[pixel1, pixel2] += 1

    glcm = glcm / np.sum(glcm)
    return glcm

def calc_homogeneity(glcm):
    homogeneity = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            homogeneity += glcm[i, j] / (1 + abs(i - j))
    return homogeneity

def calc_entropy(glcm):
    entropy = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            if glcm[i, j] > 0:
                entropy -= glcm[i, j] * np.log2(glcm[i, j])
    return entropy

def compute_matrix():
    image_path = filedialog.askopenfilename(
        filetypes=[("Imagens e MAT", "*.png *.jpg *.mat")]
    )
    if image_path:
        i = [1,2,4,8]
        
        descriptors = []

        for j in i:
            glcm = calc_glcm(cv2.imread(image_path), j)
            homogeneity = calc_homogeneity(glcm)
            entropy =calc_entropy(glcm)
            descriptors.append({
                'distance': j,
                'homogeneity': homogeneity,
                'entropy': entropy,
                'glcm': glcm  # Armazena a GLCM
            })
            update_csv(os.path.splitext(os.path.basename(image_path))[0], f"Entropia i ={j}",f"{entropy}")
            update_csv(os.path.splitext(os.path.basename(image_path))[0], f"Homogeneidade i={j}",f"{homogeneity}")
                        
    def show_descriptors(descriptors):
        hi_window = Toplevel(root)  
        hi_window.title("Descritores")
        hi_window.geometry("400x300")

        for desc in descriptors:
            result_text = (f"Distância: {desc['distance']}\n"
                        f"Homogeneidade: {desc['homogeneity']:.4f}, Entropia: {desc['entropy']:.4f}\n")
            
            label = Label(hi_window, text=result_text, font=('Arial', 10))
            label.pack(pady=5)
        
    show_descriptors(descriptors)

def tamura():
    image_path = filedialog.askopenfilename(
        filetypes=[("Imagens e MAT", "*.png *.jpg *.mat")]
    )

    if image_path:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("Error: Imagem não pode ser carregada.")
            return

        coarseness = tamura_coarseness(img)
        contrast = tamura_contrast(img)
        directionality = tamura_directionality(img)
        line_likeness = tamura_line_likeness(img)
        regularity = tamura_regularity(img)
        roughness = tamura_roughness(img)

        update_csv(os.path.splitext(os.path.basename(image_path))[0], f"coarseness",f"{coarseness}")
        update_csv(os.path.splitext(os.path.basename(image_path))[0], f"contrast",f"{contrast}")
        update_csv(os.path.splitext(os.path.basename(image_path))[0], f"directionality",f"{directionality}")
        update_csv(os.path.splitext(os.path.basename(image_path))[0], f"line-likeness",f"{line_likeness}")
        update_csv(os.path.splitext(os.path.basename(image_path))[0], f"regularity",f"{regularity}")
        update_csv(os.path.splitext(os.path.basename(image_path))[0], f"roughness",f"{roughness}")
             
        def show_tamura():
            hi_window = Toplevel(root)  
            hi_window.title("Descritores")
            hi_window.geometry("400x300")

            result_text = (f"Tamura Features:\n"
                            f"Coarseness: {coarseness:.4f}\n"
                            f"Contrast: {contrast:.4f}\n"
                            f"Directionality: {directionality:.4f}\n"
                            f"Line-Likeness: {line_likeness:.4f}\n"
                            f"Regularity: {regularity:.4f}\n"
                            f"Roughness: {roughness:.4f}\n")
                
            label = Label(hi_window, text=result_text, font=('Arial', 10))
            label.pack(pady=5)   

        show_tamura()

def tamura_coarseness(img, k_max=5):
    # k_max (int): max scale (2^k)

    # array to store the average differences
    A = np.zeros((img.shape[0], img.shape[1], k_max))

    # calc average difference for each scale k
    for k in range(k_max):
        window_size = 2 ** k
        shifted_img_right = np.roll(img, -window_size, axis=1)
        shifted_img_left = np.roll(img, window_size, axis=1)
        shifted_img_up = np.roll(img, -window_size, axis=0)
        shifted_img_down = np.roll(img, window_size, axis=0)

        # diff along x-axis and y-axis
        S_x = np.abs(img - shifted_img_right) + np.abs(img - shifted_img_left)
        S_y = np.abs(img - shifted_img_up) + np.abs(img - shifted_img_down)

        # store average differences for scale k
        A[:, :, k] = (S_x + S_y) / 2

    # find max difference at each pixel
    F_coarseness = np.max(A, axis=2)

    return np.mean(F_coarseness)
    
def tamura_contrast(img):
    # mean and standard deviation
    mean = np.mean(img)
    std_dev = np.std(img)

    fourth_moment = np.mean((img - mean) ** 4)

    # to avoid division by zero
    if std_dev != 0:
        contrast = std_dev / (fourth_moment ** 0.25)
    else:
        contrast = 0 

    return contrast

def tamura_directionality(img, num_bins=16):
    # num_bins (int): defines how finely the angle range is divided into discrete intervals

    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # angle and magnitude
    angles = np.arctan2(gy, gx)
    magnitude = np.sqrt(gx**2 + gy**2)

    # convert radians to degrees
    angles_deg = np.degrees(angles) % 180

    # histogram of edge directions weighted by magnitude
    hist, _ = np.histogram(angles_deg, bins=num_bins, range=(0, 180), weights=magnitude)
    hist = hist / np.sum(hist)

    # calc directionality
    bin_centers = (np.arange(num_bins) + 0.5) * (180 / num_bins)
    directionality = np.sum(hist * (bin_centers - np.sum(hist * bin_centers))**2)

    return directionality

def tamura_line_likeness(img):
    h, w = img.shape
    
    # gradients on x and y directions
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # angle and magnitude
    angles = np.arctan2(gy, gx)
    magnitude = np.sqrt(gx**2 + gy**2)

    # convert radians to degrees
    angles_deg = np.degrees(angles) % 180

    line_likeness = 0
    count = 0

    # calc line-likeness
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if magnitude[i, j] > 0:
                # get angle of central pixel
                central_angle = angles_deg[i, j]
                aligned_neighbors = 0
                
                # check alignment with 8 neighbor pixels
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue  # skip central pixel
                        neighbor_angle = angles_deg[i + di, j + dj]

                        angle_diff = abs(central_angle - neighbor_angle)
                        angle_diff = min(angle_diff, 180 - angle_diff)

                        if angle_diff < 20: 
                            aligned_neighbors += 1
                
                line_likeness += aligned_neighbors / 8
                count += 1

    # to avoid division by zero
    if count > 0:
        return line_likeness / count
    else:
        return 0

def tamura_regularity(img):
    coarseness = tamura_coarseness(img)
    contrast = tamura_contrast(img)
    directionality = tamura_directionality(img)
    line_likeness = tamura_line_likeness(img)

    # standard deviation
    std_devs = np.array([np.std([coarseness]), 
                         np.std([contrast]), 
                         np.std([directionality]), 
                         np.std([line_likeness])])
    
    # calc regularity with avoiding division by zero per feature
    regularity_contributions = []
    for std_dev in std_devs:
        if std_dev != 0:
            regularity_contributions.append(1 / std_dev)
        else:
            regularity_contributions.append(0)
    
    # sum contributions to get the regularity score
    regularity = np.sum(regularity_contributions)

    return regularity

def tamura_roughness(img):
    coarseness = tamura_coarseness(img)
    contrast = tamura_contrast(img)

    roughness = coarseness * contrast
    
    return roughness

def on_closing():
    root.quit()
    root.destroy()

# --------------------------------- XGBOOST --------------------------------------

def load_roi_csv_info(csv_path="rois_informations.csv"):
    global class_data
    global caracteristic_data

    df = pd.read_csv(csv_path, delimiter=";")
    caracteristic_colums = df.columns[4:]

    df['Paciente'] = df['Nome do arquivo'].apply(lambda x: x.split('_')[0])

    df['Classe'] = df['Classe'].replace('nan', pd.NA)
    df = df.dropna(subset=['Classe'])
    
    df['Classe'] = df['Classe'].map(class_mapping)

    # convert for calculations - excluding coodinates
    num_columns = df.columns[4:]
    for col in num_columns:
        try:
            df[col] = df[col].replace(",", ".", regex=True).astype(float)
            df[col] = df[col].apply(pd.to_numeric)
        except ValueError:
            pass
    
    class_data = {
        row["Nome do arquivo"]: row["Classe"]
        for _, row in df.iterrows()
    }

    caracteristic_data = {
        row["Nome do arquivo"]: row[caracteristic_colums].values
        for _, row in df.iterrows()
    } 

def xgboost():
    global aggregated_cm

    if class_data:
        # cross validation  
        for test_patient in class_data:
            X_train = []
            X_test = []
            y_train = []
            y_test = []

            # split train and test
            for item in class_data:
                # train data
                if test_patient in item:
                    X_test.append(caracteristic_data[item])
                    y_test.append(class_data[item])
                # test data
                else:
                    X_train.append(caracteristic_data[item])
                    y_train.append(class_data[item])

            params = {
                'colsample_bytree': 0.7,
                'gamma': 0.1,
                'learning_rate': 0.01,
                'max_depth': 3,
                'min_child_weight': 3,
                'n_estimators': 500,
                'subsample': 1
            }

            model = xg.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **params)
        
            warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            accuracy = accuracy_score(y_test, y_pred)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0 # true positive
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 # true negative

            accuracy_results.append(accuracy)
            sensitivities_results.append(sensitivity)
            specificities_results.append(specificity)
            aggregated_cm += cm
            XGmodels.append(model)

            # show_confusion_matrix_per_patient(test_patient, y_test, y_pred, accuracy, sensitivity, specificity)
        show_confusion_matrix_all_patients()

    else:
        messagebox.showerror("Erro", f"Dados das ROIs ainda não extraidos.")

def show_confusion_matrix_per_patient(test_patient, y_test, y_pred, accuracy, sensitivity, specificity):
    print(f"Confusion Matrix for test_patient {test_patient}:")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(cm)
    
    # Print results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Sensitivity (Recall): {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    
    # Print classification report for additional metrics (precision, recall, f1-score)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Positive', 'Negative'], zero_division=0))

    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
    plt.title(f"Confusion Matrix for {test_patient}")
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.show()

def show_confusion_matrix_all_patients():
    global aggregated_cm

    # Display cumulative confusion matrix
    print("Aggregated Confusion Matrix:")
    print(aggregated_cm)

    avg_accuracy = np.mean(accuracy_results)
    avg_sensitivity = np.mean(sensitivities_results)
    avg_specificity = np.mean(specificities_results)

    print(f"\nAverage Accuracy: {avg_accuracy:.2f}")
    print(f"Average Sensitivity: {avg_sensitivity:.2f}")
    print(f"Average Specificity: {avg_specificity:.2f}")

    # Visualize the aggregated confusion matrix
    sns.heatmap(aggregated_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Aggregated Confusion Matrix')
    plt.show()

# AJUSTAR PARA QUALQUER IMAGEM
def xgboost_classification(csv_path="rois_informations.csv"):
    load_roi_csv_info()

    # check if theirs trained models
    if not XGmodels:
        xgboost()
      
    # read image
    image_path = filedialog.askopenfilename(filetypes=[("Imagens", "*.png *.jpg")])
    df = pd.read_csv(csv_path, delimiter=";")
    column = 'Nome do arquivo'

    if image_path:
        start = image_path.rfind("/") + 1  # Add 1 to exclude the '/'
        end = image_path.rfind(".")
        image = image_path[start:end]        # Don't include the '.'
        if image in df[column].values:
            # extrair classificadores da imagem (HI, entropia, homogeneidade, ...)
            image_row = df[df['Nome do arquivo'] == image]
            image_data = image_row.iloc[:, 4:]
            image_data = image_data.replace(",", ".", regex=True).astype(float)
            image_data = image_data.apply(pd.to_numeric)
        elif image == 'current_patient_roi':
            image_data = classify_new_image(image_path)
            # print(image_data)
        else:
            messagebox.showerror("Erro", f"Primeiro corte a ROI da imagem ou escolha uma imagem presente no csv.")


    # classificar a imagem
    preds = [model.predict(image_data) for model in XGmodels]
    # Average the predictions across all models
    avg_preds = np.mean(preds, axis=0)
    class_names = [inverse_class_mapping[pred] for pred in avg_preds] # final prediction
    def classification_status():
        hi_window = Toplevel(root)  
        hi_window.title("Classificação")
        hi_window.geometry("400x100")
                
        label = Label(hi_window, text=f"Classe prevista: {class_names}", font=('Arial', 12))
        label.pack(pady=5)
    classification_status()
   

def xgboost_cut_image():
    image_path = filedialog.askopenfilename(filetypes=[("Imagens", "*.png *.jpg")])
    
    # calc HI
    cut_rois(cv2.imread(image_path), 1)

def classify_new_image(roi_path):
    global image_to_classify
    img = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)

    # calc matrix
    i = [1,2,4,8]    
    descriptors = []
    index = 1
    for j in i:
        glcm = calc_glcm(img, j)
        homogeneity = calc_homogeneity(glcm)
        entropy = calc_entropy(glcm)
        descriptors.append({
            'distance': j,
            'homogeneity': homogeneity,
            'entropy': entropy,
            'glcm': glcm
        })
        image_to_classify[index] = entropy
        index = index + 1
        image_to_classify[index] = homogeneity
        index = index + 1

    def show_descriptors(descriptors):
        hi_window = Toplevel(root)  
        hi_window.title("Descritores")
        hi_window.geometry("400x300")

        for desc in descriptors:
            result_text = (f"Distância: {desc['distance']}\n"
                            f"Homogeneidade: {desc['homogeneity']:.4f}, Entropia: {desc['entropy']:.4f}\n")
            
            label = Label(hi_window, text=result_text, font=('Arial', 10))
            label.pack(pady=5)
    show_descriptors(descriptors)

    # calc tamura
    coarseness = tamura_coarseness(img)
    contrast = tamura_contrast(img)
    directionality = tamura_directionality(img)
    line_likeness = tamura_line_likeness(img)
    regularity = tamura_regularity(img)
    roughness = tamura_roughness(img)

    image_to_classify[9] = coarseness
    image_to_classify[10] = contrast
    image_to_classify[11] = directionality
    image_to_classify[12] = line_likeness
    image_to_classify[13] = regularity
    image_to_classify[14] = roughness

    # convert array (image_to_classify) to dataframe (data)
    columns = ['HI', 'Entropia i =1', 'Homogeneidade i=1', 'Entropia i =2', 'Homogeneidade i=2', 
                'Entropia i =4', 'Homogeneidade i=4', 'Entropia i =8', 'Homogeneidade i=8', 'coarseness',
                'contrast', 'directionality', 'line-likeness', 'regularity', 'roughness']
    data = pd.DataFrame([image_to_classify], columns=columns)
    data = data.apply(pd.to_numeric)

    os.remove(roi_path)
    return data

# --------------------------------- MOBILE NET --------------------------------------
def listar_arquivos(folder = "dataset"):
    try:      
        arquivos = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        print(f'Total de arquivos: {len(arquivos)}')
        return arquivos
    except: 
        print(f"Erro ao pegar ROIs")
        return []

def organize_dataset(class_data, test_patient):
    base_dir = "dataset"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    for folder in [train_dir, val_dir]:
        if os.path.exists(folder):
            shutil.rmtree(folder)  
        os.makedirs(folder, exist_ok=True) 
        for class_name in ["saudavel", "possui_esteatose"]:
            os.makedirs(os.path.join(folder, class_name), exist_ok=True) 
    
    for item in class_data:
        if test_patient in item:
            target_dir = val_dir
        else:
            target_dir = train_dir
        
        class_name = "saudavel" if int(item.split('_')[1]) <= 16 else "possui_esteatose"

        origem = f"dataset/{item}"
        destino = os.path.join(target_dir, class_name, item)
        destino = os.path.join(target_dir, class_name)
        if not os.path.exists(destino):
            print(f"Erro: Diretório de destino não existe - {destino}")
            continue  
        if os.path.exists(origem):
            shutil.copy(origem, destino)
        else:
            print(f"Erro: Arquivo de origem não encontrado - {origem}")
    
def train_mobile_net():
    global aggregated_cm, accuracy_results,sensitivities_results,specificities_results,treinamento_acuracia,validacao_acuracia
    
    aggregated_cm = [[0, 0], [0, 0]]
    accuracy_results = []
    class_names = None
    class_data = listar_arquivos();
    if class_data:  
        for test_patient in roi_data:
        
            # Organize the dataset for cross validation
            organize_dataset(class_data,test_patient)
            
            train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                "dataset/train",
                image_size=(224, 224),
                batch_size=32
            )

            val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                "dataset/val",
                image_size=(224, 224),
                batch_size=32
            )

            class_names = train_dataset.class_names
            print("Classes:", class_names)

            #Load the pre-trained model
            pre_treined_model = MobileNetV2(weights='imagenet', include_top=False)
            pre_treined_model.trainable = False

            #Model training
            mobileNetModel = models.Sequential([
                pre_treined_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.35),
                layers.Dense(1, activation="sigmoid") 
            ])

            optimizer = RMSprop(learning_rate=0.0001)

            mobileNetModel.compile(optimizer=optimizer,
                        loss="binary_crossentropy",
                        metrics=["accuracy"])
            
            labels = []
            for _, batch_labels in train_dataset:
                labels.extend(batch_labels.numpy())

            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(labels),  
                y=labels  
            )

            class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
            
            mobileNetModel.summary()
            history = mobileNetModel.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=10,
                class_weight=class_weights_dict,
                verbose=1)


            mobileNetModel.compile(optimizer=optimizer,
                        loss="binary_crossentropy",
                        metrics=["accuracy"])

            mobileNetModel.summary()
           
           # Salvar o modelo treinado
            model_name = f"mobileNetModels/mobileNet_model_patient_{test_patient}.h5"  # Nome personalizado para cada modelo
            mobileNetModel.save(model_name)
            print(f"Modelo salvo como {model_name}")
           
            show_metrics(val_dataset, history, mobileNetModel, class_names)
            treinamento_acuracia.append(history.history['accuracy'])
            validacao_acuracia.append(history.history['val_accuracy'])
        show_final_metrics(class_names)
        base_dir = "dataset"
        train_dir = os.path.join(base_dir, "train")
        val_dir = os.path.join(base_dir, "val")
        
        for folder in [train_dir, val_dir]:
            if os.path.exists(folder):
                shutil.rmtree(folder)  
    

def show_metrics(val_dataset, history, mobileNetModel,class_names):

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
    class_report = classification_report(y_true, y_pred, target_names=class_names, labels=[0, 1])
    print("Relatório de Classificação:")
    print(class_report)

    # # Plot the confusion matrix
    # plt.figure(figsize=(8, 6))
    # plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title("Matriz de Confusão")
    # plt.colorbar()
    # tick_marks = np.arange(len(class_names))
    # plt.xticks(tick_marks, class_names, rotation=45)
    # plt.yticks(tick_marks, class_names)

    # plt.xlabel('Predição')
    # plt.ylabel('Verdadeiro')
    # plt.tight_layout()
    # plt.show()

    # # Plot accuracy graph
    # plt.figure(figsize=(10, 6))
    # plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    # plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    # plt.title('Acurácia de Treinamento e Validação')
    # plt.xlabel('Época')
    # plt.ylabel('Acurácia')
    # plt.legend(loc='lower right')
    # plt.grid(True)
    # plt.show()

def show_final_metrics(class_names):
    global aggregated_cm, accuracy_results, treinamento_acuracia, validacao_acuracia
    
    avg_accuracy = np.mean(accuracy_results)
    print(f"Acurácia: {avg_accuracy * 100:.2f}%")


    print("Matriz de Confusão:")
    print(aggregated_cm)


    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(aggregated_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.xlabel('Predição')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.show()
    
    media_treinamento = np.mean(np.array(treinamento_acuracia), axis=0)
    media_validacao = np.mean(np.array(validacao_acuracia), axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(media_treinamento, label='Acurácia de Treinamento')
    plt.plot(media_validacao, label='Acurácia de Validação')
    plt.title('Acurácia de Treinamento e Validação')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def mobile_net_classification():   
    mobileNetModels = []
    for test_patient in class_data:
        model_name = f"mobileNetModels/mobileNet_model_patient_{test_patient}.h5" 
        if os.path.exists(model_name):
            print(f"Carregando o modelo: {model_name}")
            mobileNetModels.append(load_model(model_name))
        else:
            print(f"Modelo não encontrado: {model_name}")

    image_path = filedialog.askopenfilename(filetypes=[("Imagens e MAT", "*.png *.jpg *.mat")])

    if image_path:
        img = cv2.imread(image_path)

        if img is None:
            print("Error: Imagem não pode ser carregada.")
            return
            
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        predictions = []
        class_idx = []

        for model in mobileNetModels:
            predictions.append(model.predict(img))
            class_idx.append(np.argmax(predictions))
        class_idx_max = np.argmax(np.mean(predictions, axis=0))
        mean_predictions = np.mean(predictions, axis=0)
        probability = mean_predictions[class_idx_max]

        def classification_status():
            hi_window = Toplevel(root)  
            hi_window.title("Classificação")
            hi_window.geometry("400x100")
                
            label = Label(hi_window, text=f"Classe prevista: {inverse_class_mapping[int(class_idx_max)]} com probabilidade {probability}", font=('Arial', 10))
            label.pack(pady=5)
        classification_status()
        

# ------------------------------------ GUI ------------------------------------------

customtkinter.set_appearance_mode("light")  # system, light, dark
customtkinter.set_default_color_theme("green")  # blue, dark-blue, green

root = customtkinter.CTk()

# default open settings
root.geometry("1760x960")
root.title('Sistema Auxiliar de Diagnóstico da NAFLD')
label = Label(
    root, text='Sistema Auxiliar de Diagnóstico da NAFLD', font=('Arial', 14))
label.pack(pady=10)
root.protocol("WM_DELETE_WINDOW", on_closing)

# Buttons Grid 1 ---------------------------------------------------------------- Start

button_frame = Frame(root)
button_frame.pack(pady=10)

load_images_button = Button(
    button_frame, text='Carregar Imagens', command=load_file)
load_images_button.grid(row=0, column=0, padx=5)  # load files

load_roi_button = Button(
    button_frame, text='Vizualizar ROIs', command=lambda: load_file(True))
load_roi_button.grid(row=0, column=1, padx=5)  # load files (ROI)

roi_button = Button(button_frame, text='Selecionar ROI', command=select_rois)
roi_button.grid(row=0, column=2, padx=5)  # select ROI

matrix_button = Button(button_frame, text='Computar Matriz de co-ocorrência', command=compute_matrix)
matrix_button.grid(row=0, column=3, padx=5)

tamura_button = Button(button_frame, text='Descritores Tamura', command=tamura)
tamura_button.grid(row=0, column=4, padx=5)

previous_image_button = Button(
    button_frame, text='Imagem Anterior', command=previous_image)
previous_image_button.grid(row=0, column=5, padx=5)  # previous image btn

next_image_button = Button(
    button_frame, text='Próxima Imagem', command=next_image)
next_image_button.grid(row=0, column=6, padx=5)  # next image btn

reset_pagination_button = Button(
    button_frame, text='Limpar Fluxo', command=reset_pagination)
reset_pagination_button.grid(
    row=0, column=7, padx=5)  # reset pagination btn

index_entry = Entry(button_frame, width=5)
index_entry.grid(row=0, column=8, padx=5)  # user pagination input

go_to_image_button = Button(
    button_frame, text='Ir para Imagem', command=go_to_image)
go_to_image_button.grid(row=0, column=9, padx=5)  # go to index

# Buttons Grid 1 ------------------------------------------------------------------ End

# Buttons Grid 2 ---------------------------------------------------------------- Start

button_frame_class = Frame(root)
button_frame_class.pack(pady=10)

cut_xgboost = Button(button_frame_class, text='Cut New Image ROI', command=xgboost_cut_image)
cut_xgboost.grid(row=1, column=0, padx=5)

classify_xgboost = Button(button_frame_class, text='Classificar Imagem (XGBoost)', command=xgboost_classification)
classify_xgboost.grid(row=1, column=1, padx=5)

train_button = Button(button_frame_class, text='Treinar modelos com Mobile Net', command=train_mobile_net)
train_button.grid(row=1, column=2, padx=5)

classificar_button = Button(button_frame_class, text='Classificar Imagem com Mobile Net', command=mobile_net_classification)
classificar_button.grid(row=1, column=3, padx=5)


# Buttons Grid 2 ------------------------------------------------------------------ End

# Canvas -> Scroll + Image Frames --------------------------------------------- Start

canvas_frame = Frame(root)
canvas_frame.pack(fill=BOTH, expand=True, pady=10, padx=10)

canvas = Canvas(canvas_frame)
canvas.pack(side=LEFT, fill=BOTH, expand=True)

scrollbar = Scrollbar(canvas_frame, orient=VERTICAL, command=canvas.yview)
scrollbar.pack(side=RIGHT, fill=Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(
    scrollregion=canvas.bbox("all")))

content_frame = Frame(canvas)
canvas.create_window((0, 0), window=content_frame, anchor="nw")

image_frame = Frame(content_frame, width=1000, height=600)
image_frame.pack(pady=10, padx=10)
image_frame.pack_propagate(False)

# ROIs Grid ------------------------------------------------------------------- Start

roi_frame = Frame(content_frame, width=1000, height=400)
roi_frame.pack(pady=10, padx=10)

roi_liver_frame = Frame(roi_frame, width=600, height=360)
roi_liver_frame.grid(row=0, column=0)
roi_liver_frame.pack_propagate(False)

roi_kidney_frame = Frame(roi_frame, width=600, height=360)
roi_kidney_frame.grid(row=0, column=1)
roi_kidney_frame.pack_propagate(False)

# ROIs Grid --------------------------------------------------------------------- End

# Canvas -> Scroll + Image Frames ----------------------------------------------- End

# Start the GUI Loop
root.mainloop()
