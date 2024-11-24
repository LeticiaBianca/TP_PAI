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
from tensorflow.keras.optimizers import Adam


# ----------------------------- GLOBAL VARIABLES ------------------------------------

CURRENT_IMAGE_INDEX = 0  # pagination
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
inverse_class_mapping = {v: k for k, v in class_mapping.items()}

# ----------------------------- FUNCTIONALITIES (Part 1) ----------------------------
def update_excel(file_name, column, value):
   
    file_name = "rois_informations.xlsx"
    file_path = os.path.join(os.getcwd(), file_name)

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
def save_image(roi_image):
    global patient, ultrasound
    # file name based on patient and ultrasound
    file_name = f"ROI_{str(patient).zfill(2)}_{ultrasound}.jpg"
    file_path = os.path.join(os.getcwd(), file_name)
    image = Image.fromarray(roi_image)
    image.save(file_path, 'JPEG')

    if (ultrasound > 10):
        patient += 1
        ultrasound = 0
    else:
        ultrasound += 1

# calc index HI value
def calc_HI(roi_kidney, roi_liver, coord_kidney, coord_liver):
    
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
    
    update_excel(f"ROI_{str(patient).zfill(2)}_{ultrasound}","Coordenadas Fígado", f"{coord_liver[0]}, {coord_liver[1]}")
    update_excel(f"ROI_{str(patient).zfill(2)}_{ultrasound}", "Coordenadas Cortex Renal", f"{coord_kidney[0]}, {coord_kidney[1]}")
    update_excel(f"ROI_{str(patient).zfill(2)}_{ultrasound}", "HI",hi_ratio)
    
    save_image(adjusted_roi_liver)

    show_hi_ratio_window(hi_ratio)
    

# ROI -> Region of Interest
def cut_rois(image):
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

                    calc_HI(roi_rim, roi_liver, coord_rim, coord_liver)

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
        if image_path.lower().endswith(('*.png', '*.jpg', '*.jpeg')):
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
            update_excel(os.path.splitext(os.path.basename(image_path))[0], f"Entropia i ={j}",f"{entropy}")
            update_excel(os.path.splitext(os.path.basename(image_path))[0], f"Homogeneidade i={j}",f"{entropy}")
                        
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

        update_excel(os.path.splitext(os.path.basename(image_path))[0], f"coarseness",f"{coarseness}")
        update_excel(os.path.splitext(os.path.basename(image_path))[0], f"contrast",f"{contrast}")
        update_excel(os.path.splitext(os.path.basename(image_path))[0], f"directionality",f"{directionality}")
        update_excel(os.path.splitext(os.path.basename(image_path))[0], f"line-likeness",f"{line_likeness}")
        update_excel(os.path.splitext(os.path.basename(image_path))[0], f"regularity",f"{regularity}")
        update_excel(os.path.splitext(os.path.basename(image_path))[0], f"roughness",f"{roughness}")
             
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

# def process_images_to_dataframe(image_directory=""):
#     for label, file_list in roi_data.items():
#         processed_data[label] = []
#         for file_name in file_list:
#             file_path = os.path.join(image_directory, f"{file_name}.jpg")
#             if os.path.exists(file_path):
#                 # convert image to matrix
#                 image = Image.open(file_path)
#                 image_array = np.array(image).flatten().tolist()  # flatten and convert to list
#                 processed_data[label].append(image_array)
#             else:
#                 print(f"Imagem não encontrada: {file_path}")
#     print_dataframe()

# def print_dataframe(): 
#     for label, images in processed_data.items():
#         print(f"Paciente: {label}")
#         for idx, image_array in enumerate(images):
#             print(f"  Imagem {idx}: {image_array[:10]}... (total de {len(image_array)} pixels)")

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
    # print(class_data)

def xgboost():
    global aggregated_cm

    if class_data:
        # cross validation  
        for test_patient in roi_data:
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

            model = xg.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
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
    # ler a imagem
    image = 'ROI_00_0'

    # extrair classificadores da imagem (HI, entropia, homogeneidade, ...)
    df = pd.read_csv(csv_path, delimiter=";")
    image_row = df[df['Nome do arquivo'] == image]
    image_data = image_row.iloc[:, 4:]
    image_data = image_data.replace(",", ".", regex=True).astype(float)
    image_data = image_data.apply(pd.to_numeric)

    # classificar a imagem
    preds = [model.predict(image_data) for model in XGmodels]
    # Average the predictions across all models
    avg_preds = np.mean(preds, axis=0)
    class_names = [inverse_class_mapping[pred] for pred in avg_preds]
    print(class_names)

# --------------------------------- MOBILE NET --------------------------------------
def listar_arquivos(folder = "dataset"):
    try:      
        arquivos = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        print(f'Total de arquivos: {len(arquivos)}')
        return arquivos
    except: 
        print(f"Erro ao pegar ROIs")
        return []

def create_dataset(image_paths, labels):

    def preprocess_image(filepath, label):
        img = tf.io.read_file("dataset/" + filepath)
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
    
    dataset = dataset.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
    
def train_mobile_net():
    global mobileNetModels, aggregated_cm, accuracy_results,sensitivities_results,specificities_results
    aggregated_cm = [[0, 0], [0, 0]]
    accuracy_results = []
    class_data = listar_arquivos();
    if class_data:  
        for test_patient in roi_data:
           
            val_dataset_images = []
            val_dataset_labels = []
            train_dataset_images = []
            train_dataset_labels = []

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
            
            print(f"Paciente {test_patient} - Validação Classe: {np.bincount(val_dataset_labels)}")
            print(f"Tamanho do Treino: {len(train_dataset_labels)}")
            print(f"Distribuição no Treino: {np.bincount(train_dataset_labels)}")

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
                layers.Dense(1, activation="sigmoid") 
            ])

            mobileNetModel.compile(optimizer="adam",
                        loss="binary_crossentropy",
                        metrics=["accuracy"])

            mobileNetModel.summary()
            class_weights = {0: 1.0, 1: len(train_dataset_labels) / np.bincount(train_dataset_labels)[1]}
            history = mobileNetModel.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=10,
                verbose=1,
                class_weight=class_weights)

            
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

    # # Plot the confusion matrix
    # plt.figure(figsize=(8, 6))
    # plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title("Matriz de Confusão")
    # plt.colorbar()
    # tick_marks = np.arange(len(class_mapping))
    # plt.xticks(tick_marks, class_mapping.keys(), rotation=45)
    # plt.yticks(tick_marks, class_mapping.keys())

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

def mobile_net_classification():
    global mobileNetModels
    if len(mobileNetModels) == 0:
        train_mobile_net()

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

classificar_button = Button(button_frame, text='Classificar Imagem com Mobile Net', command=mobile_net_classification)
classificar_button.grid(row=0, column=5, padx=5)

previous_image_button = Button(
    button_frame, text='Imagem Anterior', command=previous_image)
previous_image_button.grid(row=0, column=6, padx=5)  # previous image btn

next_image_button = Button(
    button_frame, text='Próxima Imagem', command=next_image)
next_image_button.grid(row=0, column=7, padx=5)  # next image btn

reset_pagination_button = Button(
    button_frame, text='Limpar Fluxo', command=reset_pagination)
reset_pagination_button.grid(
    row=0, column=8, padx=5)  # reset pagination btn

index_entry = Entry(button_frame, width=5)
index_entry.grid(row=0, column=9, padx=5)  # user pagination input

go_to_image_button = Button(
    button_frame, text='Ir para Imagem', command=go_to_image)
go_to_image_button.grid(row=0, column=10, padx=5)  # go to index

# Buttons Grid 1 ------------------------------------------------------------------ End

# Buttons Grid 2 ---------------------------------------------------------------- Start

button_frame_class = Frame(root)
button_frame_class.pack(pady=10)

# convert_roi_1d_array = Button(
#     button_frame_class, text='Converter ROIs', command=process_images_to_dataframe)
# convert_roi_1d_array.grid(row=1, column=0, padx=5)

load_roi_csv = Button(
    button_frame_class, text='Extrair dados das ROIs (csv)', command=load_roi_csv_info)
load_roi_csv.grid(row=1, column=0, padx=5)

run_xgboost = Button(
    button_frame_class, text='XGBoost', command=xgboost)
run_xgboost.grid(row=1, column=1, padx=5)

classify_xgboost = Button(
    button_frame_class, text='Classificar Imagem (XGBoost)', command=xgboost_classification)
classify_xgboost.grid(row=1, column=2, padx=5)

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
