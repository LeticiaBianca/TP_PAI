# Letícia Bianca Oliveira - 776782
# Raick Miranda Rodrigues Santos - 781755
# Nathália Mascarenhas Tenaglia - 766430
# NT = (776782 + 781755 + 766430) mod 4 = 3

# ------------------------------------ LIBRARIES & MODULES --------------------------

# GUI & OS demands
import os
import customtkinter
from tkinter import *
from tkinter import filedialog

# Data Handling
import cv2
import scipy.io
from PIL import Image
import numpy as np
from scipy.ndimage import uniform_filter

# Image Plots
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import pandas as pd

# ----------------------------- GLOBAL VARIABLES ------------------------------------

CURRENT_IMAGE_INDEX = 0  # Pagination
IMAGES = None

IsLoadImage = True
paciente = 0
ultrassom = 0

# ----------------------------- FUNCTIONALITIES (Part 1) ----------------------------
def update_excel(nome_arquivo, nome_coluna, valor):
   

    file_path = "rois_informations.xlsx"

    df = pd.read_excel(file_path)

    if nome_arquivo not in df.iloc[:, 0].values:
        print(f"O arquivo '{nome_arquivo}' não foi encontrado.")
        return

    if nome_coluna not in df.columns:
        print(f"A coluna '{nome_coluna}' não foi encontrada.")
        return

    idx = df[df.iloc[:, 0] == nome_arquivo].index[0]
    \
    try:
        df.at[idx, nome_coluna] = valor

        df.to_excel(file_path, index=False)
        print(f"Valor atualizado na célula correspondente ao arquivo '{nome_arquivo}' e coluna '{nome_coluna}'.")
    except Exception as e:
        print(f"Erro ao salvar o arquivo Excel: {idx}")
        

# Plot images along with histogram


def display_image_and_histogram(image, title="Imagem", parent=None, isROI=False):
    average_brightness = np.mean(image)
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(10, 5))

    # Image Settings
    ax_img.imshow(image, cmap='gray')
    ax_img.axis('off')
    ax_img.set_title(title)

    # Histogram Settings
    n, bins, patches = ax_hist.hist(
        image.ravel(), bins=256, color='gray', alpha=0.7)
    ax_hist.set_xlim([0, 255])
    ax_hist.set_ylim([0, max(n) * 1.1])
    ax_hist.set_title('Histograma')
    ax_hist.set_xlabel('Intensidade de Pixel')
    ax_hist.set_ylabel('Frequência')

    # If it's an ROI, display the average brightness line
    if isROI:
        ax_hist.axvline(average_brightness, color='red',
                        linestyle='dashed', linewidth=1)
        ax_hist.text(average_brightness + 5, max(n) * 0.9,
                     f'Avg: {average_brightness:.2f}', color='red')

    plt.tight_layout()

    # Embed image plots
    if parent is not None:
        for widget in parent.winfo_children():
            widget.destroy()  # Remove previous plot, if exists

        # Canvas
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, anchor="center",
                                    padx=10, pady=10, fill="both", expand=True)

        # Toolbar
        toolbar_frame = Frame(parent)
        toolbar_frame.pack(side=TOP, anchor="center", padx=10, pady=10)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()


def display_image(image, title="Imagem", parent=None):
    display_image_and_histogram(  # Regular images
        image, title, parent, isROI=False)


def display_roi(roi, title="ROI", parent=None):  # ROI images
    display_image_and_histogram(roi, title, parent, isROI=True)

# Load image files from computer


def load_file(isROI=False):
    global IMAGES, CURRENT_IMAGE_INDEX, IsLoadImage
    IsLoadImage = True

    file_path = filedialog.askopenfilename(
        filetypes=[("Imagens", "*.png;*.jpg"), ("MAT files", "*.mat")]
    )

    if file_path:  # Handling [.png/.jpg/.jpeg] files
        if file_path.lower().endswith(('*.png', '*.jpg', '*.jpeg')):
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if isROI:  # Is it an ROI?
                display_roi(image, title="Imagem Local", parent=roi_frame)
            else:
                display_image(
                    image, title="Imagem Local", parent=image_frame)

        # Handling [.mat] files
        elif file_path.lower().endswith('.mat'):
            mat = scipy.io.loadmat(file_path)
            data = mat['data']  # Images are stored in 'data' key

            IMAGES = []

            for entry in data[0]:  # Images are stored in the last element of each entry
                images_array = entry[-1]

                if images_array is not None:
                    for image in images_array:
                        IMAGES.append(image)

            CURRENT_IMAGE_INDEX = 0  # Start at the first image

            if isROI:  # Is it an ROI?
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

# Skip to the next image in the current pagination


def next_image():
    global CURRENT_IMAGE_INDEX

    # Check if IMAGES is not empty and if there is a next image
    if IMAGES is not None and CURRENT_IMAGE_INDEX < len(IMAGES) - 1:
        CURRENT_IMAGE_INDEX += 1
        destroy_plots()

        if IsLoadImage:  # Is it a regular image? Then display histogram
            display_image_and_histogram(IMAGES[CURRENT_IMAGE_INDEX],
                                        title=f"Imagem {
                                            CURRENT_IMAGE_INDEX}",
                                        parent=image_frame)
        else:  # Is it for selecting ROIs? Then display the image only in its fullsize
            cut_rois(cv2.cvtColor(
                IMAGES[CURRENT_IMAGE_INDEX], cv2.COLOR_GRAY2BGR))
    else:
        print("There are no more images to show.")


def previous_image():
    global CURRENT_IMAGE_INDEX

    if IMAGES is not None and CURRENT_IMAGE_INDEX > 0:
        CURRENT_IMAGE_INDEX -= 1
        destroy_plots()

        if IsLoadImage:  # Is it a regular image? Then display histogram
            display_image_and_histogram(IMAGES[CURRENT_IMAGE_INDEX],
                                        title=f"Imagem {
                                            CURRENT_IMAGE_INDEX}",
                                        parent=image_frame)
        else:  # Is it for selecting ROIs? Then display the image only in its fullsize
            cut_rois(cv2.cvtColor(
                IMAGES[CURRENT_IMAGE_INDEX], cv2.COLOR_GRAY2BGR))
    else:
        print("There are no previous images to show.")


def reset_pagination():
    global CURRENT_IMAGE_INDEX

    # Reset index & destroy plots
    CURRENT_IMAGE_INDEX = 0
    destroy_plots()


def go_to_image():
    global CURRENT_IMAGE_INDEX

    try:
        # Get the index from the input field
        index = int(index_entry.get())

        # Check if the index is within the valid range
        if IMAGES is not None and 0 <= index < len(IMAGES):
            CURRENT_IMAGE_INDEX = index
            destroy_plots()

            # Display image and histogram or allow ROI selection
            if IsLoadImage:  # Display image and histogram
                display_image_and_histogram(IMAGES[CURRENT_IMAGE_INDEX],
                                            title=f"Imagem {
                                                CURRENT_IMAGE_INDEX}",
                                            parent=image_frame)
            else:  # Display full image for ROI selection
                cut_rois(cv2.cvtColor(
                    IMAGES[CURRENT_IMAGE_INDEX], cv2.COLOR_GRAY2BGR))
        else:
            print(f"Invalid index. Choose a value between 0 and {
                  len(IMAGES) - 1}.")

    except ValueError:
        print("Please enter a valid number.")


# ----------------------------- FUNCTIONALITIES (Part 2) ----------------------------

# Save[.mat] informations


def save_image(roi_image):
    global paciente, ultrassom
    # Nome do arquivo com base no paciente e ultrassom
    file_name = f"ROI_{str(paciente).zfill(2)}_{ultrassom}.jpg"
    file_path = os.path.join(os.getcwd(), file_name)
    image = Image.fromarray(roi_image)
    image.save(file_path, 'JPEG')

    if (ultrassom > 10):
        paciente += 1
        ultrassom = 0
    else:
        ultrassom += 1

# Calc index HI value


def calc_HI(roi_rim, roi_figado, coord_rim, coord_figado):
    
    average_rim = np.mean(roi_rim)
    average_figado = np.mean(roi_figado)

    hi_ratio = average_figado / average_rim

    def show_hi_ratio_window(hi_ratio):
        hi_window = Toplevel(root)  
        hi_window.title("HI Ratio")
        hi_window.geometry("300x200")

        label = Label(hi_window, text=f"HI Ratio: {hi_ratio:.2f}\n"
                      f"Coord. Fígado: {coord_figado}\n"
                      f"Coord. Cortex Renal: {coord_rim}", font=('Arial', 14))
        label.pack(pady=20)
        

    adjusted_roi_figado = roi_figado * hi_ratio

    # Arredonda os valores
    adjusted_roi_figado = np.round(adjusted_roi_figado).astype(
        np.uint8)  # Converte para uint8 após arredondar

    # Certifique-se de que os valores ajustados não excedam 255
    adjusted_roi_figado = np.clip(adjusted_roi_figado, 0, 255)
    
    update_excel(f"ROI_{str(paciente).zfill(2)}_{ultrassom}","Coordenadas Figado", f"{coord_figado[0]}, {coord_figado[1]}")
    update_excel(f"ROI_{str(paciente).zfill(2)}_{ultrassom}", "Coordenadas Cortex Renal", f"{coord_rim[0]}, {coord_rim[1]}")
    update_excel(f"ROI_{str(paciente).zfill(2)}_{ultrassom}", "HI",hi_ratio)
    
    save_image(adjusted_roi_figado)

    show_hi_ratio_window(hi_ratio)
    

# ROI -> Region of Interest


def cut_rois(image):
    image_copy = image.copy()
    click_count = [0]

    roi_rim = None
    roi_figado = None
    coord_rim = None
    coord_figado = None

    # Canvas
    fig, ax = plt.subplots(figsize=(10, 5))
    canvas = FigureCanvasTkAgg(fig, master=image_frame)
    canvas.get_tk_widget().pack(side=TOP, anchor="center",
                                padx=10, pady=10, fill="both", expand=True)
    # Toolbar
    toolbar_frame = Frame(image_frame)
    toolbar_frame.pack(side=TOP, anchor="center", padx=10, pady=10)
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()

    # Exibir a imagem original
    ax.set_title("Clique para selecionar a ROI do Figado")
    ax.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    canvas.draw()

    def click_event(event):
        nonlocal roi_rim, roi_figado, coord_figado, coord_rim

        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)

            # Forçar a ROI a ter 28x28
            if x + 28 <= image.shape[1] and y + 28 <= image.shape[0]:
                roi_cropped = image[y:y + 28, x:x + 28]

                if click_count[0] == 0:
                    # Primeira seleção: Fígado
                    roi_figado = roi_cropped
                    coord_figado = (x - 14, y - 14)
                    display_roi(
                        roi_figado, title="ROI Figado (28x28)", parent=roi_liver_frame)

                    cv2.rectangle(image_copy, (x - 14, y - 14),
                                  (x + 14, y + 14), (0, 255, 0), 2)

                elif click_count[0] == 1:
                    # Segunda seleção: córtex renal
                    roi_rim = roi_cropped
                    coord_rim = (x - 14, y - 14)
                    display_roi(roi_rim, title="ROI Rim (28x28)",
                                parent=roi_kidney_frame)

                    # Desenhar o retângulo na imagem para o rim
                    cv2.rectangle(image_copy, (x - 14, y - 14),
                                  (x + 14, y + 14), (0, 255, 0), 2)

                    # Função para cálculo de HI (depois de ambas as seleções)
                    calc_HI(roi_rim, roi_figado, coord_rim, coord_figado)

                # Atualizar imagem com ROIs desenhadas
                ax.clear()
                ax.set_title("Clique para selecionar a ROI do cortex renal")
                ax.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
                canvas.draw()

                click_count[0] += 1

    # Conectar o evento de clique ao canvas
    canvas.mpl_connect('button_press_event', click_event)


def select_rois():
    global IMAGES, CURRENT_IMAGE_INDEX, image_frame, canvas, IsLoadImage

    IsLoadImage = False
    # Carregar a imagem
    image_path = filedialog.askopenfilename(
        filetypes=[("Imagens", "*.png;*.jpg"), ("MAT files", "*.mat")]
    )

    if image_path:
        if image_path.lower().endswith(('*.png', '*.jpg', '*.jpeg')):
            cut_rois(cv2.imread(image_path))
        elif image_path.lower().endswith('.mat'):
            mat = scipy.io.loadmat(image_path)
            data = mat['data']  # Images are stored in 'data' key

            IMAGES = []

            for entry in data[0]:  # Images are stored in the last element of each entry
                images_array = entry[-1]

                if images_array is not None:
                    for image in images_array:
                        IMAGES.append(image)

            CURRENT_IMAGE_INDEX = 0  # Start at the first image
            cut_rois(cv2.cvtColor(
                IMAGES[CURRENT_IMAGE_INDEX], cv2.COLOR_GRAY2BGR))

def calcular_glcm(roi, distancia):
    angles  = [0, 45, 90, 135, 180, 225, 270, 315]
    glcm = np.zeros((256, 256), dtype=np.float32)

    for angle in angles:
        # Calcular os deslocamentos para o ângulo atual
        if angle == 0:      # Horizontal
            y_offset, x_offset = 0, distancia
        elif angle == 45:   # Diagonal principal
            y_offset, x_offset = distancia, distancia
        elif angle == 90:    # Vertical
            y_offset, x_offset = distancia, 0
        elif angle == 135:  # Diagonal secundária
            y_offset, x_offset = distancia, -distancia
        elif angle == 180:  # Horizontal (oposto)
            y_offset, x_offset = 0, -distancia
        elif angle == 225:  # Diagonal secundária (oposto)
            y_offset, x_offset = -distancia, -distancia
        elif angle == 270:  # Vertical (oposto)
            y_offset, x_offset = -distancia, 0
        elif angle == 315:  # Diagonal principal (oposto)
            y_offset, x_offset = -distancia, distancia
    
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

def calcular_homogeneidade(glcm):
    homogeneidade = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            homogeneidade += glcm[i, j] / (1 + abs(i - j))
    return homogeneidade

def calcular_entropia(glcm):
    entropia = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            if glcm[i, j] > 0:
                entropia -= glcm[i, j] * np.log2(glcm[i, j])
    return entropia

def compute_matriz():
    image_path = filedialog.askopenfilename(
        filetypes=[("Imagens", "*.png;*.jpg"), ("MAT files", "*.mat")]
    )
    if image_path:
        i = [1,2,4,8]
        
        descritores = []

        for j in i:
            glcm = calcular_glcm(cv2.imread(image_path), j)
            homogeneidade = calcular_homogeneidade(glcm)
            entropia =calcular_entropia(glcm)
            descritores.append({
                'distancia': j,
                'homogeneidade': homogeneidade,
                'entropia': entropia,
                'glcm': glcm  # Armazena a GLCM
            })
            update_excel(os.path.splitext(os.path.basename(image_path))[0], f"Entropia i ={j}",f"{entropia}")
            update_excel(os.path.splitext(os.path.basename(image_path))[0], f"Homogeneidade i={j}",f"{entropia}")
                        
    def show_descritores(descritores):
        hi_window = Toplevel(root)  
        hi_window.title("Descritores")
        hi_window.geometry("400x300")

        for desc in descritores:
            result_text = (f"Distância: {desc['distancia']}\n"
                        f"Homogeneidade: {desc['homogeneidade']:.4f}, Entropia: {desc['entropia']:.4f}\n")
            
            label = Label(hi_window, text=result_text, font=('Arial', 10))
            label.pack(pady=5)
        
    show_descritores(descritores)

def tamura():
    image_path = filedialog.askopenfilename(
        filetypes=[("JPG", "*.jpg"), ("MAT files", "*.mat")]
    )

    if image_path:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("Error: Could not load image.")
            return

        coarseness = tamura_coarseness(img)
        contrast = tamura_contrast(img)
        directionality = tamura_directionality(img)
        line_likeness = tamura_line_likeness(img)
        regularity = tamura_regularity(img)
        roughness = tamura_roughness(img)

        print("Tamura Features:")
        print(f"Coarseness: {coarseness}")
        print(f"Contrast: {contrast}")
        print(f"Directionality: {directionality}")
        print(f"Line-Likeness: {line_likeness}")
        print(f"Regularity: {regularity}")
        print(f"Roughness: {roughness}")

    

def tamura_coarseness(img, k_max=5):
    # k_max (int): max scale (2^k)

    h, w = img.shape
    # create array to store the max difference for each pixel
    S_best = np.zeros((h, w))

    # iterate over scales 2^k
    for k in range(1, k_max + 1):
        window_size = 2 ** k
        
        # local averages
        avg_kernel = uniform_filter(img, size=window_size, mode='reflect')
        
        # difference between neighboring regions in x and y
        diff_x = np.abs(avg_kernel[:, :-window_size] - avg_kernel[:, window_size:])
        diff_y = np.abs(avg_kernel[:-window_size, :] - avg_kernel[window_size:, :])

        S_best[:-window_size, :-window_size] = np.maximum(
            S_best[:-window_size, :-window_size],
            np.maximum(diff_x, diff_y)
        )

    coarseness = np.mean(S_best)
    return coarseness
    
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
    
    # calc sum inverse of standard deviations
    regularity = np.sum(1 / (std_devs + 1e-6))

    return regularity

def tamura_roughness(img):
    coarseness = tamura_coarseness(img)
    contrast = tamura_contrast(img)

    roughness = coarseness * contrast
    
    return roughness

def on_closing():
    root.quit()
    root.destroy()

# ------------------------------------ GUI ------------------------------------------


customtkinter.set_appearance_mode("light")  # system, light, dark
customtkinter.set_default_color_theme("green")  # blue, dark-blue, green

root = customtkinter.CTk()

# Default Opening Settings
root.geometry("1760x960")
root.title('Sistema Auxiliar de Diagnóstico da NAFLD')
label = Label(
    root, text='Sistema Auxiliar de Diagnóstico da NAFLD', font=('Arial', 14))
label.pack(pady=10)
root.protocol("WM_DELETE_WINDOW", on_closing)

# Buttons Grid ---------------------------------------------------------------- Start

button_frame = Frame(root)
button_frame.pack(pady=10)

load_images_button = Button(
    button_frame, text='Carregar Imagens', command=load_file)
load_images_button.grid(row=0, column=0, padx=5)  # Load files

load_roi_button = Button(
    button_frame, text='Vizualizar ROIs', command=lambda: load_file(True))
load_roi_button.grid(row=0, column=1, padx=5)  # Load files (ROI)

roi_button = Button(button_frame, text='Selecionar ROI', command=select_rois)
roi_button.grid(row=0, column=2, padx=5)  # Select ROI

matriz_button = Button(button_frame, text='Computar Matriz de co-ocorrência', command=compute_matriz)
matriz_button.grid(row=0, column=3, padx=5)

carac_button = Button(button_frame, text='Caracterizar ROI')
carac_button.grid(row=0, column=4, padx=5)

classificar_button = Button(button_frame, text='Classificar Imagem')
classificar_button.grid(row=0, column=5, padx=5)

classificar_button = Button(button_frame, text='Descritores Tamura', command=tamura)
classificar_button.grid(row=0, column=5, padx=5)

previous_image_button = Button(
    button_frame, text='Imagem Anterior', command=previous_image)
previous_image_button.grid(row=0, column=6, padx=5)  # Previous Image Button

next_image_button = Button(
    button_frame, text='Próxima Imagem', command=next_image)
next_image_button.grid(row=0, column=7, padx=5)  # Next Image Button

reset_pagination_button = Button(
    button_frame, text='Limpar Fluxo', command=reset_pagination)
reset_pagination_button.grid(
    row=0, column=8, padx=5)  # Reset Pagination Button

index_entry = Entry(button_frame, width=5)
index_entry.grid(row=0, column=9, padx=5)  # User Pagination Input

go_to_image_button = Button(
    button_frame, text='Ir para Imagem', command=go_to_image)
go_to_image_button.grid(row=0, column=10, padx=5)  # Go to Index

# Buttons Grid ------------------------------------------------------------------ End

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



