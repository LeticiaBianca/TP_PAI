# Letícia Bianca Oliveira - 776782
# Raick Miranda Rodrigues Santos - 781755
# NT = (776782 + 781755) mod 4 = 1

# ------------------------------------ LIBRARIES & MODULES --------------------------

# GUI & OS demands
import os
import customtkinter
from tkinter import *
from tkinter import filedialog

# Data Handling
import cv2
import scipy.io
from scipy.io import savemat
import numpy as np

# Image Plots
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ----------------------------- GLOBAL VARIABLES ------------------------------------

CURRENT_IMAGE_INDEX = 0  # Pagination
IMAGES = None

IsLoadImage = True
paciente = 0
ultrassom = 0

# ----------------------------- FUNCTIONALITIES (Part 1) ----------------------------

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
        filetypes=[("Imagens e MAT", ".png;.jpg;*.mat")]
    )

    if file_path:  # Handling [.png/.jpg/.jpeg] files
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if isROI:  # Is it an ROI?
                display_roi(image, title="Imagem Local", parent=roi_frame)
            else:
                display_image(
                    image, title="Imagem Local", parent=image_frame)

        # Handling [.mat] files
        elif file_path.lower().endswith('.mat'):
            mat = scipy.io.loadmat(file_path)
            print(mat.keys())
            data = mat['data']  # images are stored in 'data' key
            IMAGES = data[0, 0][-1]
            CURRENT_IMAGE_INDEX = 0

            if isROI:  # Is it an ROI?
                display_roi(IMAGES[CURRENT_IMAGE_INDEX],
                            title=f"Imagem {CURRENT_IMAGE_INDEX + 1}",
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

    if IMAGES is not None and CURRENT_IMAGE_INDEX < IMAGES.shape[0] - 1:
        CURRENT_IMAGE_INDEX += 1
        destroy_plots()

        if IsLoadImage:  # Is it a regular image? Then display histogram
            display_image_and_histogram(IMAGES[CURRENT_IMAGE_INDEX],
                                        title=f"Imagem {
                                            CURRENT_IMAGE_INDEX + 1}",
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
                                            CURRENT_IMAGE_INDEX + 1}",
                                        parent=image_frame)
        else:  # Is it for selecting ROIs? Then display the image only in its fullsize
            cut_rois(cv2.cvtColor(
                IMAGES[CURRENT_IMAGE_INDEX], cv2.COLOR_GRAY2BGR))
    else:
        print("There are no more images to show.")


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
        if IMAGES is not None and 0 <= index < IMAGES.shape[0]:
            CURRENT_IMAGE_INDEX = index
            destroy_plots()

            # Display image and histogram or allow ROI selection
            if IsLoadImage:  # Display image and histogram
                display_image_and_histogram(IMAGES[CURRENT_IMAGE_INDEX],
                                            title=f"Imagem {
                                                CURRENT_IMAGE_INDEX + 1}",
                                            parent=image_frame)
            else:  # Display full image for ROI selection
                cut_rois(cv2.cvtColor(
                    IMAGES[CURRENT_IMAGE_INDEX], cv2.COLOR_GRAY2BGR))
        else:
            print(f"Invalid index. Choose a value between 0 and {
                  IMAGES.shape[0] - 1}.")

    except ValueError:
        print("Please enter a valid number.")


# ----------------------------- FUNCTIONALITIES (Part 2) ----------------------------

# Save[.mat] informations


def save_custom_mat(paciente, ultrassom, roi_image):
    file_name = f"roi_{paciente}_{ultrassom}.mat"
    file_path = os.path.join(os.getcwd(), file_name)

    roi_cropped_float = roi_image.astype(np.float32)
    data_structure = np.array([[roi_cropped_float]])
    savemat(file_path, {'data': data_structure})

    if (ultrassom > 10):
        paciente += 1
        ultrassom = 0
    else:
        ultrassom += 1

# Calc index HI value


def calc_HI(roi_rim, roi_figado, coord_rim, coord_figado):
    global paciente, ultrassom

    average_rim = np.mean(roi_rim)
    average_figado = np.mean(roi_figado)

    hi_ratio = average_figado / average_rim

    def show_hi_ratio_window(hi_ratio):
        # Cria uma nova janela
        hi_window = Toplevel(root)  # Usa Toplevel para criar uma nova janela
        hi_window.title("HI Ratio")
        hi_window.geometry("300x200")

        # Adiciona um label com o hi_ratio
        label = Label(hi_window, text=f"HI Ratio: {hi_ratio:.2f}\n"
                      f"Coord. Fígado: {coord_figado}\n"
                      f"Coord. Cortex Renal: {coord_rim}", font=('Arial', 14))
        label.pack(pady=20)

        # Botão para fechar a janela
        close_button = Button(hi_window, text="Fechar",
                              command=hi_window.destroy)
        close_button.pack(pady=10)

    adjusted_roi_figado = roi_figado * hi_ratio

    # Arredonda os valores
    adjusted_roi_figado = np.round(adjusted_roi_figado).astype(
        np.uint8)  # Converte para uint8 após arredondar

    # Certifique-se de que os valores ajustados não excedam 255
    adjusted_roi_figado = np.clip(adjusted_roi_figado, 0, 255)

    save_custom_mat(paciente, ultrassom, adjusted_roi_figado)

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

        if event.inaxes == ax:  # Verificar se o clique foi dentro da imagem
            x, y = int(event.xdata), int(event.ydata)

            # Forçar a ROI a ter 28x28 pixels a partir do ponto clicado
            if x + 28 <= image.shape[1] and y + 28 <= image.shape[0]:
                roi_cropped = image[y:y + 28, x:x + 28]

                if click_count[0] == 0:
                    # Primeira seleção: Fígado
                    roi_figado = roi_cropped
                    coord_figado = (x - 14, y - 14)
                    display_roi(
                        roi_figado, title="ROI Figado (28x28)", parent=roi_liver_frame)

                    # Desenhar o retângulo na imagem para o fígado
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
        filetypes=[("Imagens", ".png;.jpg;*.mat")]
    )

    if image_path:
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            cut_rois(cv2.imread(image_path))
        elif image_path.lower().endswith('.mat'):
            mat = scipy.io.loadmat(image_path)
            data = mat['data']  # imagens são armazenadas na chave 'data'
            IMAGES = data[0, 0][-1]
            CURRENT_IMAGE_INDEX = 0
            cut_rois(cv2.cvtColor(
                IMAGES[CURRENT_IMAGE_INDEX], cv2.COLOR_GRAY2BGR))

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

matriz_button = Button(button_frame, text='Computar Matriz de co-ocorrência')
matriz_button.grid(row=0, column=3, padx=5)

carac_button = Button(button_frame, text='Caracterizar ROI')
carac_button.grid(row=0, column=4, padx=5)

classificar_button = Button(button_frame, text='Classificar Imagem')
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
