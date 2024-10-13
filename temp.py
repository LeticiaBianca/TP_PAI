# Letícia Bianca Oliveira - 776782
# Raick Miranda Rodrigues Santos - 781755

# ------------------------------------ LIBRARIES & MODULES ------------------------------------

# GUI & OS demands
import os
import customtkinter
from tkinter import *
from tkinter import filedialog
from scipy.io import savemat

# Data Handling
import cv2
import scipy.io
import numpy as np

# Image Handling
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ------------------------------------ CODE ------------------------------------

# Global Variables -> Pagination
CURRENT_IMAGE_INDEX = 0
IMAGES = None

# Plot images along with histogram and ROI information


def display_image_and_histogram(image, isROI=False, title="Imagem", parent=None):
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

    # If we're dealing with a ROI, we want to plot some characteristics along with the image
    if (isROI):
        ax_hist.axvline(average_brightness, color='red',
                        linestyle='dashed', linewidth=1)
        ax_hist.text(average_brightness + 5, max(n) * 0.9,
                     f'Avg: {average_brightness:.2f}', color='red')

    plt.tight_layout()

    # Embed image plots
    if parent is not None:
        for widget in parent.winfo_children():
            widget.destroy()  # Remove previous plot, if exists

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, anchor="center", padx=10, pady=10)

# Loading image files from computer supports the following file types: [.png/.jpg/.jpeg/.mat]


def load_file(isROI=False):
    global IMAGES, CURRENT_IMAGE_INDEX

    file_path = filedialog.askopenfilename(
        filetypes=[("Imagens e MAT", ".png;.jpg;*.mat")]
    )

    if file_path:
        # Handling [.png/.jpg/.jpeg] files
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            display_image_and_histogram(
                image, isROI, title="Imagem Local", parent=image_frame)

        # Handling [.mat] files
        elif file_path.lower().endswith('.mat'):
            mat = scipy.io.loadmat(file_path)
            data = mat['data']  # images are stored in 'data' key
            IMAGES = data[0, 0][-1]
            CURRENT_IMAGE_INDEX = 0
            display_image_and_histogram(IMAGES[CURRENT_IMAGE_INDEX], isROI,
                                        title=f"Imagem {
                                            CURRENT_IMAGE_INDEX + 1}",
                                        parent=image_frame)

# Skip to the next image in the current pagination


def next_image():
    global CURRENT_IMAGE_INDEX

    if IMAGES is not None and CURRENT_IMAGE_INDEX < IMAGES.shape[0] - 1:
        CURRENT_IMAGE_INDEX += 1
        display_image_and_histogram(IMAGES[CURRENT_IMAGE_INDEX],
                                    title=f"Imagem {CURRENT_IMAGE_INDEX + 1}",
                                    parent=image_frame)
    else:
        print("Não há mais imagens para mostrar.")

# Returns to the previous image in the current pagination


def previous_image():
    global CURRENT_IMAGE_INDEX

    if IMAGES is not None and CURRENT_IMAGE_INDEX > 0:
        CURRENT_IMAGE_INDEX -= 1
        display_image_and_histogram(IMAGES[CURRENT_IMAGE_INDEX],
                                    title=f"Imagem {CURRENT_IMAGE_INDEX + 1}",
                                    parent=image_frame)
    else:
        print("Não há imagens anteriores para mostrar.")

# ROI Calculations


def calc_HI(roi_rim, roi_figado):
    paciente = 0
    ultrassom = 0

    average_rim = np.mean(roi_rim)
    average_figado = np.mean(roi_figado)

    hi_ratio = average_figado / average_rim

    def show_hi_ratio_window(hi_ratio):
        # Cria uma nova janela
        hi_window = Toplevel(root)  # Usa Toplevel para criar uma nova janela
        hi_window.title("HI Ratio")
        hi_window.geometry("300x200")

        # Adiciona um label com o hi_ratio
        label = Label(hi_window, text=f"HI Ratio: {
                      hi_ratio:.2f}", font=('Arial', 14))
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

    # Nome da ROI baseado na posição
    roi_file_name = f"roi_{paciente}_{ultrassom}.mat"
    roi_save_path = os.path.join(os.getcwd() + "/", roi_file_name)

    # Garantir que a ROI é convertida para float para ser salva no formato .mat
    roi_cropped_float = adjusted_roi_figado.astype(np.float32)

    # Salvando a ROI em um dicionário
    savemat(roi_save_path, {'roi': roi_cropped_float})

    if (ultrassom > 10):
        paciente += 1
        ultrassom = 0
    else:
        ultrassom += 1

    show_hi_ratio_window(hi_ratio)

# ROI -> Region of Interest


def select_rois():
    global image_frame, canvas

    # Carregar a imagem
    image_path = filedialog.askopenfilename(
        filetypes=[("Imagens", ".png;.jpg;")])

    if image_path:
        image = cv2.imread(image_path)
        image_copy = image.copy()
        click_count = [0]

        roi_rim = None
        roi_figado = None

        fig, ax = plt.subplots(figsize=(5, 5))
        canvas = FigureCanvasTkAgg(fig, master=image_frame)
        canvas.get_tk_widget().pack(pady=10, padx=10)

        # Exibir a imagem original
        ax.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        canvas.draw()

        def show_message_on_image(ax, message):
            ax.text(0.5, -0.1, message, fontsize=12, ha='center',
                    transform=ax.transAxes, color='red')
            canvas.draw()

        def click_event(event):
            nonlocal roi_rim, roi_figado

            if event.inaxes == ax:  # Verificar se o clique foi dentro da imagem
                x, y = int(event.xdata), int(event.ydata)

                # Forçar a ROI a ter 28x28 pixels a partir do ponto clicado
                if x + 28 <= image.shape[1] and y + 28 <= image.shape[0]:
                    roi_cropped = image[y:y + 28, x:x + 28]

                    # Mostrar a ROI selecionada no frame da interface
                    display_image_and_histogram(
                        roi_cropped, True, title="ROI Recortada (28x28)", parent=image_frame)

                    # Desenhar o retângulo da ROI na imagem
                    cv2.rectangle(image_copy, (x, y),
                                  (x + 28, y + 28), (0, 255, 0), 2)
                    ax.clear()  # Limpar o eixo para redesenhar a imagem com a ROI
                    ax.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))

                    click_count[0] += 1

                    # Se já foram feitos dois cliques (rim e fígado), parar o processo
                    if click_count[0] >= 2:
                        roi_figado = roi_cropped
                        calc_HI(roi_rim, roi_figado)
                    else:
                        roi_rim = roi_cropped
                        # Atualiza a mensagem para selecionar o fígado
                        show_message_on_image(
                            ax, "Clique para selecionar a ROI do Figado")

                    canvas.draw()

        # Mensagem inicial
        show_message_on_image(ax, "Clique para selecionar a ROI do Rim")

        # Conectar o evento de clique ao canvas
        canvas.mpl_connect('button_press_event', click_event)

# ------------------------------------ GUI ----------------------------------------


customtkinter.set_appearance_mode("light")  # system (default), light, dark
customtkinter.set_default_color_theme(
    "blue")  # blue (default), dark-blue, green

root = customtkinter.CTk()

# Default Opening Settings
root.geometry("1100x600")
root.title('Sistema Auxiliar de Diagnóstico da NAFLD')

# Header
label = Label(
    root, text='Sistema Auxiliar de Diagnóstico da NAFLD', font=('Arial', 14))
label.pack(pady=10)

# Buttons Grid ------------------- Start

button_frame = Frame(root)
button_frame.pack(pady=10)

load_images_button = Button(
    button_frame, text='Vizualizar Imagens', command=load_file)
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

# Buttons Grid ------------------- End

image_frame = Frame(root, width=500, height=300,)
image_frame.pack(pady=10, padx=10)

# GUI Loop
root.mainloop()
