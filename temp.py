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

# Plot images along with histogram


def display_image_and_histogram(image, title="Imagem", parent=None, isROI=False):
    average_brightness = np.mean(image)
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(10, 5))

    # Configurações da Imagem
    ax_img.imshow(image, cmap='gray')
    ax_img.axis('off')
    ax_img.set_title(title)

    # Configurações do Histograma
    n, bins, patches = ax_hist.hist(
        image.ravel(), bins=256, color='gray', alpha=0.7)
    ax_hist.set_xlim([0, 255])
    ax_hist.set_ylim([0, max(n) * 1.1])
    ax_hist.set_title('Histograma')
    ax_hist.set_xlabel('Intensidade de Pixel')
    ax_hist.set_ylabel('Frequência')

    # Se for uma ROI, exibir a linha da média
    if isROI:
        ax_hist.axvline(average_brightness, color='red',
                        linestyle='dashed', linewidth=1)
        ax_hist.text(average_brightness + 5, max(n) * 0.9,
                     f'Avg: {average_brightness:.2f}', color='red')

    plt.tight_layout()

    # Incorporar o gráfico na interface
    if parent is not None:
        for widget in parent.winfo_children():
            widget.destroy()  # Remove plot anterior, se existir

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, anchor="center",
                                    padx=10, pady=10, expand=False)


def display_image(image, title="Imagem", parent=None):
    # Chama a função genérica sem mostrar a linha de média
    display_image_and_histogram(
        image, title, parent, isROI=False)


def display_roi(roi, title="ROI", parent=None):
    # Chama a função genérica com a linha de média ativada
    display_image_and_histogram(roi, title, parent, isROI=True)


def load_file(isROI=False):
    global IMAGES, CURRENT_IMAGE_INDEX

    file_path = filedialog.askopenfilename(
        filetypes=[("Imagens e MAT", ".png;.jpg;*.mat")]
    )

    if file_path:
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            # Verifica se é uma ROI para chamar a função correta
            if isROI:
                display_roi(image, title="Imagem Local", parent=roi_frame)
            else:
                display_image(
                    image, title="Imagem Local", parent=image_frame)

        elif file_path.lower().endswith('.mat'):
            mat = scipy.io.loadmat(file_path)
            data = mat['data']  # imagens são armazenadas na chave 'data'
            IMAGES = data[0, 0][-1]
            CURRENT_IMAGE_INDEX = 0
            # Verifica se é uma ROI para chamar a função correta
            if isROI:
                display_roi(IMAGES[CURRENT_IMAGE_INDEX],
                            title=f"Imagem {CURRENT_IMAGE_INDEX + 1}",
                            parent=roi_frame)
            else:
                display_image(IMAGES[CURRENT_IMAGE_INDEX],
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


def calc_HI(roi_rim, roi_figado, coord_rim, coord_figado):
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
        filetypes=[("Imagens", ".png;.jpg;")]
    )

    if image_path:
        image = cv2.imread(image_path)
        image_copy = image.copy()
        click_count = [0]

        roi_rim = None
        roi_figado = None
        coord_rim = None
        coord_figado = None

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
                        cv2.rectangle(image_copy, (x - 14, y - 14), (x + 14, y + 14), (0, 255, 0), 2)

                        show_message_on_image(ax, "Clique para selecionar a ROI do cortex renal")

                    elif click_count[0] == 1:
                        # Segunda seleção: córtex renal
                        roi_rim = roi_cropped
                        coord_rim = (x - 14, y - 14)
                        display_roi(roi_rim, title="ROI Rim (28x28)",
                                    parent=roi_kidney_frame)

                        # Desenhar o retângulo na imagem para o rim
                        cv2.rectangle(image_copy, (x - 14, y - 14), (x + 14, y + 14), (0, 255, 0), 2)
                        

                        # Função para cálculo de HI (depois de ambas as seleções)
                        calc_HI(roi_rim, roi_figado, coord_rim, coord_figado)

                    # Atualizar imagem com ROIs desenhadas
                    ax.clear()
                    ax.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
                    canvas.draw()

                    click_count[0] += 1

        # Mensagem inicial
        show_message_on_image(ax, "Clique para selecionar a ROI do Figado")

        # Conectar o evento de clique ao canvas
        canvas.mpl_connect('button_press_event', click_event)

# ------------------------------------ GUI ----------------------------------------


customtkinter.set_appearance_mode("light")  # system (default), light, dark
customtkinter.set_default_color_theme(
    "blue")  # blue (default), dark-blue, green

root = customtkinter.CTk()

# Default Opening Settings
root.geometry("1760x960")
root.title('Sistema Auxiliar de Diagnóstico da NAFLD')

# Header
label = Label(
    root, text='Sistema Auxiliar de Diagnóstico da NAFLD', font=('Arial', 14))
label.pack(pady=10)

# Buttons Grid ------------------- Start

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

# Buttons Grid ------------------- End

image_frame = Frame(root, width=600, height=360)
image_frame.pack(pady=10, padx=10)
image_frame.pack_propagate(False)

# ROIs Grid ------------------- Start

roi_frame = Frame(root, width=1000, height=400)
roi_frame.pack(pady=10, padx=10)

roi_liver_frame = Frame(roi_frame, width=600, height=360)
roi_liver_frame.grid(row=0, column=0)
roi_liver_frame.pack_propagate(False)

roi_kidney_frame = Frame(roi_frame, width=600, height=360)
roi_kidney_frame.grid(row=0, column=1)
roi_kidney_frame.pack_propagate(False)

# ROIs Grid ------------------- End

# GUI Loop
root.mainloop()
