# Letícia Bianca Oliveira - 776782
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

# Image Save
import os
from scipy.io import savemat

# Global variables
CURRENT_IMAGE_INDEX = 0
IMAGES = None

# Display image along with histogram
def display_image_and_histogram(image, isROI = False, title="Imagem"):

    average_brightness = np.mean(image)

    # Configurando a exibição da imagem
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Exibindo a imagem
    ax_img.imshow(image, cmap='gray')
    ax_img.axis('off')
    ax_img.set_title(title)
    
    # Exibindo o histograma
    n, bins, patches = ax_hist.hist(image.ravel(), bins=256, color='gray', alpha=0.7)
    ax_hist.set_xlim([0, 255])
    ax_hist.set_ylim([0, max(n) * 1.1])
    ax_hist.set_title('Histograma')
    ax_hist.set_xlabel('Intensidade de Pixel')
    ax_hist.set_ylabel('Frequência')

    if(isROI):
        ax_hist.axvline(average_brightness, color='red', linestyle='dashed', linewidth=1)
        ax_hist.text(average_brightness + 5, max(n) * 0.9, f'Avg: {average_brightness:.2f}', color='red')

    plt.tight_layout()
    plt.show()


def load_file(isROI = False):
    global IMAGES, CURRENT_IMAGE_INDEX

    # Abrir o diálogo para escolher arquivos, permitindo tanto imagens quanto arquivos MAT
    file_path = filedialog.askopenfilename(
        filetypes=[("Imagens e MAT", ".png;.jpg;*.mat")])

    if file_path:
        # Verificar se o arquivo é uma imagem (PNG/JPG)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(file_path)
            display_image_and_histogram(image, isROI,  title="Imagem Carregada")

        # Verificar se o arquivo é um arquivo MAT
        elif file_path.lower().endswith('.mat'):
            mat = scipy.io.loadmat(file_path)
            data = mat['data']
            IMAGES = data[0, 0][-1]
            CURRENT_IMAGE_INDEX = 0
            display_image_and_histogram(IMAGES[CURRENT_IMAGE_INDEX], isROI, title=f"Imagem {
                                        CURRENT_IMAGE_INDEX + 1}")

def next_image():
    global CURRENT_IMAGE_INDEX
    if IMAGES is not None and CURRENT_IMAGE_INDEX < IMAGES.shape[0] - 1:
        CURRENT_IMAGE_INDEX += 1
        display_image_and_histogram(IMAGES[CURRENT_IMAGE_INDEX], title=f"Imagem {
                                    CURRENT_IMAGE_INDEX + 1}")
    else:
        print("Não há mais imagens para mostrar.")

#Calculos ROI
def calc_HI(roi_rim, roi_figado, coord_rim, coord_figado):
    paciente = 0
    ultrassom = 0

    average_rim = np.mean(roi_rim)
    average_figado= np.mean(roi_figado)

    hi_ratio = average_figado / average_rim

    def show_hi_ratio_window():
        # Cria uma nova janela
        hi_window = Toplevel(root)  # Usa Toplevel para criar uma nova janela
        hi_window.title("HI Ratio")
        hi_window.geometry("300x200")

        label = Label(hi_window, text=f"HI Ratio: {hi_ratio:.2f}\n"
                                      f"Coord. Rim: {coord_rim}\n"
                                      f"Coord. Fígado: {coord_figado}", font=('Arial', 14))
        label.pack(pady=20)

        # Botão para fechar a janela
        close_button = Button(hi_window, text="Fechar", command=hi_window.destroy)
        close_button.pack(pady=10)

    adjusted_roi_figado = roi_figado * hi_ratio
    
    # Arredonda os valores
    adjusted_roi_figado = np.round(adjusted_roi_figado).astype(np.uint8)  # Converte para uint8 após arredondar

    # Certifique-se de que os valores ajustados não excedam 255
    adjusted_roi_figado = np.clip(adjusted_roi_figado, 0, 255)

    roi_file_name = f"roi_{paciente}_{ultrassom}.mat"  # Nome da ROI baseado na posição
    roi_save_path = os.path.join(os.getcwd() + "/", roi_file_name)

    # Garantir que a ROI é convertida para float para ser salva no formato .mat
    roi_cropped_float = adjusted_roi_figado.astype(np.float32)
                    
    # Salvando a ROI em um dicionário
    savemat(roi_save_path, {'roi': roi_cropped_float})

    if(ultrassom > 10):
        paciente+=1
        ultrassom = 0
    else:   
        ultrassom+=1

    show_hi_ratio_window()

# ROI -> Region of Interest


def select_rois():
    image_path = filedialog.askopenfilename(
        filetypes=[("Imagens", ".png;.jpg")])

    if image_path:
        image = cv2.imread(image_path)

        # Copiar a imagem original para exibir o corte das ROIs
        image_copy = image.copy()
        click_count = [0]

        roi_rim = None  
        roi_figado = None  
        coord_rim = None
        coord_figado = None

        def show_message_on_image(image, message):
            font = cv2.FONT_HERSHEY_PLAIN
            font_scale = 0.75
            color = (0, 0, 255)
            thickness = 1
            position = (10, 30) 

            background_color = (255, 255, 255)
            text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]

            top_left = (position[0] - 5, position[1] - 20)  
            bottom_right = (position[0] + text_size[0] + 5, position[1] + 5) 
            cv2.rectangle(image, top_left, bottom_right, background_color, -1)

            # Desenha a nova mensagem sobre o retângulo
            cv2.putText(image, message, position, font, font_scale, color, thickness, cv2.LINE_AA)

       
        def click_event(event, x, y, flags, param):
            nonlocal roi_rim, roi_figado, coord_rim, coord_figado
            if event == cv2.EVENT_LBUTTONDOWN:               

                # Forçar a ROI a ter 28x28 pixels a partir do ponto clicado
                if x + 28 <= image.shape[1] and y + 28 <= image.shape[0]:
                    roi_cropped = image[y:y+28, x:x+28]

                    # Mostrar a ROI selecionada
                    display_image_and_histogram(
                        roi_cropped, True, title="ROI Recortada (28x28)")

                    # Desenhar a ROI na imagem original para visualização
                    cv2.rectangle(image_copy, (x - 14, y - 14), (x + 14, y + 14), (0, 255, 0), 2)
                    cv2.imshow("Imagem", image_copy)

                    click_count[0] += 1

                    # Se já foram feitos dois cliques (rim e fígado), parar o processo
                    if click_count[0] >= 2:
                        roi_figado = roi_cropped
                        coord_figado = (x - 14, y - 14)
                        calc_HI(roi_rim,roi_figado,coord_rim, coord_figado)
                        cv2.destroyAllWindows()
                    else:
                        roi_rim = roi_cropped
                        
                        # Atualiza a mensagem na interface para o próximo órgão
                        message = "Clique para selecionar a ROI do Figado"
                        image_with_message = image_copy.copy()
                        show_message_on_image(image_with_message, message)
                        cv2.imshow("Imagem", image_with_message)


        show_message_on_image(image_copy, "Clique para selecionar a ROI do Rim")

        # Exibir a janela de imagem e configurar o evento de clique
        cv2.imshow("Imagem", image_copy)
        cv2.setMouseCallback("Imagem", click_event)

        # Aguardar o fechamento da janela
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ------------------------------------ GUI ----------------------------------------


# Init
# Modes: system (default), light, dark
customtkinter.set_appearance_mode("light")
# Themes: blue (default), dark-blue, green
customtkinter.set_default_color_theme("blue")
root = customtkinter.CTk()

# Config
root.geometry("600x400")
root.title('Sistema Auxiliar de Diagnóstico da NAFLD')

# Header
label = Label(
    root, text='Sistema Auxiliar de Diagnóstico da NAFLD', font=('Arial', 14))
label.pack(pady=10)

# Load images (png / jpg) from computer [ BUTTON ]
load_images_button = Button(root, text='Vizualizar Imagens', command=load_file)
load_images_button.pack(pady=10)

# Load images (png / jpg) from computer [ BUTTON ]
load_roi_button = Button(root, text='Vizualizar ROIs', command=lambda: load_file(True))
load_roi_button.pack(pady=10)

# Select ROI [ BUTTON ]
roi_button = Button(root, text='Selecionar ROI', command=select_rois)
roi_button.pack(pady=10)

# Computar Matriz de co-ocorrência [ BUTTON ]
matriz_button = Button(root, text='Computar Matriz de co-ocorrência')
matriz_button.pack(pady=10)

# Computar Matriz de co-ocorrência [ BUTTON ]
carac_button = Button(root, text='Caracterizar ROI')
carac_button.pack(pady=10)

# Computar Matriz de co-ocorrência [ BUTTON ]
classificar_button = Button(root, text='Classificar Imagem')
classificar_button.pack(pady=10)

# Next image [ BUTTON ]
next_image_button = Button(root, text='Próxima Imagem', command=next_image)
next_image_button.pack(pady=10)

# GUI Loop
root.mainloop()