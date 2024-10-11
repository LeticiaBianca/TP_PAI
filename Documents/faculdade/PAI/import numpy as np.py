import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from matplotlib.widgets import Slider

# Usando o arquivo já carregado
mat = scipy.io.loadmat('dataset_liver_bmodes_steatosis_assessment_IJCARS.mat')

data = mat['data']

images = data[0, 0][-1]

current_image_index = 0

def show_image(index):
    # Função para atualizar a exibição da imagem com zoom
    def update(val):
        zoom_level = slider.val
        ax_img.imshow(images[index], cmap='gray', extent=[0, zoom_level, 0, zoom_level])
        fig.canvas.draw_idle()

    # Configurando a exibição da imagem
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Exibindo a imagem
    ax_img.imshow(images[index], cmap='gray')
    ax_img.axis('off')
    ax_img.set_title(f'Imagem {index + 1}')
    
    # Exibindo o histograma
    ax_hist.hist(images[index].ravel(), bins=256, color='gray', alpha=0.7)
    ax_hist.set_xlim([0, 255])
    ax_hist.set_ylim([0, 5000])
    ax_hist.set_title('Histograma')
    ax_hist.set_xlabel('Intensidade de Pixel')
    ax_hist.set_ylabel('Frequência')

    # Adicionando o slider de zoom
    ax_slider = plt.axes([0.25, 0.01, 0.5, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Zoom', 1.0, 10.0, valinit=1.0, valstep=0.1)
    slider.on_changed(update)
    
    plt.tight_layout()
    plt.show()

def on_next_button_clicked(b):
    global current_image_index
    if current_image_index < images.shape[0] - 1:
        current_image_index += 1
        clear_output(wait=True)
        show_image(current_image_index)
        display(next_button, exit_button)


    else:
        print("Não há mais imagens para mostrar.")

def on_exit_button_clicked(b):
    clear_output(wait=True)
    print("Saindo da visualização de imagens.")

show_image(current_image_index)

next_button = widgets.Button(description="Próxima Imagem")
exit_button = widgets.Button(description="Sair")

next_button.on_click(on_next_button_clicked)
exit_button.on_click(on_exit_button_clicked)

display(next_button, exit_button)