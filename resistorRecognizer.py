# De início vamos testar tudo na imagem teste pq ela é muito boa

# Passo a passo dado pelo gemini
# Passo 1: Carregar e Limpar (Pré-processamento)
    #Filtro Bilateral. Para "embaçar" o fundo (listras do resistor afiadas)
# Passo 2: Isolar o Resistor (Segmentação)
    #Transformamos a imagem para tons de cinza e usamos um detector de bordas (como o algoritmo Canny) 
    # para achar os contornos. 
    # A gente manda o código procurar o maior retângulo da imagem e 
    # recorta (cropa) só essa parte.
# Passo 3: Mudar para HSV e Filtrar as Cores (A Mágica)
    #Convertamos a imagem cortada para o espaço HSV. Depois, criamos "máscaras" (filtros)
    # para as cores principais
# Passo 4: Ler a Posição e Calcular
    # O algoritmo varre a imagem da esquerda para a direita anotando qual cor aparece primeiro, qual 
    # aparece depois, e o multiplicador no final.
    # fica limitado mas é o jeito mais fácil.

import cv2
import numpy as np

image_path = 'testresistor.jpg'
original_image = cv2.imread(image_path)

if original_image is None:
    print("Erro ao carregar a imagem. Verifique o caminho.")
else:
    # Redimensiona a imagem para uma largura de 800 pixels, mantendo a proporção
    # gemini recomenda pq se a imagem for gigante, trava, então é uma garantia
    proportion = 800.0/original_image.shape[1]
    new_size = (800, int(original_image.shape[0] * proportion))
    resized_image = cv2.resize(original_image, new_size) # imagem redimensionada

    # Passo 1: Filtro Bilateral para suavizar a imagem sem perder as bordas
    # é usado em https://github.com/SupreethRao99/CVResist.git
    filtered_image = cv2.bilateralFilter(resized_image, 15, 80, 80)

    # testando o resultado do filtro bilateral
    cv2.imshow('Original Image', resized_image)
    cv2.imshow('Filtered Image', filtered_image)

    # Trava a tela até apertar qualquer tecla (se não colocar isso, a janela abre e fecha na hora)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Passo 2: Isolar o Resistor (Segmentação)
