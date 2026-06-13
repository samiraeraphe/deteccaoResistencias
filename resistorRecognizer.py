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
    ''''
    # Não deu certo usando Canny, mas deixo o código aqui pra referência futura

    gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gray_image, 50, 150)

    # Encontrar os contornos
    contours, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Encontrar o maior contorno (assumindo que seja o resistor)
        largest_contour = max(contours, key=cv2.contourArea)

        # Obter as coordenadas do retângulo delimitador
        x, y, largura, altura = cv2.boundingRect(largest_contour)

        # Recortar a imagem para isolar o resistor
        cropped_image = filtered_image[y:y+altura, x:x+largura]

        # Mostrar a imagem recortada
        cv2.imshow('Bordas', bordes)
        cv2.imshow('Cropped Resistor', cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Nenhum contorno encontrado.")
    '''
    # Não deu certo usando Canny
    # Mudei para o método do repositório https://github.com/SupreethRao99/CVResist.git:
    imagem_cinza = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    # Usando os mesmos parâmetrosdo repositório:
    # - 79 é o tamanho do "bloco" que ele analisa por vez (precisa ser número ímpar).
    # - 2 é um ajuste fino (subtraído da média do bloco).
    mascara_fundo = cv2.adaptiveThreshold(
        imagem_cinza, 
        255, # Se for diferente do fundo, pinta de branco (255)
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 
        79, 
        2
    )

    # Inverter a máscara para ficar com o fundo preto e resistor branco
    mascara_resistor = cv2.bitwise_not(mascara_fundo)

    # Preenchendo os buracos com Fechamento Morfológico
    # Quadrado 75x75 pixels para engolir as faixas
    kernel = np.ones((75, 75), np.uint8)
    
    # Operação de Fechamento (MORPH_CLOSE)
    mascara_solida = cv2.morphologyEx(mascara_resistor, cv2.MORPH_CLOSE, kernel)

    # Teste para verificar as máscaras
    cv2.imshow("4 - Mascara Global (Com buracos)", mascara_resistor)
    cv2.imshow("5 - Mascara Solida (Corrigida)", mascara_solida)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Passo 3: Cores HSV
    