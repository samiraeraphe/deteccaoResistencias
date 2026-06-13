# De início vamos testar tudo na imagem teste pq ela é muito boa

# Passo a passo dado pelo gemini
# Passo 1: Carregar e Limpar (Pré-processamento)
    #Filtro Bilateral. Para "embaçar" o fundo (listras do resistor afiadas)
# Passo 2: Isolar o Resistor (Segmentação)
    #Transformamos a imagem para tons de cinza e usamos um detector de bordas (como o algoritmo Canny) 
    # para achar os contornos. Mudança do método utilizado.
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

# Colour_range é uma variável de https://github.com/SupreethRao99/CVResist.git
# Mas fizemos algumas alterações de valores pq com os valores originais tava dando muito erro
# principalmente no verde e amarelo, eles estavam sobrepostos
# Estamos utilizando a lógica dele de segmentação pq pelo canny não funcionou
# O canny foi a primeira ideia

# Colours are thresholded in the HSV colour space. more can be found at (https://en.wikipedia.org/wiki/HSL_and_HSV)
Colour_Range = [
    # Limites inferior e superior do HSV, nome da cor, valor numérico da cor (potêncai de 10 para multiplicar o 
    # último valor), cor para desenhar o retângulo
    [(0, 0, 0), (179, 255, 30), "BLACK", 0, (0, 0, 0)],
    [(0, 90, 10), (15, 250, 100), "BROWN", 1, (0, 51, 102)], 
    [(0, 150, 80), (10, 255, 255), "RED", 2, (0, 0, 255)],
    [(11, 150, 150), (22, 255, 255), "ORANGE", 3, (0, 128, 255)], # Laranja: 11 a 22
    [(23, 100, 100), (35, 255, 255), "YELLOW", 4, (0, 255, 255)], # Amarelo: 23 a 35
    [(36, 100, 50), (89, 255, 255), "GREEN", 5, (0, 255, 0)],     # Verde: 36 a 89
    [(90, 150, 0), (139, 255, 255), "BLUE", 6, (255, 0, 0)],      # Azul: 90 a 139
    [(140, 40, 100), (159, 250, 220), "VIOLET", 7, (255, 0, 127)],
    [(0, 0, 50), (179, 50, 80), "GRAY", 8, (128, 128, 128)],
    [(0, 0, 90), (179, 15, 250), "WHITE", 9, (255, 255, 255)],
]

# Função para verificar se a mancha achada tem formato de "faixa" (alta e fina)
def eh_uma_faixa_valida(contorno):
    # Ignora sujeiras pequenas
    if cv2.contourArea(contorno) < 100: 
        return False
    
    x, y, largura, altura = cv2.boundingRect(contorno)
    
    # Se a largura for mais de 40% da altura, é muito largo para ser uma faixa
    if float(largura) / altura > 0.40: 
        return False
        
    return True

# -------------------------------------------------
# Passo 1: Carregar a imagem e aplicar o filtro bilateral para suavizar sem perder as bordas

image_path = 'testresistor1.jpg'
original_image = cv2.imread(image_path)

if original_image is None:
    print("Erro ao carregar a imagem. Verifique o caminho.")
else:
    # Redimensiona a imagem para uma largura de 800 pixels, mantendo a proporção
    # gemini recomenda pq se a imagem for gigante, trava, então é uma garantia
    proportion = 800.0/original_image.shape[1]
    new_size = (800, int(original_image.shape[0] * proportion))
    resized_image = cv2.resize(original_image, new_size) # imagem redimensionada

    # Adicionando margem para não dar problema ao colocar o quadradinho do resistor
    # Parâmetros: (imagem, margem_topo, margem_baixo, margem_esq, margem_dir, tipo_borda, cor_BGR)
    # Adicionando 100 pixels no teto e no chão, e 50 pixels nas laterais. Cor Branca = (255, 255, 255)
    resized_image = cv2.copyMakeBorder(resized_image, 100, 100, 50, 50, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # ------------------------------------------

    # O Filtro Bilateral continua igual, mas agora roda na imagem com borda!
    filtered_image = cv2.bilateralFilter(resized_image, 15, 80, 80)

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
    # Tampa os buracos e une as partes do resistor que foram separadas pelas faixas
    # Depois vamos usar essa máscara corrigida para filtrar as cores, garantindo que só analisamos o resistor
    mascara_solida_fechamento = cv2.morphologyEx(mascara_resistor, cv2.MORPH_CLOSE, kernel)

    # Teste para verificar as máscaras
    cv2.imshow("4 - Mascara Global (Com buracos)", mascara_resistor)
    cv2.imshow("5 - Mascara Solida (Corrigida)", mascara_solida_fechamento)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Passo 3: Cores HSV
    '''
    # Testando com uma cor só primeiro 
'
    # Convertendo a imagem filtrada para o espaço de cores HSV
    image_hsv = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)

    # Calibrações do AZUL da tabela Colour_Range
    # [(100, 150, 0), (140, 255, 255), "BLUE", 6, (255, 0, 0)]
    blue_inferior_limite = np.array([100, 150, 0])
    blue_superior_limite = np.array([140, 255, 255])

    blue_mask = cv2.inRange(image_hsv, blue_inferior_limite, blue_superior_limite)
    blue_mask_result = cv2.bitwise_and(blue_mask, mascara_solida_fechamento)

    # Mostrar os resultados
    #cv2.imshow("6 - TUDO que e Azul", blue_mask)
    #cv2.imshow("7 - Azul no Resistor", blue_mask_result)

    # Funcionou pro azul!

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    '''
    # Fazendo para todas as cores da tabela:
    
    # Convertendo a imagem filtrada para o espaço de cores HSV
    image_hsv = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)

    # O vermelho é dividido em 2 no hsv, então tem que fazer 2 máscaras e juntar depois
    red_top_high = np.array([179, 255, 200])
    red_top_low = np.array([160, 150, 80])

    faixas_encontradas = [] # Lista para armazenar as faixas encontradas

    # Loop para cada cor na tabela
    for color in Colour_Range:
        # pega a primeira e segunda tupla da lista (que são os limites inferior e superior do HSV)
        color_mask = cv2.inRange(image_hsv, color[0], color[1])

        if color[2] == "RED": # Se for vermelho, tem que fazer a máscara extra e juntar
            red_mask = cv2.inRange(image_hsv, red_top_low, red_top_high)
            color_mask = cv2.bitwise_or(red_mask, color_mask)

        # Só aceita a cor se ela estiver dentro da Máscara Sólida (no resistor)
        color_mask_result = cv2.bitwise_and(color_mask, mascara_solida_fechamento)

        # Acha os contornos das manchas que sobraram na máscara limpa
        contours, _ = cv2.findContours(color_mask_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop para cada contorno encontrado
        for contour in contours:
            if eh_uma_faixa_valida(contour):
                x, y, largura, altura = cv2.boundingRect(contour)

                # Desenha um retângulo envolta da cor achada na imagem original
                # Fazendo isso pq tava dando erro, aí queria ver onde era
                cv2.rectangle(resized_image, (x, y), (x + largura, y + altura), color[4], 2)

                # Se for uma faixa válida, adiciona à lista de faixas encontradas
                faixas_encontradas.append((x,color[2], color[3])) # Guarda a posição x, o nome da cor e o valor numérico da cor

    if faixas_encontradas:
        # Ordena as faixas pela posição x (da esquerda para a direita)
        faixas_ordenadas = sorted(faixas_encontradas, key=lambda x: x[0])

        print("Faixas encontradas (da esquerda para a direita):")
        for faixa in faixas_ordenadas:
            print(f"Cor: {faixa[1]}, Valor: {faixa[2]}")   

        # Passo 4: Calcular o valor da resistência com base nas faixas encontradas
        if len(faixas_ordenadas) in [3, 4, 5]: # Verifica se o número de faixas é válido (3, 4 ou 5)
            valor_resistencia = ""
            for faixa in faixas_ordenadas[:-1]: # Para as faixas de valor (todas menos a última)
                valor_resistencia += str(faixa[2]) # Concatena o valor numérico da cor

            valor_resistencia = int(valor_resistencia)
            multiplicador = 10 ** faixas_ordenadas[-1][2] # O multiplicador é 10 elevado ao valor da última faixa
            # [-1][2] pega a última faixa e retorna o valor numérico dela
            valor_resistencia *= multiplicador 

            texto_resultado = f"{valor_resistencia} Ohms"
            print(f"Valor da resistencia: {valor_resistencia} ohms") 

           # Aqui é só para colocar um quadradinho no resistor e deixar a resistência calculada na imagem
            
            #Acha o contorno da Máscara Sólida (o corpo todo do resistor)
            contornos_resistor, _ = cv2.findContours(mascara_solida_fechamento, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contornos_resistor:
                # Pega o maior contorno (para ignorar sujeiras)
                maior_contorno_resistor = max(contornos_resistor, key=cv2.contourArea)
                
                # Pega as coordenadas X, Y, Largura e Altura do resistor
                xr, yr, larg_r, alt_r = cv2.boundingRect(maior_contorno_resistor)
                
                # Desenha uma caixa fina ao redor do resistor inteiro
                cv2.rectangle(resized_image, (xr, yr), (xr + larg_r, yr + alt_r), (200, 200, 200), 1)

                # Calcula a posição do texto (Um pouco acima do Y do resistor)
                posicao_texto = (xr, yr - 15) 
                
                # Segurança: Se o resistor estiver muito colado no topo da foto, escreve embaixo dele
                if yr - 15 < 20: 
                    posicao_texto = (xr, yr + alt_r + 30)
                
                # Escreve o texto grudado no resistor                                            tam_letra, cor preta, espessura letra
                cv2.putText(resized_image, texto_resultado, posicao_texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            

        else:
            print("Numero de faixas encontrado eh invalido para calculo da resistencia")

    else:
        print("Nenhuma faixa encontrada")

    cv2.imshow("Resultado Final", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()         
