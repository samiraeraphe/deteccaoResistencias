# Funciona para testresistor.jpg e testresistor1.jpg
# não funciona em testresistor3.jpg pq as cores estão ruins


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
    [(0, 0, 0), (179, 255, 100), "BLACK", 0, (0, 0, 0)],
    [(0, 80, 45), (15, 255, 130), "BROWN", 1, (0, 51, 102)],
    [(0, 150, 80), (10, 255, 255), "RED", 2, (0, 0, 255)],
    [(11, 150, 150), (18, 255, 255), "ORANGE", 3, (0, 128, 255)],
    [(24, 100, 100), (35, 255, 255), "YELLOW", 4, (0, 255, 255)],
    [(36, 100, 50), (89, 255, 255), "GREEN", 5, (0, 255, 0)],     # Verde: 36 a 89
    [(90, 150, 0), (139, 255, 255), "BLUE", 6, (255, 0, 0)],      # Azul: 90 a 139
    [(140, 40, 100), (159, 250, 220), "VIOLET", 7, (255, 0, 127)],
    [(0, 0, 50), (179, 50, 80), "GRAY", 8, (128, 128, 128)],
    # Aumentamos a saturação permitida e baixamos o brilho para o Branco 
    [(0, 0, 150), (179, 40, 255), "WHITE", 9, (255, 255, 255)],
]

# Função para verificar se a mancha achada tem formato de "faixa" (alta e fina)
def eh_uma_faixa_valida(contorno):
    # Ignora sujeiras pequenas
    if cv2.contourArea(contorno) < 250: 
        return False
    
    x, y, largura, altura = cv2.boundingRect(contorno)
    
    # Se a largura for mais de 40% da altura, é muito largo para ser uma faixa
    if float(largura) / altura > 0.40: 
        return False
    
        
    return True


# -------------------------------------------------
# Passo 1: Carregar a imagem e aplicar o filtro bilateral para suavizar sem perder as bordas

image_path = 'images_ok\\testresistor4.jpg'
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
 

    # Filtro Bilateral para suavizar a imagem sem perder as bordas
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
   # Isolar o Resistor (Segmentação Robusta por Saturação)
    
    # Converte para HSV para pegar a Saturação (a quantidade de cor)
    imagem_hsv_mascara = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)
    canal_saturacao = imagem_hsv_mascara[:, :, 1]
    
    # Tudo que tem cor (Sat > 30) fica branco. Fundo preto ou branco fica preto
    _, mascara_cores = cv2.threshold(canal_saturacao, 30, 255, cv2.THRESH_BINARY)

    # Fechamento leve para tampar buraquinhos (15x15 para não deformar)
    kernel_suave = np.ones((15, 15), np.uint8)
    mascara_fechada = cv2.morphologyEx(mascara_cores, cv2.MORPH_CLOSE, kernel_suave)

    # Acha o maior objeto da tela e ignora o resto
    contornos_fundo, _ = cv2.findContours(mascara_fechada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Cria uma tela preta limpa
    mascara_solida_fechamento = np.zeros_like(mascara_fechada) 
    
    if contornos_fundo:
        # Acha o resistor e pinta ele de branco na nossa tela limpa
        maior_contorno = max(contornos_fundo, key=cv2.contourArea)
        cv2.drawContours(mascara_solida_fechamento, [maior_contorno], -1, 255, thickness=cv2.FILLED)

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

        # Costura pedaços de faixas que foram cortadas ao meio por reflexos de luz
        kernel_costura = np.ones((25, 5), np.uint8)
        color_mask_result = cv2.morphologyEx(color_mask_result, cv2.MORPH_CLOSE, kernel_costura)
   
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
                retangulo_girado = cv2.minAreaRect(contour)
                x_centro = int(retangulo_girado[0][0])
                
                # --- NOVO: Calculamos a área para usar como critério de desempate ---
                area_contorno = cv2.contourArea(contour)
                faixas_encontradas.append((x_centro, color[2], color[3], area_contorno))
                

    # =======================================================================
    # PASSO 4: FILTRAGEM ESPACIAL E CÁLCULO
    # =======================================================================
    if faixas_encontradas:
        # Ordena da esquerda para a direita
        faixas_ordenadas = sorted(faixas_encontradas, key=lambda x: x[0])

        # --- A MÁGICA DA SUPRESSÃO DE SOBREPOSIÇÃO ---
        faixas_sem_sobreposicao = []
        
        for faixa in faixas_ordenadas:
            if not faixas_sem_sobreposicao:
                faixas_sem_sobreposicao.append(faixa)
            else:
                ultima_salva = faixas_sem_sobreposicao[-1]
                
                # Se a distância X for menor que 25 pixels, elas estão na mesma listra física!
                if abs(faixa[0] - ultima_salva[0]) < 25:
                    # Desempate: Quem tem a maior área fica.
                    if faixa[3] > ultima_salva[3]:
                        faixas_sem_sobreposicao[-1] = faixa # Substitui o perdedor pelo ganhador
                else:
                    faixas_sem_sobreposicao.append(faixa) # É uma faixa nova, apenas adiciona
        # ---------------------------------------------

        print("\n--- RESULTADO DA LEITURA (PÓS-FILTRO) ---")
        
        # AGORA USAMOS A LISTA LIMPA (faixas_sem_sobreposicao) PARA O RESTO DO CÓDIGO
        for faixa in faixas_sem_sobreposicao:
            print(f"Cor: {faixa[1]}, Valor: {faixa[2]}") 

        # (IGNORANDO A TOLERÂNCIA)
        
        # --- MUDANÇA: Medimos o total de faixas usando a lista LIMPA e com o len() ---
        total_faixas = len(faixas_sem_sobreposicao)
        
        # Só tenta calcular se achou pelo menos 4 faixas (2 digitos + 1 multiplicador + 1 tolerância)
        if total_faixas >= 4: 

            # --- MUDANÇA: Usamos a lista limpa (faixas_sem_sobreposicao) para a matemática ---
            # Separa a última faixa (a tolerância) e guarda o resto para o cálculo
            faixa_tolerancia = faixas_sem_sobreposicao[-1]
            faixas_de_calculo = faixas_sem_sobreposicao[:-1] 
            
            print(f"\nIgnorando a ultima faixa (Tolerancia assumida): {faixa_tolerancia[1]}")
            
            valor_resistencia_string = ""
            
            # Pega todas as faixas (exceto a última que agora é o multiplicador) e junta os números
            for faixa in faixas_de_calculo[:-1]: 
                valor_resistencia_string += str(faixa[2]) 

            valor_base_inteiro = int(valor_resistencia_string)
            
            # O multiplicador é a última faixa da nossa lista de cálculo
            multiplicador = 10 ** faixas_de_calculo[-1][2] 
            
            resistencia_final = valor_base_inteiro * multiplicador 

            texto_resultado = f"{resistencia_final} Ohms"
            print(f"Valor da resistencia: {texto_resultado}\n") 

            #Acha o contorno da Máscara Sólida (o corpo todo do resistor)
            contornos_resistor, _ = cv2.findContours(mascara_solida_fechamento, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contornos_resistor:
                maior_contorno_resistor = max(contornos_resistor, key=cv2.contourArea)
                xr, yr, larg_r, alt_r = cv2.boundingRect(maior_contorno_resistor)
                
                cv2.rectangle(resized_image, (xr, yr), (xr + larg_r, yr + alt_r), (200, 200, 200), 1)

                posicao_texto = (xr, yr - 15) 
                
                if yr - 15 < 20: 
                    posicao_texto = (xr, yr + alt_r + 30)
                
                cv2.putText(resized_image, texto_resultado, posicao_texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
        else:
            print("\nNumero de faixas encontrado eh insuficiente para calculo (Minimo de 4).")

    else:
        print("\nNenhuma faixa encontrada")

    cv2.imshow("Resultado Final", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()