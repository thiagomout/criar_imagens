import cv2
import numpy as np
import random
import sys

def criar_imagem():
    # Carregar a imagem original e a imagem de fundo
    img = cv2.imread('/home/thiago/img_micra/MD022508_training_set/Planktic/MD022508-0-0-Planktic-0009.jpg')
    fundo = cv2.imread('/home/thiago/img_micra/fundo/fundo1.jpg')

    #blur image
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Converter a imagem original para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Criar a máscara binária
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Inverter a máscara para isolar o objeto
    mask_inverted = 255 - mask

    # Redimensionar a imagem de fundo para o tamanho da imagem original
    fundo_redimensionado = cv2.resize(fundo, (img.shape[1], img.shape[0]))

    # Isolar o objeto da imagem original usando a máscara
    img_cortada = cv2.bitwise_and(img, img, mask=mask)

    img_cortada = cv2.GaussianBlur(img_cortada, (5, 5), 0)

    # Redimensionar a imagem cortada e a máscara
    escala = sys.argv[1]
    escala = float(escala)
    largura = int(img_cortada.shape[1] * escala)
    altura = int(img_cortada.shape[0] * escala)
    dimensao = (largura, altura)
    img_cortada_redimensionada = cv2.resize(img_cortada, dimensao)
    print("Dimensão da imagem cortada redimensionada:", img_cortada_redimensionada.shape)
    mask_resized = cv2.resize(mask, dimensao, interpolation=cv2.INTER_LANCZOS4)
    mask_inverted_resized = cv2.resize(mask_inverted, dimensao, interpolation=cv2.INTER_LANCZOS4)

    # Garantir que a máscara redimensionada tenha o mesmo número de canais que a imagem
    mask_resized_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
    mask_inverted_resized_bgr = cv2.cvtColor(mask_inverted_resized, cv2.COLOR_GRAY2BGR)

    # Definir a posição para colar a imagem cortada no fundo
    x_offset = random.randint(0, 319)
    y_offset = random.randint(0, 319)
    #x_offset, y_offset = 50, 50  # Ajustar conforme necessário
    print(x_offset, y_offset)

    # Copiar a imagem cortada redimensionada para o fundo
    fundo_redimensionado[y_offset:y_offset + altura, x_offset:x_offset + largura] = cv2.bitwise_and(fundo_redimensionado[y_offset:y_offset + altura, x_offset:x_offset + largura], mask_inverted_resized_bgr)
    fundo_redimensionado[y_offset:y_offset + altura, x_offset:x_offset + largura] += img_cortada_redimensionada
    print(fundo_redimensionado.shape)

    # Salvar a imagem
    cv2.imwrite('resultado_final2.png', fundo_redimensionado)

    # Exibir o resultado final
    cv2.imshow("Resultado Final", fundo_redimensionado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    criar_imagem()