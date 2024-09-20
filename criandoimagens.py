import cv2
import numpy as np

# Carregar a imagem original e a imagem de fundo
img = cv2.imread('/home/thiago/img_micra/MD022508_training_set/Planktic/MD022508-0-0-Planktic-0007.jpg')
fundo = cv2.imread('/home/thiago/img_micra/fundo/istockphoto-1209961067-612x612.jpg')

# Converter a imagem original para escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Criar a máscara binária
_, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Inverter a máscara para isolar o objeto (preservar o objeto)
mask_inverted = 255 - mask

# Redimensionar a imagem de fundo para ser do mesmo tamanho que a original
fundo_redimensionado = cv2.resize(fundo, (img.shape[1], img.shape[0]))

# Isolar o objeto da imagem original usando a máscara invertida
img_cortada = cv2.bitwise_and(img, img, mask=mask)

# Aumentar o contraste do objeto recortado
alpha = 1.0  # Contraste (1.0 = sem ajuste)
beta = 0    # Brilho (0 = sem ajuste)
img_cortada = cv2.convertScaleAbs(img_cortada, alpha=alpha, beta=beta)

# Isolar o fundo onde o objeto será colado (máscara invertida)
fundo_mascarado = cv2.bitwise_and(fundo_redimensionado, fundo_redimensionado, mask=mask_inverted)

# Combinar o objeto isolado com o fundo modificado
resultado_final = cv2.add(fundo_mascarado, img_cortada)

# Salvar o resultado final
cv2.imwrite('resultado_final.png', resultado_final)

# Exibir o resultado final
cv2.imshow("Resultado Final", resultado_final)
cv2.waitKey(0)
cv2.destroyAllWindows()
