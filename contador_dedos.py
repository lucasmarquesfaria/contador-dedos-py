from pathlib import Path
import cv2
import mediapipe as mp

# Iniciar a captura de vídeo da câmera (câmera web) com índice 0
video = cv2.VideoCapture(0)

# Inicializar o módulo de detecção de mãos
hand = mp.solutions.hands
Hands = hand.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

while True:
    # Ler um quadro de vídeo
    check, img = video.read()

    # Converter a imagem para formato RGB (necessário para o Mediapipe)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Processar a imagem para detectar as mãos
    results = Hands.process(imgRGB)
    handsPoints = results.multi_hand_landmarks
    h, w, _ = img.shape
    pontos = []

    if handsPoints:
        for landmarks in handsPoints:
            # Desenhar os pontos da mão e as conexões
            mpDraw.draw_landmarks(img, landmarks, hand.HAND_CONNECTIONS)

            for id, cord in enumerate(landmarks.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                pontos.append((cx, cy))

        if len(pontos) >= 21:  # Certificar-se de que haja pontos suficientes
            dedos = [8, 12, 16, 20]
            contador = 0
            for x in dedos:
                if pontos[x][1] < pontos[x - 2][1]:
                    contador += 1
            # Mostrar o número de dedos na imagem
            cv2.putText(img, str(contador), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # Mostrar a imagem com os pontos da mão
    cv2.imshow("Imagem", img)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
video.release()
cv2.destroyAllWindows()
