import cv2
import mediapipe as mp
import numpy as np

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Inicializa a tela de desenho
canvas = None
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontal para espelhar
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Prepara a imagem para MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Inicializa o canvas na primeira iteração
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Verifica se há alguma mão detectada
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Obtém a posição do indicador
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Desenha no canvas se o indicador estiver próximo ao polegar
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Checa a distância entre o indicador e o polegar
            if abs(x - thumb_x) < 20 and abs(y - thumb_y) < 20:
                cv2.circle(canvas, (x, y), 5, (255, 255, 255), -1)

            # Desenha os pontos da mão na tela
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Combina o canvas com o frame da câmera
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Exibe o resultado
    cv2.imshow("Desenho em tempo real", combined)

    # Sai ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
