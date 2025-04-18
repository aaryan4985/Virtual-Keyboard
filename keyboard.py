# keyboard.py

import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Initialize MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Keyboard layout
key_width, key_height = 100, 100
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M"],
    ["Space"]
]

# Text input buffer
typed_text = ""
last_pressed_time = 0

def draw_keyboard(img, hovered_key=None):
    y_offset = 100
    for row in keys:
        x_offset = 100
        for key in row:
            x1, y1 = x_offset, y_offset
            x2, y2 = x_offset + key_width, y_offset + key_height
            if hovered_key == key:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), -1)  # Red for hover
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            label = " " if key == "Space" else key
            cv2.putText(img, label, (x_offset + 25, y_offset + 65), font, 1, (255, 255, 255), 2)

            x_offset += key_width + 10
        y_offset += key_height + 10

def is_hovering(finger_tip, key_pos):
    fx, fy = finger_tip
    x1, y1, x2, y2 = key_pos
    return x1 < fx < x2 and y1 < fy < y2

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)
    hovered_key = None
    finger_tip_pos = None

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = handLms.landmark
            index_tip = lmList[8]
            thumb_tip = lmList[4]

            finger_tip_pos = (int(index_tip.x * img.shape[1]), int(index_tip.y * img.shape[0]))
            thumb_tip_pos = (int(thumb_tip.x * img.shape[1]), int(thumb_tip.y * img.shape[0]))

            y_offset = 100
            for row in keys:
                x_offset = 100
                for key in row:
                    x1, y1 = x_offset, y_offset
                    x2, y2 = x_offset + key_width, y_offset + key_height

                    if is_hovering(finger_tip_pos, (x1, y1, x2, y2)):
                        hovered_key = key
                    x_offset += key_width + 10
                y_offset += key_height + 10

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Tap gesture detection
            if hovered_key and distance(finger_tip_pos, thumb_tip_pos) < 40:
                current_time = time.time()
                if current_time - last_pressed_time > 0.5:  # Debounce for 0.5s
                    key_to_add = " " if hovered_key == "Space" else hovered_key
                    typed_text += key_to_add
                    print("Pressed:", key_to_add)
                    last_pressed_time = current_time

    # Draw keyboard and typed text
    draw_keyboard(img, hovered_key)
    cv2.rectangle(img, (100, 20), (1100, 80), (0, 0, 0), -1)
    cv2.putText(img, typed_text, (110, 65), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    cv2.imshow("Virtual Keyboard", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
