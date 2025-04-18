# virtual_keyboard.py

import cv2
import numpy as np
import mediapipe as mp
from math import hypot

# Mediapipe initialization
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Keyboard Layout
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M"],
    ["Space", "←", "⏎"]
]

# Store typed text
typed_text = ""
hovered_key = ""
key_to_add = ""

# Function to draw keys
def draw_keyboard(img):
    global hovered_key
    hovered_key = ""
    key_positions = []
    start_y = 100
    for i, row in enumerate(keys):
        start_x = 100 + (50 * i)  # shift rows
        for j, key in enumerate(row):
            x = start_x + j * 60
            y = start_y + i * 70
            w, h = 55, 55
            key_positions.append((key, (x, y)))

            # Draw key box
            color = (255, 0, 255) if hovered_key == key else (255, 255, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
            cv2.putText(img, key, (x + 10, y + 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    return key_positions

# Function to calculate distance between two points
def distance(p1, p2):
    return hypot(p2[0] - p1[0], p2[1] - p1[1])

# Main loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    key_positions = draw_keyboard(img)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                lmList.append((int(lm.x * w), int(lm.y * h)))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if lmList:
                # Index fingertip and thumb tip
                finger_tip_pos = lmList[8]
                thumb_tip_pos = lmList[4]

                # Show fingertip as cursor
                cv2.circle(img, finger_tip_pos, 10, (0, 255, 0), cv2.FILLED)

                # Detect key hover
                for key, pos in key_positions:
                    x, y = pos
                    if x < finger_tip_pos[0] < x + 55 and y < finger_tip_pos[1] < y + 55:
                        hovered_key = key
                        cv2.rectangle(img, (x, y), (x + 55, y + 55), (0, 255, 0), 3)

                        # Pinch Detection for Click
                        if distance(finger_tip_pos, thumb_tip_pos) < 40:
                            if hovered_key == "Space":
                                key_to_add = " "
                            elif hovered_key == "←":
                                typed_text = typed_text[:-1]
                                key_to_add = ""
                            elif hovered_key == "⏎":
                                print("Final Text:", typed_text)
                                typed_text = ""
                                key_to_add = ""
                            else:
                                key_to_add = hovered_key
                            typed_text += key_to_add
                            print("Pressed:", hovered_key)
                            cv2.waitKey(200)  # delay to avoid multiple keypresses

    # Display typed text
    cv2.rectangle(img, (100, 30), (1000, 80), (0, 0, 0), -1)
    cv2.putText(img, typed_text, (110, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)

    cv2.imshow("Virtual Keyboard", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
