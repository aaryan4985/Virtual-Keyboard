# keyboard.py

import cv2
import mediapipe as mp
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Initialize MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Virtual Keyboard layout
key_width, key_height = 100, 100
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M"],
    ["Space"]
]

# Function to draw the keyboard on screen
def draw_keyboard(img):
    y_offset = 100  # Starting Y position
    for row in keys:
        x_offset = 100  # Starting X position
        for key in row:
            # Draw rectangle for each key
            cv2.rectangle(img, (x_offset, y_offset), 
                          (x_offset + key_width, y_offset + key_height), (0, 255, 0), 2)
            # Add text to each key
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, key, (x_offset + 35, y_offset + 65), font, 1, (0, 0, 255), 2)
            x_offset += key_width + 10  # Space between keys
        y_offset += key_height + 10  # Space between rows

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror image for natural control
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect hand
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Draw the virtual keyboard
    draw_keyboard(img)

    cv2.imshow("Virtual Keyboard", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
