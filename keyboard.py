# keyboard.py

import cv2
import mediapipe as mp
import numpy as np
import math

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
def draw_keyboard(img, hovered_key=None):
    y_offset = 100  # Starting Y position
    for row in keys:
        x_offset = 100  # Starting X position
        for key in row:
            # Calculate key position on screen
            x1, y1 = x_offset, y_offset
            x2, y2 = x_offset + key_width, y_offset + key_height

            # Highlight hovered key with a different color
            if hovered_key == key:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), -1)  # Red fill
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green outline

            # Add text to each key
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, key, (x_offset + 35, y_offset + 65), font, 1, (0, 0, 255), 2)

            x_offset += key_width + 10  # Space between keys
        y_offset += key_height + 10  # Space between rows

# Function to check if the finger is hovering over a key
def is_hovering_over_key(finger_tip, key_position):
    fx, fy = finger_tip
    kx1, ky1, kx2, ky2 = key_position
    # Check if the finger is inside the key bounds
    if kx1 < fx < kx2 and ky1 < fy < ky2:
        return True
    return False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror image for natural control
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect hand
    results = hands.process(imgRGB)

    hovered_key = None  # Variable to store the hovered key

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Get position of index finger tip (landmark 8)
            index_finger_tip = handLms.landmark[8]
            finger_tip = (int(index_finger_tip.x * img.shape[1]), int(index_finger_tip.y * img.shape[0]))

            # Check each key if the finger is hovering over it
            y_offset = 100
            for row in keys:
                x_offset = 100
                for key in row:
                    # Calculate key position on screen
                    x1, y1 = x_offset, y_offset
                    x2, y2 = x_offset + key_width, y_offset + key_height

                    # If the finger is hovering over this key
                    if is_hovering_over_key(finger_tip, (x1, y1, x2, y2)):
                        hovered_key = key
                    x_offset += key_width + 10
                y_offset += key_height + 10

            # Draw landmarks on hand
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Draw the virtual keyboard with highlighted hovered key
    draw_keyboard(img, hovered_key)

    # Show the webcam feed with keyboard overlay
    cv2.imshow("Virtual Keyboard", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
