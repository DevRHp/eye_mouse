# eye_mouse
O projeto visa melhorar a acessibilidade em computadores para pessoas com deficiências motoras, permitindo o controle do mouse por meio dos movimentos oculares. Utilizando Python, OpenCV e MediaPipe, o sistema oferece uma alternativa intuitiva e assistiva, dispensando dispositivos de entrada tradicionais.

O projeto visa melhorar a acessibilidade em computadores para pessoas com deficiências motoras, permitindo o controle do mouse por meio dos movimentos oculares. Utilizando Python, OpenCV e MediaPipe, o sistema oferece uma alternativa intuitiva e assistiva, dispensando dispositivos de entrada tradicionais.


#made by: Rafael Henrique Prudencio
#contact: https://github.com
#         https://www.linkedin.com/in/rafael-prudencio-4222382b6/
#date of creation: 2025/01/08
#version: 2.3.1




import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()
calibration_mode = False
calibration_points = []
click_threshold = 0.25

last_eye_x, last_eye_y = screen_width // 2, screen_height // 2
smooth_factor = 0.3

def calibrate_points(landmarks):
    global calibration_points
    if len(calibration_points) < 5:
        x, y = int(landmarks.x * screen_width), int(landmarks.y * screen_height)
        calibration_points.append((x, y))
        print(f"Calibração ponto {len(calibration_points)}: {x}, {y}")
    return calibration_points

def map_eyes_to_screen(left_eye, right_eye, sensitivity=1.5):
    eye_x = int((left_eye.x + right_eye.x) * screen_width / 2 * sensitivity)
    eye_y = int((left_eye.y + right_eye.y) * screen_height / 2 * sensitivity)
    return eye_x, eye_y

def detect_blink(landmarks):
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[23]
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[253]
    
    left_eye_dist = np.linalg.norm([left_eye_top.x - left_eye_bottom.x, left_eye_top.y - left_eye_bottom.y])
    right_eye_dist = np.linalg.norm([right_eye_top.x - right_eye_bottom.x, right_eye_top.y - right_eye_bottom.y])
    
    if left_eye_dist < 0.025 and right_eye_dist < 0.025:
        return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = face_landmarks.landmark[145]
            right_eye = face_landmarks.landmark[374]
            
            if calibration_mode:
                calibration_points = calibrate_points(left_eye)
            else:
                eye_x, eye_y = map_eyes_to_screen(left_eye, right_eye)
                
                eye_x = int(last_eye_x + smooth_factor * (eye_x - last_eye_x))
                eye_y = int(last_eye_y + smooth_factor * (eye_y - last_eye_y))
                
                pyautogui.moveTo(eye_x, eye_y)
                
                last_eye_x, last_eye_y = eye_x, eye_y
                
                if detect_blink(face_landmarks.landmark):
                    pyautogui.click()
                    print("Clique detectado!")

            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

    cv2.imshow("Eye Controlled Mouse", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        calibration_mode = not calibration_mode
        if calibration_mode:
            print("Modo de calibração ativado")
        else:
            print("Modo de calibração desativado")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

