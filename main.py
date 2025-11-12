# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import pyautogui as pag
import time
import math
import subprocess
import os
import threading
import pyaudio
import struct
import numpy as np
import signal
import pickle
import json

# Enable PyAutoGUI fail-safe (move mouse to corner to abort)
pag.FAILSAFE = False

# Audio input settings for clap detection
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CLAP_THRESHOLD = 4000

# Start audio stream
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# State tracking variables
clap_count = 0
clap_detected = False
mode_number = 0
keyboard_proc = None
pointer_proc = None
tempx, tempy = 0, 0

mouth_open = False
prev_mouth_open = False
num_hands_detected = 0

ref_finger_pos = None
ref_cursor_pos = None
anchor_set = False

gesture_last_time = {"6": 0, "8": 0, "5": 0}
gesture_cooldown = 1.5  # seconds
dragging = False
last_mode2_click_time = 0
mode2_click_cooldown = 1.5

# Load trained gesture recognition model
with open("gesture_model.pkl", "rb") as f:
    gesture_model = pickle.load(f)

def extract_features_from_landmarks(landmarks):
    lm = np.array([(lm.x, lm.y, lm.z) for lm in landmarks]).reshape((21, 3))
    origin = lm[0]
    rel_lm = lm - origin

    def get_angle(a, b, c):
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return cosine

    angles = []
    finger_indices = [
        (5, 6, 7), (6, 7, 8),
        (9,10,11), (10,11,12),
        (13,14,15), (14,15,16),
        (17,18,19), (18,19,20),
        (1, 2, 3), (2, 3, 4),
    ]
    for i1, i2, i3 in finger_indices:
        angles.append(get_angle(lm[i1], lm[i2], lm[i3]))

    thumb_tip = lm[4]
    index_tip = lm[8]
    tip_dist = np.linalg.norm(thumb_tip - index_tip)

    features = list(rel_lm.flatten()) + angles + [tip_dist]
    return features

# Background thread: listen to claps
def audio_listener():
    global clap_detected
    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = struct.unpack(str(CHUNK) + 'h', data)
            rms = np.sqrt(np.mean(np.square(audio_data)))
            if rms > CLAP_THRESHOLD:
                if num_hands_detected >= 2:
                    print("Clap detected (with two hands)")
                    clap_detected = True
        except Exception as e:
            print(f"Audio listener error: {e}")
            break

audio_thread = threading.Thread(target=audio_listener, daemon=True)
audio_thread.start()

# Start video capture and MediaPipe hands + face
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calc_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def is_mouth_open(landmarks):
    top = landmarks[13]
    bottom = landmarks[14]
    return calc_distance(top, bottom) > 0.03

def terminate_process(proc):
    if proc:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception as e:
            print(f"Failed to send CTRL_BREAK_EVENT: {e}")


screen_w, screen_h = pag.size()
x_min = (screen_w - 1000) // 2
x_max = x_min + 1000
y_min = (screen_h - 350) // 2
y_max = y_min + 350

keyboard_action_file = "keyboard_event.json"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    num_hands_detected = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0

    if clap_detected:
        clap_detected = False
        clap_count += 1
        mode_number = (clap_count - 1) % 3 + 1
        print(f"Mode switched to {mode_number}")

        terminate_process(keyboard_proc)
        keyboard_proc = None
        terminate_process(pointer_proc)
        pointer_proc = None

        if mode_number == 1:
            pass
        elif mode_number == 2:
            keyboard_proc = subprocess.Popen(["python", "keyboard_interface.py"], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            pointer_proc = subprocess.Popen(["python", "pointer_overlay.py"], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        elif mode_number == 3:
            clap_count = 0
            mode_number = 0

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            mouth_open = is_mouth_open(landmarks)

            if mode_number == 1:
                if not mouth_open and prev_mouth_open:
                    anchor_set = True

                if mouth_open:
                    if anchor_set:
                        if hand_results.multi_hand_landmarks:
                            for hand_landmarks in hand_results.multi_hand_landmarks:
                                index_finger_tip = hand_landmarks.landmark[8]
                                screen_w, screen_h = pag.size()
                                ref_finger_pos = (index_finger_tip.x, index_finger_tip.y)
                                ref_cursor_pos = pag.position()
                                anchor_set = False
                                print(f"[Saved] ref_finger: {ref_finger_pos}, ref_cursor: {ref_cursor_pos}")
                                break

                    if ref_finger_pos and ref_cursor_pos and hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            index_finger_tip = hand_landmarks.landmark[8]
                            dx = index_finger_tip.x - ref_finger_pos[0]
                            dy = index_finger_tip.y - ref_finger_pos[1]

                            screen_w, screen_h = pag.size()
                            scale = 1.8
                            new_x = int(ref_cursor_pos[0] + dx * screen_w * scale)
                            new_y = int(ref_cursor_pos[1] + dy * screen_h * scale)

                            new_x = max(1, min(new_x, screen_w - 2))
                            new_y = max(1, min(new_y, screen_h - 2))

                            print(f"[Move] dx: {dx:.3f}, dy: {dy:.3f} -> ({new_x}, {new_y})")
                            pag.moveTo(new_x, new_y)
                            tempx, tempy = new_x, new_y
                            break

                elif not mouth_open and hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        features = extract_features_from_landmarks(hand_landmarks.landmark)
                        pred_label = gesture_model.predict([features])[0]
                        cursor_x, cursor_y = pag.position()
                        if pred_label in gesture_last_time:
                            now = time.time()
                            if now - gesture_last_time[pred_label] > gesture_cooldown:
                                gesture_last_time[pred_label] = now
                                if pred_label == '6':
                                    print("[Gesture] Single Click")
                                    pag.click(cursor_x, cursor_y)
                                elif pred_label == '8':
                                    print("[Gesture] Double Click")
                                    pag.doubleClick(cursor_x, cursor_y)
                                elif pred_label == '5':
                                    print("[Gesture] Right Click")
                                    pag.rightClick(cursor_x, cursor_y)
                        if pred_label == '3':
                            if not dragging:
                                print("[Gesture] Drag Start")
                                pag.mouseDown(cursor_x, cursor_y)
                                dragging = True
                        else:
                            if dragging:
                                print("[Gesture] Drag End")
                                pag.mouseUp()
                                dragging = False

                prev_mouth_open = mouth_open

            if mode_number == 2:
                if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                    pointer_data = {}
                    for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                        label = handedness.classification[0].label
                        index_tip = hand_landmarks.landmark[8]
                        pointer_data[label] = {"x": index_tip.x, "y": index_tip.y}
                        
                    if mouth_open:
                        try:
                            with open("keyboard_pointer.json", "w", encoding="utf-8") as f:
                                json.dump(pointer_data, f)
                        except Exception as e:
                            print(f"[Pointer Write Error] {e}")
                    else:
                        try:
                            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                                features = extract_features_from_landmarks(hand_landmarks.landmark)
                                pred_label = gesture_model.predict([features])[0]
                                label = handedness.classification[0].label

                                if pred_label == '3':
                                    try:
                                        with open("keyboard_pointer.json", "r", encoding="utf-8") as f:
                                            pointer_data = json.load(f)
                                    except Exception as e:
                                        print(f"[Pointer Read Error] {e}")
                                        pointer_data = {}

                                    if label in pointer_data:
                                        screen_w, screen_h = pag.size()
                                        x = int(pointer_data[label]["x"] * screen_w)
                                        y = int(pointer_data[label]["y"] * screen_h)
                                        x = max(x_min, min(x, x_max))
                                        y = max(y_min, min(y, y_max))
                                        
                                        now = time.time()
                                        if now - last_mode2_click_time > mode2_click_cooldown:
                                            last_mode2_click_time = now
                                            print(f"[Gesture Click] Hand: {label}, Pos: ({x}, {y})")
                                            terminate_process(pointer_proc)
                                            pointer_proc = None
                                            pag.click(x, y)
                                            time.sleep(0.1)
                                            pointer_proc = subprocess.Popen(["python", "pointer_overlay.py"], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                        except Exception as e:
                            print(f"[Mode2 Gesture Error] {e}")
                            
            if mode_number != 2 and pointer_proc is not None:
                terminate_process(pointer_proc)
                pointer_proc = None

    if os.path.exists(keyboard_action_file):
        try:
            with open(keyboard_action_file, "r", encoding="utf-8") as f:
                action_data = json.load(f)

            if isinstance(action_data, dict) and "type" in action_data: 
                pag.click(tempx, tempy)
                if action_data["type"] == "text":
                    pag.write(action_data["value"])
                    pag.press("enter")
                elif action_data["type"] == "shortcut":
                    pag.hotkey(*action_data["keys"])
            with open(keyboard_action_file, "w", encoding="utf-8") as f:
                json.dump({}, f)
        except Exception as e:
            print(f"[Keyboard Action Error] {e}")
            
    overlay_lines = [
        f"Clap Count: {clap_count}",
        f"Mode Number: {mode_number}",
        f"Mouth Open: {'Yes' if mouth_open else 'No'}",
        f"Hands Detected: {num_hands_detected}"
    ]
    for i, line in enumerate(overlay_lines):
        cv2.putText(frame, line, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Interface Controller", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
p.terminate()
terminate_process(keyboard_proc)
terminate_process(pointer_proc)
