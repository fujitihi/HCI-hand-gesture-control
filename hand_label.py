import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

output_file = 'hand_data.csv'
if not os.path.exists(output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
        writer.writerow(header)

record_ready = False

cap = cv2.VideoCapture(0)

print("s to start take, 0-9 to choose label, q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    mode_text = "Recording: ON" if record_ready else "Recording: OFF"
    cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0) if record_ready else (0, 0, 255), 2)
            
    cv2.imshow("Hand Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        record_ready = not record_ready
        if record_ready:
            print("Ready to take:")
        else:
            print("Stop")

    elif record_ready and key == ord('d'):
        try:
            with open(output_file, 'r', newline='') as f:
                lines = f.readlines()
            if len(lines) > 1:
                with open(output_file, 'w', newline='') as f:
                    f.writelines(lines[:-1])
                print("Latest data deleted")
            else:
                print("There is only head")
        except Exception as e:
            print("Error:", e)

    elif key == ord('q'):
        break

    elif record_ready and key in [ord(str(d)) for d in range(10)]:
        label = chr(key)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            data = [label] + [coord for lm in hand.landmark for coord in (lm.x, lm.y, lm.z)]

            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)

            print(f"Label {label} saved")

        else:
            print("I cannot find your hand")

cap.release()
cv2.destroyAllWindows()
