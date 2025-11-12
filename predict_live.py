import cv2
import mediapipe as mp
import pickle
import numpy as np

with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def extract_features(landmarks):
    lm = np.array(landmarks).reshape((21, 3))
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

cap = cv2.VideoCapture(0)

print("Show hand pose (q to quit)")

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

            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            features = extract_features(landmarks)
            prediction = model.predict([features])[0]

            cv2.putText(frame, f'Prediction: {prediction}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    else:
        cv2.putText(frame, "No hand detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow("Live Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
