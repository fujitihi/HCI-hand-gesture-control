import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

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

def load_and_extract(filename):
    X, y = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            label = row[0]
            coords = list(map(float, row[1:]))
            features = extract_features(coords)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

def main():
    X, y = load_and_extract("hand_data.csv")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    with open("gesture_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved:gesture_model.pkl")

if __name__ == "__main__":
    main()
