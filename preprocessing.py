import os
import cv2
import numpy as np
import mediapipe as mp
from keras.utils import to_categorical

# Emotion labels
emotion_labels = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def get_landmarks(image):
    """
    Extract 468 facial landmarks and flatten into a vector.
    """
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        return np.array([(l.x, l.y) for l in landmarks]).flatten()  # Flatten (468, 2) -> (936,)
    return np.zeros(936)  # If no landmarks detected, return zeros

def preprocess_data(base_folder):
    X_images, X_landmarks, y_data = [], [], []

    # Iterate through 'train' and 'test' directories
    for split in ['train', 'test']:
        split_folder = os.path.join(base_folder, split)
        if not os.path.exists(split_folder):
            print(f"Missing split folder: {split_folder}")
            continue

        for label in os.listdir(split_folder):
            if label in emotion_labels:
                label_folder = os.path.join(split_folder, label)
                if not os.path.isdir(label_folder):
                    print(f"Skipping {label_folder}: Not a directory")
                    continue
                print(f"Processing {split}/{label}...")

                for img_name in os.listdir(label_folder):
                    img_path = os.path.join(label_folder, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Invalid or unreadable image: {img_path}")
                        continue
                    img = cv2.resize(img, (48, 48))
                    X_images.append(img)

                    # Extract landmarks
                    original_img = cv2.imread(img_path)
                    if original_img is not None:
                        landmarks = get_landmarks(original_img)
                    else:
                        print(f"Original image unreadable for landmarks: {img_path}")
                        landmarks = np.zeros(936)  # Placeholder for missing landmarks

                    X_landmarks.append(landmarks)
                    y_data.append(emotion_labels[label])

    print(f"Total samples processed: {len(X_images)}")
    if len(X_images) == 0:
        raise ValueError("No samples were found. Please check the dataset path and structure.")

    X_images = np.array(X_images)
    X_landmarks = np.array(X_landmarks)
    y_data = np.array(y_data)

    # Normalize and reshape
    X_images = X_images.astype('float32') / 255.0
    X_images = X_images.reshape(-1, 48, 48, 1)

    # One-hot encode labels
    y_data = to_categorical(y_data, num_classes=7)

    return X_images, X_landmarks, y_data







