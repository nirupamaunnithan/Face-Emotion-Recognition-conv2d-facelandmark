import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing import get_landmarks

def predict_emotion(image_path):
    # Load the trained model
    model = load_model('Conv2D_FER_with_landmarks.h5')

    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (48, 48))
    img_normalized = img_resized.astype('float32') / 255.0
    img_normalized = img_normalized.reshape(1, 48, 48, 1)

    # Extract landmarks
    full_img = cv2.imread(image_path)
    landmarks = get_landmarks(full_img)
    landmarks = landmarks.reshape(1, -1)  # Reshape for prediction

    # Predict emotion
    prediction = model.predict([img_normalized, landmarks])
    predicted_class = np.argmax(prediction)
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    return emotions[predicted_class]

test_image = 'images (3) copy.jpg'
emotion = predict_emotion(test_image)
print(f'Predicted Emotion: {emotion}')