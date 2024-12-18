from preprocessing import preprocess_data
import numpy as np
from tensorflow.keras import layers, models, Model, Input
from sklearn.model_selection import train_test_split


folder = 'FER2013 copy'
X_images, X_landmarks, y_data = preprocess_data(folder)

def create_model():
    # Image Input Branch
    image_input = Input(shape=(48, 48, 1), name='image_input')
    x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)

    # Landmark Input Branch
    landmark_input = Input(shape=(936,), name='landmark_input')
    y = layers.Dense(128, activation='relu')(landmark_input)
    y = layers.Dense(64, activation='relu')(y)

    # Combine Features
    combined = layers.concatenate([x, y])
    z = layers.Dense(128, activation='relu')(combined)
    z = layers.Dense(7, activation='softmax')(z)

    # Build and Compile Model
    model = Model(inputs=[image_input, landmark_input], outputs=z)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


X_train_img, X_test_img, X_train_land, X_test_land, y_train, y_test = train_test_split(
        X_images, X_landmarks, y_data, test_size=0.2, random_state=42)

model = create_model()
history = model.fit([X_train_img, X_train_land], y_train, epochs=15, batch_size=32, validation_data=([X_test_img, X_test_land], y_test))

# To print each training accuracy
for epoch in range(len(history.history['accuracy'])):
    train_acc = history.history['accuracy'][epoch]
    val_acc = history.history['val_accuracy'][epoch]
    print(f"Epoch {epoch+1}: Training Accuracy = {train_acc:.4f}, Validation Accuracy = {val_acc:.4f}")

test_loss, test_acc = model.evaluate([X_test_img, X_test_land], y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

model.save('Conv2D_FER_with_landmarks.h5')

