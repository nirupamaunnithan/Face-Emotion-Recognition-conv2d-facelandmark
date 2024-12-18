In this project we are implementing Face Emotion Recognition using facial landmarks to recognise whether the facial expression of the person is anger, sadness, happy, disgust, suprise or neutral

To get the facial landmark, the library named mediapipe is being used.

The dataset used is FER-2013 dataset The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.

The model consist of 10 layers including
- 2 convolutional layers (Conv2D) + 2 max-pooling layers (MaxPooling2D) for the image input.
- 2 dense layers for the landmark input.
- 1 concatenate layer.
- 2 dense layers for the combined features.
- 1 output layer (dense layer with 7 neurons).

The training accuracy of the model is 0.8486
The training validation accuracy is 0.5488
The test accuracy is 0.5488




