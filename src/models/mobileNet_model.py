import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Assuming you already have your dataset loaded as numpy arrays:
# potato_images (array of images), potato_labels (array of corresponding labels)

def train_mobilenet(potato_images, potato_labels):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(potato_images, potato_labels, test_size=0.2, random_state=42)

    # Normalize the image data (MobileNet expects input values between 0 and 1)
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Convert labels to categorical (one-hot encoding)
    num_classes = len(np.unique(y_train))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Load the MobileNet model with pre-trained ImageNet weights, excluding the top layers
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    # Add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Add a fully connected layer
    x = Dense(1024, activation='relu')(x)

    # Add the output layer with softmax activation for multi-class classification
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Image Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )

    # Train the model
    model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(classification_report(y_test_classes, y_pred_classes))

# Example usage:
# train_mobilenet(potato_images, potato_labels)
