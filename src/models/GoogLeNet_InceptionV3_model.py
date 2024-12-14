import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_inceptionv3(potato_images, potato_labels):
    # Normalize image data
    potato_images = np.array(potato_images) / 255.0
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(potato_images, potato_labels, test_size=0.2, random_state=42)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Create the InceptionV3 model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(len(np.unique(potato_labels)), activation='softmax')(x)  # Assuming potato_labels are integer-encoded

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              validation_data=(X_val, y_val),
              epochs=5)  # You can adjust the number of epochs as needed

    # Evaluate the model on the validation set
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Get the predicted class labels

    # Print classification report
    print(classification_report(y_val, y_pred_classes))

    return model
