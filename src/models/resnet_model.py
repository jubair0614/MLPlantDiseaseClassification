import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

def train_resnet(potato_images, potato_labels_encoded, epochs=2, batch_size=32, test_size=0.2):
    """
    Trains a ResNet50 model on the given dataset of potato images and labels.

    Parameters:
        potato_images (numpy array): Input images of potato leaves.
        potato_labels_encoded (numpy array): One-hot encoded labels for the images.
        epochs (int): Number of epochs to train the model. Default is 10.
        batch_size (int): Batch size for training. Default is 32.
        test_size (float): The proportion of the dataset to include in the test split. Default is 0.2.

    Returns:
        model: Trained ResNet50 model.
        history: Training history object containing loss and accuracy metrics.
    """
    
    # Ensure the labels are one-hot encoded (if not already)
    if len(potato_labels_encoded.shape) == 1 or potato_labels_encoded.shape[1] == 1:
        encoder = OneHotEncoder(sparse=False)
        potato_labels_encoded = encoder.fit_transform(potato_labels_encoded.reshape(-1, 1))
    
    # Step 1: Split the dataset into training and test sets (20% test data)
    X_train, X_test, y_train, y_test = train_test_split(potato_images, potato_labels_encoded, test_size=test_size, random_state=42)

    # Step 2: Data Augmentation using ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Step 3: Flow data from generators
    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
    test_generator = ImageDataGenerator().flow(X_test, y_test, batch_size=batch_size)

    # Step 4: Load Pretrained ResNet50 Model (without the top layer)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    # Step 5: Freeze the base layers
    for layer in base_model.layers:
        layer.trainable = False

    # Step 6: Add custom layers for classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    
    # Ensure the correct number of output classes
    num_classes = y_train.shape[1]  # Number of classes (columns in one-hot encoded labels)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Step 7: Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Step 8: Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Step 9: Train the model
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=epochs,
        steps_per_epoch=len(X_train) // batch_size,
        validation_steps=len(X_test) // batch_size
    )

    # Step 10: Evaluate the model on the test set
    y_test_pred = model.predict(X_test)
    y_test_pred_classes = y_test_pred.argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_test_pred_classes))

    return model, history