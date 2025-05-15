import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import os

def create_data_generators(train_dir, validation_dir):
    """
    Create data generators for training and validation datasets.

    Parameters:
    - train_dir: Directory path for the training images.
    - validation_dir: Directory path for the validation images.

    Returns:
    - train_generator: Generator for training data.
    - validation_generator: Generator for validation data.
    """
    # Check if directories exist
    if not os.path.exists(train_dir):
        print(f"Training directory {train_dir} does not exist.")
        return None, None
    if not os.path.exists(validation_dir):
        print(f"Validation directory {validation_dir} does not exist.")
        return None, None

    print("Directories verified.")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    try:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )
    except Exception as e:
        print(f"Error in creating data generators: {e}")
        return None, None

    print("Data generators created successfully.")
    print(f"Classes found: {train_generator.class_indices}")
    
    return train_generator, validation_generator

if __name__ == "__main__":
    train_dir = 'E:\\College Files\\Projects\\Garbage Classification\\FINAL PROJECT\\Garbage classification'
    validation_dir = 'E:\\College Files\\Projects\\Garbage Classification\\FINAL PROJECT\\Garbage classification'

    train_generator, validation_generator = create_data_generators(train_dir, validation_dir)
    if train_generator and validation_generator:
        print("Data generators are ready for training.")
    else:
        print("Data generators could not be created.")
