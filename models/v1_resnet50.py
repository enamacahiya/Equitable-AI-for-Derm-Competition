# author: ena macahiya
# using cindy deng's provided starter code to create pipeline

# 1. import libraries
# data manipulation
import pandas as pd # data processing, CSV file i/o
import numpy as np # linear alg
import os
# sklearn: split data, encode labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
# - tensorflow.keras: for building and training the neural network
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

# 2. load data
train_df = pd.read_csv('C:\\Users\\Ena\\Downloads\\bttai-ajl-2025\\train.csv')
test_df = pd.read_csv('C:\\Users\\Ena\\Downloads\\bttai-ajl-2025\\test.csv')

# add .jpg extension to md5hash column to reference the file_name
train_df['md5hash'] = train_df['md5hash'].astype(str) + '.jpg'
test_df['md5hash'] = test_df['md5hash'].astype(str) + '.jpg'
# combine label and md5hash to form the correct path
train_df['file_path'] = train_df['label'] + '/' + train_df['md5hash']

# 3. data preprocessing
label_encoder = LabelEncoder() # encode labels
train_df['encoded_label'] = label_encoder.fit_transform(train_df['label'])
train_df['encoded_label'] = train_df['encoded_label'].astype(str)
non_augmented_df = train_df[~train_df['fitzpatrick_scale'].between(4, 6)]  # Select non-augmented images
augmented_df = train_df[train_df['fitzpatrick_scale'].between(4, 6)]  

# split to train and validation
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
# define img data generators for train and validation
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_dir = 'C:\\Users\\Ena\\Downloads\\bttai-ajl-2025\\train\\train'

def create_generator(dataframe, directory, batch_size=10, target_size=(224, 224), is_train=True): # mess with batch size and filter
    """
    Creates image generators for training and validation datasets.
    """
    if not is_train:
        datagen = ImageDataGenerator(rescale=1./255)  # no augmentation for validation
        generator = datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=directory,
            x_col='file_path',
            y_col='encoded_label',
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            validate_filenames=False
        )
        return generator

    non_augmented_datagen = ImageDataGenerator(rescale=1./255)
    augmented_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
        channel_shift_range=20.0,
        preprocessing_function=lambda x: tf.image.random_jpeg_quality(x, 70, 100)  # Inject noise via JPEG compression
    )

    non_augmented_train_generator = non_augmented_datagen.flow_from_dataframe(
        dataframe=non_augmented_df,
        directory=train_dir,
        x_col='file_path',
        y_col='encoded_label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        validate_filenames=False
    )
    augmented_train_generator = augmented_datagen.flow_from_dataframe(
        dataframe=augmented_df,
        directory=train_dir,
        x_col='file_path',
        y_col='encoded_label',
        target_size=(224, 224),  # Resizing the images to fit model input size (224x224 for ResNet50)
        batch_size=32,
        class_mode='categorical',
        validate_filenames=False
    )
    
    return non_augmented_train_generator, augmented_train_generator

# Create generators
non_augmented_train_generator, augmented_train_generator = create_generator(train_data, train_dir, is_train=True)
val_generator = create_generator(val_data, train_dir, is_train=False)

# Combine non-augmented and augmented generators
def combined_generator(gen1, gen2):
    while True:
        for data1, data2 in zip(gen1, gen2):
            yield (np.concatenate([data1[0], data2[0]], axis=0), np.concatenate([data1[1], data2[1]], axis=0))

# resnet50 model architecture
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers [:120]:  # freeze first 140 layers to prevent overfit
    layer.trainable = False
# custom layers
x = GlobalAveragePooling2D()(base_model.output)  # reduce dimensions
x = BatchNormalization()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)  # dropout for regularization
output = Dense(21, activation="softmax")(x)  # softmax for multi-class, 21 skin conditions
# softmax activation function: "transforms the raw outputs of the neural network into a vector of probabilities, essentially a probability distribution over the input classes."

# model
model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=Adam(learning_rate=1e-4),  # fine-tuning learning rate
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
# model.summary()

# train model
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True) # early stopping if in 5 epochs there is no improvement, prevents overfit

history = model.fit(
    combined_generator(augmented_train_generator, non_augmented_train_generator),
    epochs=10,
    steps_per_epoch=len(augmented_train_generator) + len(non_augmented_train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[early_stopping]
)

# 6. predict test data
def preprocess_test_data(test_df, directory):
    """
    Template for loading and preprocessing test images.
    """
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(
        ## set the test_generator here (COPIED FROM TRAIN, diff vals)
        dataframe=test_df,
        directory=directory,
        x_col="md5hash",  # image filename
        target_size=(224, 224),
        batch_size=1, # 1 image per batch
        class_mode=None,  # no labels
        shuffle=False  # ensure filenames match predictions
    )
    return test_generator

# load test data
test_dir = 'C:\\Users\\Ena\\Downloads\\bttai-ajl-2025\\test\\test'
test_generator = preprocess_test_data(test_df, test_dir)
# generate predictions
predictions = model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)
predicted_class_names = label_encoder.inverse_transform(predicted_labels)

clean_filenames = []
for filename in test_generator.filenames:
    clean_filenames.append(filename.replace('.jpg', ''))

# save to csv
submission = pd.DataFrame({
    "md5hash": clean_filenames,           # img file names
    "label": predicted_class_names        # predicted label
})
submission.to_csv("submission.csv", index=False)
print("saved file")