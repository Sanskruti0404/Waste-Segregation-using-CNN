import os
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Dataset path
dataset_path = r"C:\Users\Sanskruti\Downloads\Waste Segregation"

# Image data generators for preprocessing
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)


# Load training and validation data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),    
    batch_size=32,
    class_mode='categorical',   
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),  # Added more filters
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),  # Increased dense layer neurons
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),loss=' categorical_crossentropy',metrics=['accuracy'])

#Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# Train the model
history = model.fit(
    train_data,
    epochs=18,
    validation_data=val_data,
    callbacks=[early_stopping]
)

# Save the model
model.save("waste_segregation_model.h5")

# Evaluate the model
loss, accuracy = model.evaluate(val_data)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

#Predicting New Images**
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict_image(image_path):
    img = load_img(image_path, target_size=(128, 128))  
    img_array = img_to_array(img) / 255.0              
    img_array = np.expand_dims(img_array, axis=0)      
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)            
    confidence = np.max(prediction)                    
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")

# Plot training and validation accuracy
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
