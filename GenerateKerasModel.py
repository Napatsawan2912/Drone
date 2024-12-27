import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Data Preprocessing

# Define your image data directory paths
train_data_dir = "dataset/hand_getsure/train" 
validation_data_dir = "dataset/hand_getsure/valid"

# Set image size
img_width, img_height = 300, 300 
num_classes = 8

# Create ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Generate batches of image data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax') # Replace 'num_classes' with your number of classes
])

# 3. Compile the Model

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 4. Train the Model
epochs = 8

model.fit(
    train_generator,
    steps_per_epoch=2000 // 32, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=800 // 32 
)

# 5. Evaluate the Model (Using `evaluate` instead of `evaluate_generator`)

loss, accuracy = model.evaluate(validation_generator, steps=800 // 32)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# 6. Save the Model

model.save('models/hand_gesture_control_8a.keras')
