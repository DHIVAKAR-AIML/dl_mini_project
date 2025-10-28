import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Paths (adjust if needed for your environment)
train_path = "C:/face/fer2013/train"
test_path = "C:/face/fer2013/test"

# Data generators (added validation split for better evaluation)
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True, 
    validation_split=0.2  # Use part of train for validation
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path, 
    target_size=(48, 48), 
    color_mode="grayscale",
    batch_size=64, 
    class_mode="categorical", 
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    train_path, 
    target_size=(48, 48), 
    color_mode="grayscale",
    batch_size=64, 
    class_mode="categorical", 
    subset="validation"
)

test_generator = test_datagen.flow_from_directory(
    test_path, 
    target_size=(48, 48), 
    color_mode="grayscale",
    batch_size=64, 
    class_mode="categorical"
)

# CNN model (minor tweaks for robustness)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(), 
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(), 
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(), 
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for better training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("best_emotion_model.keras", save_best_only=True, monitor='val_accuracy')

# Train (using validation_generator instead of test for monitoring)
history = model.fit(
    train_generator, 
    epochs=30, 
    validation_data=validation_generator, 
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Save final model (using .keras format, as .h5 is deprecated)
model.save("emotion_model.keras")
print("✅ Model trained and saved")
