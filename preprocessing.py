from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = "E:\dlproject\face\fer2013\train"  # e.g., "E:/dlproject/face_dataset/train"
test_path = "E:\dlproject\face\fer2013\test"

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True, 
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path, target_size=(48, 48), color_mode="grayscale",
    batch_size=64, class_mode="categorical", subset="training"
)
validation_generator = train_datagen.flow_from_directory(
    train_path, target_size=(48, 48), color_mode="grayscale",
    batch_size=64, class_mode="categorical", subset="validation"
)
test_generator = test_datagen.flow_from_directory(
    test_path, target_size=(48, 48), color_mode="grayscale",
    batch_size=64, class_mode="categorical"
)