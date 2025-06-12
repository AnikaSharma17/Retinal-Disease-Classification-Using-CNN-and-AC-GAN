import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.metrics import classification_report, roc_auc_score

# 1. Define Dataset Paths and Parameters
train_data_dir = r"C:\Users\Vansh\OneDrive\Desktop\AC-GAN-Retinal-Disease\dataset\train" 
validation_data_dir = r"C:\Users\Vansh\OneDrive\Desktop\AC-GAN-Retinal-Disease\dataset\val" 

img_height, img_width = 128, 128
batch_size = 32
target_classes = ['Hypertension', 'Cholesterol', 'Heart Disease']  
num_classes = 5  
seed = 42

# Function for histogram equalization
def enhance_image(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y_channel = img_yuv[:,:,0]
    if y_channel.dtype != np.uint8:
        y_channel = (y_channel * 255).astype(np.uint8)
    equalized_y = cv2.equalizeHist(y_channel)
    img_yuv[:,:,0] = equalized_y
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_output

# Custom data generator with image enhancement
class EnhancedImageDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, target_size=(256, 256),
                             color_mode='rgb', classes=None, class_mode='categorical',
                             batch_size=32, shuffle=True, seed=None,
                             save_to_dir=None, save_prefix='', save_format='png',
                             follow_links=False, subset=None, interpolation='nearest'):
        return EnhancedDirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format,
            follow_links=follow_links, subset=subset, interpolation=interpolation
        )

class EnhancedDirectoryIterator(tf.keras.preprocessing.image.DirectoryIterator):
    def _get_batches_of_transformed_samples(self, index_array):
        batch = super()._get_batches_of_transformed_samples(index_array)
        enhanced_batch_x = np.array([enhance_image(x) for x in batch[0]])
        return (enhanced_batch_x, batch[1])

# 2. Data Preprocessing and Augmentation
train_datagen = EnhancedImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    classes=target_classes,
    shuffle=True,
    seed=seed
)

val_datagen = EnhancedImageDataGenerator(rescale=1./255)
validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    classes=target_classes,
    shuffle=False,
    seed=seed
)

class_names = list(train_generator.class_indices.keys())
print("Class Names:", class_names)

# 4. Build the CNN Model (slightly larger to handle more classes)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'), 
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),  
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  
])

# 5. Compile the Model with metrics
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.AUC(name='auc')])

# 6. Train the Model 
epochs = 15
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# ===== SIMPLIFIED LEARNING CURVES =====
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# 7. Print final metrics
print("\nFinal Training Metrics:")
print(f"Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Precision: {history.history['precision'][-1]:.4f}")
print(f"Recall: {history.history['recall'][-1]:.4f}")
print(f"AUC: {history.history['auc'][-1]:.4f}")

print("\nFinal Validation Metrics:")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Validation Precision: {history.history['val_precision'][-1]:.4f}")
print(f"Validation Recall: {history.history['val_recall'][-1]:.4f}")
print(f"Validation AUC: {history.history['val_auc'][-1]:.4f}")

# 8. Generate classification report and AUC scores
validation_generator.reset()
Y_val = validation_generator.classes
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(Y_val, y_pred, target_names=class_names))

print("\nAUC Scores:")
for i, class_name in enumerate(class_names):
    auc_score = roc_auc_score((Y_val == i).astype(int), Y_pred[:, i])
    print(f"{class_name}: {auc_score:.4f}")

# 9. Save the Trained Model 
model.save('C3_HypvsCHvsHD.h5')
print("Model saved as C3_HypvsCHvsHD.h5")