# 1. Required Imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import numpy as np
import cv2
import os

# 2. Define Paths and Parameters
test_data_dir = r"C:\Users\Vansh\OneDrive\Desktop\AC-GAN-Retinal-Disease\dataset\test"
img_height, img_width = 128, 128
batch_size = 32
target_classes = ['Cholesterol', 'Control', 'DM']
seed = 42

# 3. Enhancement Function (Histogram Equalization)
def enhance_image(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y_channel = img_yuv[:,:,0]
    if y_channel.dtype != np.uint8:
        y_channel = (y_channel * 255).astype(np.uint8)
    img_yuv[:,:,0] = cv2.equalizeHist(y_channel)
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

# 4. Custom Enhanced Data Generator Classes
class EnhancedImageDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, **kwargs):
        return EnhancedDirectoryIterator(directory, self, **kwargs)

class EnhancedDirectoryIterator(DirectoryIterator):
    def _get_batches_of_transformed_samples(self, index_array):
        batch = super()._get_batches_of_transformed_samples(index_array)
        batch_x = np.array([enhance_image(x) for x in batch[0]])
        return (batch_x, batch[1])

# 5. Create Test Generator
test_datagen = EnhancedImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    classes=target_classes,
    shuffle=False,
    seed=seed
)

# 6. Evaluate Model
# NOTE: This assumes 'model' is already trained and exists in memory
loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
