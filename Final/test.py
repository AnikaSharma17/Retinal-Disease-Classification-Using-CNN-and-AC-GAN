import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# --------- Parameters ---------
test_data_dir = r"C:\Users\Vansh\OneDrive\Desktop\AC-GAN-Retinal-Disease\dataset\test"
img_height, img_width = 128, 128
batch_size = 16
target_classes = ['Hypertension', 'Cholesterol', 'Heart Disease']
num_classes = len(target_classes)
seed = 42

# --------- Histogram Equalization ---------
def enhance_image(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y_channel = img_yuv[:, :, 0]
    if y_channel.dtype != np.uint8:
        y_channel = (y_channel * 255).astype(np.uint8)
    equalized_y = cv2.equalizeHist(y_channel)
    img_yuv[:, :, 0] = equalized_y
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_output

# --------- Enhanced Generator ---------
class EnhancedImageDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, **kwargs):
        return EnhancedDirectoryIterator(directory, self, **kwargs)

class EnhancedDirectoryIterator(tf.keras.preprocessing.image.DirectoryIterator):
    def _get_batches_of_transformed_samples(self, index_array):
        batch = super()._get_batches_of_transformed_samples(index_array)
        enhanced_batch_x = np.array([enhance_image(x) for x in batch[0]])
        return (enhanced_batch_x, batch[1])

# --------- Load Model ---------
model_path = r"C:\Users\Vansh\OneDrive\Desktop\AC-GAN-Retinal-Disease\Final\C3_HypvsCHvsHD.h5"
model = load_model(model_path)
print("✅ Model loaded.")

# --------- Test Generator ---------
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

# --------- Evaluate Model ---------
results = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f"\nTest Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")

# --------- Precision, Recall, AUC ---------
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator, steps=np.ceil(test_generator.samples / batch_size))
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=target_classes))

# AUC
if num_classes == 2:
    auc = tf.keras.metrics.AUC()
    auc.update_state(y_true, y_pred_probs[:, 1])
    print(f"AUC: {auc.result().numpy():.4f}")
else:
    auc = tf.keras.metrics.AUC(multi_label=True, num_labels=num_classes)
    y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
    auc.update_state(y_true_one_hot, y_pred_probs)
    print(f"Multi-Class AUC: {auc.result().numpy():.4f}")
