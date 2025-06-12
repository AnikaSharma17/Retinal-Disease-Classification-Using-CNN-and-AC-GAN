#def define_discriminator(in_shape=(64, 64, 3), n_classes=5):
#    # Weight initialization
#    init = RandomNormal(stddev=0.02)
#    
#    in_image = Input(shape=in_shape)
#
#    # Convolutional Layer 1
#    fe = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_regularizer=l2(0.01))(in_image)
#    fe = BatchNormalization()(fe)
#    fe = LeakyReLU(alpha=0.2)(fe)
#    fe = Dropout(0.5)(fe)
#    
#   
#    
#    # Convolutional Layer 2
#    fe = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_regularizer=l2(0.01))(fe)
#    fe = BatchNormalization()(fe)
#    fe = LeakyReLU(alpha=0.2)(fe)
#    fe = Dropout(0.5)(fe)
#    
#    # Convolutional Layer 3
#    fe = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_regularizer=l2(0.01))(fe)
#    fe = BatchNormalization()(fe)
#    fe = LeakyReLU(alpha=0.2)(fe)
#    fe = Dropout(0.5)(fe)
#    
#    # Convolutional Layer 4
#    fe = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_regularizer=l2(0.01))(fe)
#    fe = BatchNormalization()(fe)
#    fe = LeakyReLU(alpha=0.2)(fe)
#    fe = Dropout(0.5)(fe)
#    
#   #  Flatten feature maps
#    fe = Flatten()(fe)
#    
#    # Real/fake output
#    out1 = Dense(1, activation='sigmoid')(fe)
#    
#     #Class label output
#    out2 = Dense(n_classes, activation='softmax')(fe)
#    
#     #Define model
#    model = Model(in_image, [out1, out2])
#    
#    # Compile model
#    opt = Adam(learning_rate=0.0002, beta_1=0.5)
#    
#    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
#    
#    return model
#
#def define_generator(latent_dim, n_classes=5):
#    # Weight initialization
#    init = RandomNormal(stddev=0.02)
#    
#    # Label input
#    in_label = Input(shape=(1,))
#     #Embedding for categorical input
#    li = Embedding(n_classes, 50)(in_label)
#     #Linear multiplication
#    n_nodes = 4 * 4
#    li = Dense(n_nodes, kernel_regularizer=l2(0.01))(li)
#    # Reshape to additional channel
#    li = Reshape((4, 4, 1))(li)
#    
#    # Image generator input
#    in_lat = Input(shape=(latent_dim,))
#    # Foundation for 4x4 image
#    n_nodes = 256 * 4 * 4
#    gen = Dense(n_nodes, kernel_regularizer=l2(0.01))(in_lat)
#    gen = Activation('relu')(gen)
#    gen = Reshape((4, 4, 256))(gen)
#    #kernel_initializer=init
#    
#     #Merge image gen and label input
#    merge = Concatenate()([gen, li])
#    
#     #Upsample to 8x8
#    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_regularizer=l2(0.01))(merge)
#    gen = BatchNormalization()(gen)
#    gen = LeakyReLU(alpha=0.2)(gen)
#    
#     #Upsample to 16x16
#    gen = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_regularizer=l2(0.01))(gen)
#    gen = BatchNormalization()(gen)
#    gen = LeakyReLU(alpha=0.2)(gen)
#    
#     #Upsample to 32x32
#    gen = Conv2DTranspose(32, (4,4), strides=(2,2), padding='same', kernel_regularizer=l2(0.01))(gen)
#    gen = BatchNormalization()(gen)
#    gen = LeakyReLU(alpha=0.2)(gen)
#    
#     #Upsample to 64x64
#    gen = Conv2DTranspose(16, (4,4), strides=(2,2), padding='same', kernel_regularizer=l2(0.01))(gen)
#    gen = BatchNormalization()(gen)
#    gen = LeakyReLU(alpha=0.2)(gen)
#    
#     #Output 64x64 RGB image
#    gen = Conv2DTranspose(3, (4,4), strides=(1,1), padding='same', kernel_regularizer=l2(0.01))(gen)
#    out_layer = Activation('tanh')(gen)
#    
#     #Define model
#    model = Model([in_lat, in_label], out_layer)
#    return model
import tensorflow as tf
import io
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import ReLU
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
import pandas as pd
from keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Dropout, Flatten, Dense, GlobalMaxPooling2D
#from tensorflow.keras.optimizers import LeakyReLU
#from keras.optimizers import Adam
#from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.models import load_model
import numpy as np
#from keras.utils.visualize_utils import plot
from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
#from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Concatenate
from keras.initializers import RandomNormal
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import tqdm
from numba import jit, cuda
from numpy import expand_dims
from tensorflow.keras.utils import plot_model
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#from keras.optimizers import Adam
from keras.models import Model
from tensorflow.keras.applications import VGG16,InceptionV3, ResNet50,Xception,VGG19,InceptionResNetV2, EfficientNetB1
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Activation
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import textwrap
import io
from matplotlib import pyplot
from keras import backend
from numpy import expand_dims
from keras.models import load_model
from keras.datasets.mnist import load_data
import glob
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
#cat= glob.glob('newcatdog/train/0/*.*')
#dog= glob.glob('newcatdog/train/1/*.*')
#96%
HR= glob.glob('drhr/HR/*.*')
D1= glob.glob('drhr/D1/*.*')
D2= glob.glob('drhr/D2/*.*')
D3= glob.glob('drhr/D3/*.*')
G=glob.glob('drhr/glaucoma/*.*')
N=glob.glob('drhr/Normal/*.*')

#94%
Glaucomatous= glob.glob('scans/Glaucomatous eye images/*.*')
Normal_eye= glob.glob('scans/Normal eye/*.*')

#Glaucomatous= glob.glob('category/*.*')
#Normal_eye= glob.glob('scans/Normal eye/*.*')

#drishti
Glaucoma_d= glob.glob('Training/glaucoma/*.*')
Normal_d= glob.glob('Training/normal/*.*')


#APtOS
No_DR= glob.glob('colored_images/c/N0_DR/*.*')
Mild= glob.glob('colored_images/c/Mild/*.*')
Moderate= glob.glob('colored_images/c/Moderate/*.*')
Severe= glob.glob('colored_images/c/Severe/*.*')
Proliferate_DR= glob.glob('colored_images/c/Proliferate_DR/*.*')

# ODIR dataset
D11= glob.glob('disease1/D/*.*')
Normal1= glob.glob('disease1/N/*.*')
Glaucoma1= glob.glob('disease1/G/*.*')
Hypertension1= glob.glob('disease1/H/*.*')
AD1= glob.glob('disease1/A/*.*')
others= glob.glob('disease1/O/*.*')
myopia= glob.glob('disease1/M/*.*')
cataract= glob.glob('disease1/C/*.*')
others2=Hypertension1+others+myopia+cataract

I=glob.glob('eyepac1/0/*.*')
II=glob.glob('eyepac1/1/*.*')
III=glob.glob('eyepac1/2/*.*')
IV=glob.glob('eyepac1/3/*.*')
V=glob.glob('eyepac1/4/*.*')

#  89%
normal = glob.glob('dataset/normal/*.*')
glaucoma = glob.glob('dataset/glaucoma/*.*')
DR = glob.glob('dataset/diabetic_retinopathy/*.*')
#cataract = glob.glob('dataset/cataract/*.*')

#doctor dataset  85%
Hypertension= glob.glob('Monika/Hypertension/*.*')
Heart_disease= glob.glob('Monika/Heart Disease/*.*')
DM= glob.glob('Monika/DM/*.*')
Control= glob.glob('Monika/Control/*.*')
Cholesterol= glob.glob('Monika/Cholesterol/*.*')

NN=glob.glob('HRF/healthy/*.*')
DD=glob.glob('HRF/diabetic_retinopathy/*.*')
GG=glob.glob('HRF/glaucoma/*.*')

data1=[]
labels1=[]
data = []
labels = []
filenames=[]
#img=tf.keras.preprocessing.image.load_img('Monika/Cholesterol/1.jpg',color_mode='rgb', target_size= (300,300))
#print(type(img))
#print(img.format)
#print(img.mode)
#print(img.size)
#
# 
## show the image
#img.show()
#for i in dataaug:   
#    image=tf.keras.preprocessing.image.load_img(i, color_mode='grayscale', 
#    target_size= (300,300))
#    image=np.array(image)
#    data1.append(image)
#    labels1.append(9)
for i in normal[0:500]:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (128,128))
    image=np.array(image)
    data.append(image)
    labels.append(0)
    filenames.append(i)
for i in glaucoma[0:500]:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (128,128))
    image=np.array(image)
    data.append(image)
    labels.append(1)
    filenames.append(i)
for i in DR[0:500]:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (128,128))
    image=np.array(image)
    data.append(image)
    labels.append(2)
    filenames.append(i)
#for i in cataract[0:500]:   
#    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
#    target_size= (64,64))
#    image=np.array(image)
#    data.append(image)
#    labels.append(3)
#    filenames.append(i)
#for i in III[0:1500]:   
#    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
#    target_size= (224,224))
#    image=np.array(image)
#    data.append(image)
#    labels.append(2)
#    filenames.append(i)
#for i in IV[0:1500]:   
#    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
#    target_size= (224,224))
#    image=np.array(image)
#    data.append(image)
#    labels.append(3)
#    filenames.append(i)
#for i in V[0:1500]:   
#    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
#    target_size= (224,224))
#    image=np.array(image)
#    data.append(image)
#    labels.append(4)
#    filenames.append(i) 
#for i in DM[0:800]:   
#    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
#    target_size= (64,64))
#    image=np.array(image)
#    data.append(image)
#    labels.append(5)
#    filenames.append(i) 
#for i in Control[0:800]:   
#    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
#    target_size= (64,64))
#    image=np.array(image)
#    data.append(image)
#    labels.append(1)
#    filenames.append(i) 
#for i in Cholesterol[0:800]:   
#    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
#    target_size= (64,64))
#    image=np.array(image)
#    data.append(image)
#    labels.append(3)
#    filenames.append(i)  
print(len(data))  
data = np.array(data)
labels = np.array(labels)
filenames = np.array(filenames)
#data1 = np.array(data1)
#labels1 = np.array(labels1)
#
from sklearn.model_selection import train_test_split
trainX, testX, trainy, testy, train_filenames, test_filenames = train_test_split(data, labels,filenames, test_size=0.2,
                                                random_state=42)
print(trainX.shape, trainy.shape)
print(testX.shape, testy.shape)
print(train_filenames[0])
model = load_model('discriminator_0183.h5')
 #load the dataset

#expand to 3d, e.g. add channels
trainX = expand_dims(trainX, axis=-1)
testX = expand_dims(testX, axis=-1)
 #convert from ints to floats
trainX = trainX.astype('float32')
testX = testX.astype('float32')
# scale from [0,255] to [-1,1]
trainX = (trainX - 127.5) / 127.5
testX = (testX - 127.5) / 127.5

 #Multiclass
predictions = model.predict(testX, batch_size=64)
y_pred_probs = predictions[1]  # Assuming this is the output of the model

y_pred_labels = np.argmax(y_pred_probs, axis=1)
y_true_labels = testy
accuracy = np.mean(y_pred_labels == y_true_labels)
print('Test Accuracy: %.3f%%' % (accuracy * 100))

predictions1 = model.predict(trainX, batch_size=64)
y_pred_labels_train = np.argmax(predictions1[1], axis=1)
y_true_labels_train = trainy
accuracy1 = np.mean(y_pred_labels_train == y_true_labels_train)
print('Train Accuracy: %.3f%%' % (accuracy1 * 100))

precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
auc1 = roc_auc_score(y_true_labels, y_pred_probs, multi_class='ovr')
#, multi_class='ovr'

print('Precision: %.3f' % precision)
print('Recall: %.3f' % recall)
print('AUC: %.3f' % auc1)

## Compute confusion matrix
#conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
#n_classes = conf_matrix.shape[0]
#
## Calculate sensitivity (recall for the positive class) and specificity for each class
#sensitivities = []
#specificities = []
#
#n_classes = conf_matrix.shape[0]
#
#for i in range(n_classes):
#    TP = conf_matrix[i, i]
#    FN = conf_matrix[i, :].sum() - TP
#    FP = conf_matrix[:, i].sum() - TP
#    TN = conf_matrix.sum() - (TP + FP + FN)
#    
#    # Sensitivity (Recall) for class i
#    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
#    sensitivities.append(sensitivity)
#    
#    # Specificity for class i
#    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
#    specificities.append(specificity)
#
## Average specificity across all classes
#average_specificity = np.mean(specificities)
#
#print('Sensitivities for each class:', sensitivities)
#print('Average Specificity:', average_specificity)
#
#
#
#
#
#df = pd.DataFrame({
#    'Image Name': test_filenames,
#    'Predicted Label': y_pred_labels,  # Make sure y_pred_labels is defined
#    'True Label': testy
#})
#
#df.to_excel('predictions.xlsx', index=False, engine='openpyxl')
#
#print('Predictions saved to predictions.xlsx')
#
#predictions = model.predict(testX, batch_size=64)
#y_pred_probs = predictions[1]  # Assuming this is the output of the model
#y_true_labels = testy
#
## Convert labels to one-hot encoded format
#y_true_binarized = label_binarize(y_true_labels, classes=np.arange(y_pred_probs.shape[1]))
#
#plt.figure()
#for i in range(n_classes):
#    fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_probs[:, i])
#    roc_auc = auc(fpr, tpr)
#    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
#
## Plot the diagonal line (no skill classifier)
#plt.plot([0, 1], [0, 1], 'k--', lw=2)
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC) for Multi-Class')
#plt.legend(loc="lower right")
#plt.show()
#
#
#
## Binary
#
#
#
#
#
#
##
### for binary classification
##predictions = model.predict(testX, batch_size=64)
##y_pred_probs = predictions[1]  # Assuming this is the output of the model
##y_pred_probs_positive_class = y_pred_probs[:, 1]
##y_pred_labels = np.argmax(y_pred_probs, axis=1)
##y_true_labels = testy
##accuracy = np.mean(y_pred_labels == y_true_labels)
##print('Test Accuracy: %.3f%%' % (accuracy * 100))
##
##predictions1 = model.predict(trainX, batch_size=64)
##y_pred_labels_train = np.argmax(predictions1[1], axis=1)
##y_true_labels_train = trainy
##accuracy1 = np.mean(y_pred_labels_train == y_true_labels_train)
##print('Train Accuracy: %.3f%%' % (accuracy1 * 100))
##
##precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
##recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
##auc1 = roc_auc_score(y_true_labels, y_pred_probs_positive_class)
###, multi_class='ovr'
##
##print('Precision: %.3f' % precision)
##print('Recall: %.3f' % recall)
##print('AUC: %.3f' % auc1)
##
### Compute confusion matrix
##conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
##n_classes = conf_matrix.shape[0]
##
### Calculate sensitivity (recall for the positive class) and specificity for each class
##sensitivity = []
##specificity = []
##
##for i in range(n_classes):
##    TP = conf_matrix[i, i]
##    FN = conf_matrix[i, :].sum() - TP
##    FP = conf_matrix[:, i].sum() - TP
##    TN = conf_matrix.sum() - (TP + FP + FN)
##    
##    sensitivity.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
##    specificity.append(TN / (TN + FP) if (TN + FP) > 0 else 0)
##
##print('Sensitivity per class:', sensitivity)
##print('Specificity per class:', specificity)
##
##TN, FP, FN, TP = conf_matrix.ravel()
##
### Calculate overall sensitivity (recall) and specificity
##sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
##specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
##
##print(f'Overall Sensitivity (Recall): {sensitivity:.3f}')
##print(f'Overall Specificity: {specificity:.3f}')
##
##
##
##
##
##df = pd.DataFrame({
##    'Image Name': test_filenames,
##    'Predicted Label': y_pred_labels,  # Make sure y_pred_labels is defined
##    'True Label': testy
##})
##
##df.to_excel('predictions.xlsx', index=False, engine='openpyxl')
##
##print('Predictions saved to predictions.xlsx')
##
##
##
### Compute predictions
##
### Predict probabilities for the positive class
##predictions = model.predict(testX, batch_size=64)
##y_pred_probs = predictions[1]  # Assuming this is the output of the model
##y_true_labels = testy
##
### Extract probabilities for the positive class
##y_pred_probs_positive_class = y_pred_probs[:, 1]
##
### Compute ROC curve
##fpr, tpr, _ = roc_curve(y_true_labels, y_pred_probs_positive_class)
##roc_auc = auc(fpr, tpr)
##
### Plot ROC curve
##plt.figure()
##plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
##
### Plot the diagonal line (no skill classifier)
##plt.plot([0, 1], [0, 1], 'k--', lw=2)
##plt.xlim([0.0, 1.0])
##plt.ylim([0.0, 1.05])
##plt.xlabel('False Positive Rate')
##plt.ylabel('True Positive Rate')
##plt.title('Receiver Operating Characteristic (ROC) for Binary Classification')
##plt.legend(loc="lower right")
##plt.show()
##
##
##
### Load models
###import matplotlib.pyplot as plt
###import pandas as pd
###import numpy as np
###from tensorflow.keras.models import load_model
###
#### Load models
###generator = load_model('model_0499.h5')
###discriminator = load_model('discriminator_0499.h5')
###
#### Function to extract model summary information
###def get_model_summary(model):
###    summary_list = []
###    for layer in model.layers:
###        layer_type = layer.__class__.__name__
###        output_shape = str(layer.output_shape)
###        params = layer.count_params()
###        summary_list.append([layer.name, layer_type, params, output_shape])
###    return summary_list
###
#### Extract summaries
###generator_summary = get_model_summary(generator)
###discriminator_summary = get_model_summary(discriminator)
###
#### Create DataFrames
###generator_df = pd.DataFrame(generator_summary, columns=['Layer Name', 'Layer Type', 'Parameters', 'Output Shape'])
###discriminator_df = pd.DataFrame(discriminator_summary, columns=['Layer Name', 'Layer Type', 'Parameters', 'Output Shape'])
###
#### Plot and save the generator model summary
###fig, ax = plt.subplots(figsize=(17, 17))  # Increase figsize for better visibility
###ax.axis('tight')
###ax.axis('off')
###table_data = generator_df.values
###table = plt.table(cellText=table_data, colLabels=generator_df.columns, cellLoc='center', loc='center', rowColours=['#f5f5f5']*len(table_data), colColours=['#d9d9d9']*len(generator_df.columns))
###table.auto_set_font_size(False)
###table.set_fontsize(19)  # Increase font size for better readability
###table.scale(1, 3)  # Increase row height (scaling factor for row height)
###plt.title('Generator Model Summary', fontsize=16)
###plt.savefig('generator_summary.png', bbox_inches='tight')
###
#### Plot and save the discriminator model summary
###fig, ax = plt.subplots(figsize=(17, 17))  # Increase figsize for better visibility
###ax.axis('tight')
###ax.axis('off')
###table_data = discriminator_df.values
###table = plt.table(cellText=table_data, colLabels=discriminator_df.columns, cellLoc='center', loc='center', rowColours=['#f5f5f5']*len(table_data), colColours=['#d9d9d9']*len(discriminator_df.columns))
###table.auto_set_font_size(False)
###table.set_fontsize(19)  # Increase font size for better readability
###table.scale(1, 3)  # Increase row height (scaling factor for row height)
###plt.title('Discriminator Model Summary', fontsize=16)
###plt.savefig('discriminator_summary.png', bbox_inches='tight')
###
###print("Summary images saved as 'generator_summary.png' and 'discriminator_summary.png'")
