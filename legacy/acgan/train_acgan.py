import tensorflow as tf
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
from sklearn.metrics import mean_squared_error
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
import tqdm
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from numpy.random import randint, randn 
from numpy import expand_dims, ones, zeros 
from keras.models import load_model
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
from matplotlib import pyplot
from keras import backend
from numpy import expand_dims
from keras.models import load_model
from keras.datasets.mnist import load_data
import glob
from sklearn.metrics import classification_report
#cat= glob.glob('newcatdog/train/0/*.*')
#dog= glob.glob('newcatdog/train/1/*.*')
#96%
HR= glob.glob('drhr/HR/*.*')
D1= glob.glob('drhr/D1/*.*')
D2= glob.glob('drhr/D2/*.*')
D3= glob.glob('drhr/D3/*.*')
G=glob.glob('drhr/glaucoma/*.*')
N=glob.glob('drhr/Normal/*.*')

#94%AI
Glaucomatous= glob.glob('scans/Glaucomatous eye images/*.*')
Normal_eye= glob.glob('scans/Normal eye/*.*')

#drishti

Glaucoma_d= glob.glob('Training/glaucoma/*.*')
Normal_d= glob.glob('Training/normal/*.*')

#Glaucomatous= glob.glob('category/*.*')
#Normal_eye= glob.glob('scans/Normal eye/*.*')


#APtOS
No_DR= glob.glob('colored_images/c/N0_DR/*.*')
Mild= glob.glob('colored_images/c/Mild/*.*')
Moderate= glob.glob('colored_images/c/Moderate/*.*')
Severe= glob.glob('colored_images/c/Severe/*.*')
Proliferate_DR= glob.glob('colored_images/c/Proliferate_DR/*.*')

#messidor
No_DR1= glob.glob('aptos_new/0/*.*')
Mild1= glob.glob('aptos_new/1/*.*')
Moderate1= glob.glob('aptos_new/2/*.*')
Severe1= glob.glob('aptos_new/3/*.*')
Proliferate_DR1= glob.glob('aptos_new/4/*.*')

#eyepac
I=glob.glob('eyepac1/0/*.*')
II=glob.glob('eyepac1/1/*.*')
III=glob.glob('eyepac1/2/*.*')
IV=glob.glob('eyepac1/3/*.*')
V=glob.glob('eyepac1/4/*.*')

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


#  89%kaggle(nor/cataract/dr/glaucoma)
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



#HRF
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
for i in Glaucomatous[0:500]:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (64,64))
    image=np.array(image)
    data.append(image)
    labels.append(0)
    filenames.append(i)
for i in Normal_eye[0:500]:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (64,64))
    image=np.array(image)
    data.append(image)
    labels.append(1)
    filenames.append(i)
#for i in DM[0:500]:   
#    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
#    target_size= (64,64))
#    image=np.array(image)
#    data.append(image)
#    labels.append(2)
#    filenames.append(i)
#for i in Control[0:500]:   
#    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
#    target_size= (64,64))
#    image=np.array(image)
#    data.append(image)
#    labels.append(3)
#    filenames.append(i)
#for i in Control[0:1000]:   
#    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
#    target_size= (64,64))
#    image=np.array(image)
#    data.append(image)
#    labels.append(4)
#    filenames.append(i)
##for i in DM[0:800]:   
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
from tensorflow.keras.regularizers import l2 
from keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Reshape, Concatenate, Conv2DTranspose, BatchNormalization, Activation
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.applications import VGG16,ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import GlobalAveragePooling2D
from matplotlib import pyplot
from keras.models import Sequential
import numpy as np

 #Define the standalone discriminator model




#
#
#
# Number of classes is taken from the loaded labels so the generator, the discriminator's
# auxiliary classifier and the sampled fake labels always agree.
N_CLASSES = int(labels.max()) + 1


def define_discriminator(in_shape=(64, 64, 3), n_classes=N_CLASSES):
    # Weight initialization
    init = RandomNormal(stddev=0.02)
    
    in_image = Input(shape=in_shape)

    # Convolutional Layer 1
    fe = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    
   
    
    # Convolutional Layer 2
    fe = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    
    # Convolutional Layer 3
    fe = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    
    # Convolutional Layer 4
    fe = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.5)(fe)
    
   #  Flatten feature maps
    fe = Flatten()(fe)
    
    # Real/fake output
    out1 = Dense(1, activation='sigmoid')(fe)
    
     #Class label output
    out2 = Dense(n_classes, activation='softmax')(fe)
    
     #Define model
    model = Model(in_image, [out1, out2])
    
    # Compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    
    return model

def define_generator(latent_dim, n_classes=N_CLASSES):
    # Weight initialization
    init = RandomNormal(stddev=0.02)
    
    # Label input
    in_label = Input(shape=(1,))
     #Embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
     #Linear multiplication
    n_nodes = 4 * 4
    li = Dense(n_nodes, kernel_initializer=init)(li)
    # Reshape to additional channel
    li = Reshape((4, 4, 1))(li)
    
    # Image generator input
    in_lat = Input(shape=(latent_dim,))
    # Foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
    gen = Activation('relu')(gen)
    gen = Reshape((4, 4, 256))(gen)
    #kernel_initializer=init
    
     #Merge image gen and label input
    merge = Concatenate()([gen, li])
    
     #Upsample to 8x8
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merge)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    
     #Upsample to 16x16
    gen = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    
     #Upsample to 32x32
    gen = Conv2DTranspose(32, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    
     #Upsample to 64x64
    gen = Conv2DTranspose(16, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    
     #Output 64x64 RGB image
    gen = Conv2DTranspose(3, (4,4), strides=(1,1), padding='same', kernel_initializer=init)(gen)
    out_layer = Activation('tanh')(gen)
    
     #Define model
    model = Model([in_lat, in_label], out_layer)
    return model


#def define_discriminator(in_shape=(128, 128, 3), n_classes=3):
#    #baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(128, 128, 3)))
#    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(128, 128, 3)))
##    
##    # Freeze the base model's layers
#    for layer in base_model.layers:
#        layer.trainable = False
#    
#    # Create the input layer
#    in_image = Input(shape=in_shape)
#    
#    # Use the InceptionV3 base model for feature extraction
#    x = base_model(in_image)
#    init = RandomNormal(stddev=0.02)
#
#    #in_image = Input(shape=in_shape)
#
#    # Convolutional Layer 1
#    fe = Conv2D(16, (3, 3), padding='same', kernel_initializer=init)(x)
#    fe = BatchNormalization()(fe)
#    fe = LeakyReLU(alpha=0.2)(fe)
#
#    # Convolutional Layer 2
#    fe = Conv2D(8, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(fe)
#    fe = BatchNormalization()(fe)
#    fe = LeakyReLU(alpha=0.2)(fe)
#
#    # Convolutional Layer 3
#    fe = Conv2D(16, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(fe)
#    fe = BatchNormalization()(fe)
#    fe = LeakyReLU(alpha=0.2)(fe)
#
#    # Convolutional Layer 4
#    fe = Conv2D(8, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(fe)
#    fe = BatchNormalization()(fe)
#    fe = LeakyReLU(alpha=0.2)(fe)
#
#    # Convolutional Layer 5
#    fe = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(fe)
#    fe = BatchNormalization()(fe)
#    fe = LeakyReLU(alpha=0.2)(fe)
#
#    # Flatten feature maps
#    fe = Flatten()(fe)
#    fe = Dropout(0.5)(fe)
#
#    # Real/fake output
#    out1 = Dense(1, activation='sigmoid')(fe)
#
#    # Class label output
#    out2 = Dense(n_classes, activation='softmax')(fe)
#
#    # Define model
#    model = Model(in_image, [out1, out2])
#
#    # Compile model
#    opt = Adam(learning_rate=0.0001,beta_1=0.5 )
#    #beta_1=0.5
#    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
#
#    return model
#
#
#
#
#
#
#
#
#
#def define_generator(latent_dim, n_classes=3):
#    init = RandomNormal(stddev=0.02)
#
#    # Label input
#    in_label = Input(shape=(1,))
#    # Embedding for categorical input
#    li = Embedding(n_classes, 50)(in_label)
#    # Linear multiplication
#    n_nodes = 16 * 16
#    li = Dense(n_nodes, kernel_initializer=init)(li)
#    # Reshape to additional channel
#    li = Reshape((16, 16, 1))(li)
#
#    # Image generator input
#    in_lat = Input(shape=(latent_dim,))
#    # Foundation for 16x16 image
#    n_nodes = 256 * 16 * 16
#    gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
#    gen = LeakyReLU(alpha=0.2)(gen)
#    gen = Reshape((16, 16, 256))(gen)
#
#    # Merge image gen and label input
#    merge = Concatenate()([gen, li])
#
#    # Upsample to 32x32
#    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merge)
#    gen = BatchNormalization()(gen)
#    gen = LeakyReLU(alpha=0.2)(gen)
#
#    # Upsample to 64x64
#    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(gen)
#    gen = BatchNormalization()(gen)
#    gen = LeakyReLU(alpha=0.2)(gen)
#
#    # Upsample to 128x128
#    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(gen)
#    gen = BatchNormalization()(gen)
#    gen = LeakyReLU(alpha=0.2)(gen)
#
#    # Output 128x128 RGB image
#    gen = Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(gen)
#    out_layer = Activation('tanh')(gen)
#
#    # Define model
#    model = Model([in_lat, in_label], out_layer)
#    return model



 #Define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
  #   Make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
   #  Connect the outputs of the generator to the inputs of the discriminator
    gan_output = d_model(g_model.output)
    # Define gan model as taking noise and label and outputting real/fake and label outputs
    model = Model(g_model.input, gan_output)
     #Compile model
    opt = Adam(learning_rate=0.0001, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model

 #Load images
def load_real_samples(trainX, trainy):
  #   Load dataset
    #(trainX, trainy), (_, _) = load_data()
   #  Expand to 3D, e.g., add channels
    # Convert from ints to floats
    X = trainX.astype('float32')
     #Scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    print(X.shape, trainy.shape)
    return [X, trainy]

 #Select real samples
def generate_real_samples(dataset, n_samples):
  #   Split into images and labels
    images, labels = dataset
   #  Choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # Select images and labels
    X, labels = images[ix], labels[ix]
     #Generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y

 #Generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=N_CLASSES):
  #   Generate points in the latent space
    x_input = randn(latent_dim * n_samples)
   #  Reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # Generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]

 #Use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples, n_classes=N_CLASSES):
  #   Generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples, n_classes)
   #  Predict outputs
    images = generator.predict([z_input, labels_input])
    # Create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y

 #Evaluate the discriminator's multiclass classification accuracy
def evaluate_discriminator_accuracy(d_model, dataset, batch_size):
    images, labels = dataset
  #   Generate predictions
    predictions = d_model.predict(images, batch_size=batch_size)
    y_pred_labels = np.argmax(predictions[1], axis=1)
    y_true_labels = labels
    
   #  Compute accuracy
    accuracy = np.mean(y_pred_labels == y_true_labels)
    return accuracy

 #Generate samples and save as a plot and save the model
#def summarize_performance(step, g_model, d_model, latent_dim, n_samples=500):
#  #   Prepare fake examples
#    [X, _], _ = generate_fake_samples(g_model, latent_dim, n_samples)
#   #  Scale from [-1,1] to [0,1]
#    X = (X + 1) / 2.0
#    # Plot images
#    for i in range(4):
#     #    Define subplot
#        pyplot.subplot(2, 2, 1 + i)
#      #   Turn off axis
#        pyplot.axis('off')
#       #  Plot raw pixel data
#        pyplot.imshow(X[i, :, :])
#     #Save plot to file
#    filename1 = 'generated_plot_%04d.png' % (step+1)
#    pyplot.savefig(filename1)
#    pyplot.close()
#     #Save the generator model
#    filename2 = 'model_%04d.h5' % (step+1)
#    g_model.save(filename2)
#     #Save the discriminator model
#    filename3 = 'discriminator_%04d.h5' % (step+1)
#    d_model.save(filename3)
#    print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))
def summarize_performance(step, g_model, d_model, latent_dim, real_samples, n_samples=500):
    # Prepare fake examples
    [X_fake, _], _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # Scale from [-1,1] to [0,1]
    X_fake = (X_fake + 1) / 2.0

    # Select real samples from the dataset
    X_real, _ = real_samples
    X_real = (X_real + 1) / 2.0  # Same scaling for real images

    # Calculate Mean Squared Error (MSE) between real and fake images
    mse = mean_squared_error(X_real.flatten(), X_fake.flatten())
    
    # Print the MSE to track the quality of generated images
    print(f"Step {step+1} - MSE between real and fake images: {mse:.4f}")
    
    # Plot images
    for i in range(4):
        # Define subplot
        pyplot.subplot(2, 2, 1 + i)
        # Turn off axis
        pyplot.axis('off')
        # Plot raw pixel data
        pyplot.imshow(X_fake[i, :, :])

    # Save plot to file
    filename1 = 'generated_plot_%04d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()

    # Save the generator model
    filename2 = 'model_%04d.h5' % (step+1)
    g_model.save(filename2)

    # Save the discriminator model
    filename3 = 'discriminator_%04d.h5' % (step+1)
    d_model.save(filename3)
    print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))

# Train the generator and discriminator
#def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=500, n_batch=64):
#  #   Calculate the number of batches per training epoch
#    bat_per_epo = int(dataset[0].shape[0] / n_batch)
#   #  Calculate the number of training iterations
#    n_steps = bat_per_epo * n_epochs
#    # Calculate the size of half a batch of samples
#    half_batch = int(n_batch / 2)
#     #Manually enumerate epochs
#    for i in range(n_steps):
#      #   Get randomly selected 'real' samples
#        [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
#       #  Update discriminator model weights
#        _, d_r1, d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
#        
#        # Generate 'fake' examples
#        [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
#         #Update discriminator model weights
#        _, d_f, d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
#        
#         #Prepare points in latent space as input for the generator
#        [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
#         #Create inverted labels for the fake samples
#        y_gan = ones((n_batch, 1))
#         #Update the generator via the discriminator's error
#        _, g_1, g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
#        
#         #Print loss for the batch
#        print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_r1, d_r2, d_f, d_f2, g_1, g_2))
#        
#         #Evaluate and print the discriminator's accuracy every epoch
#        if (i+1) % bat_per_epo == 0:
#            epoch = (i + 1) // bat_per_epo
#            accuracy = evaluate_discriminator_accuracy(d_model, dataset, n_batch)
#            print('Epoch: %d, Discriminator Accuracy: %.3f' % (epoch, accuracy))
#            summarize_performance(epoch, g_model, d_model, latent_dim)
#   
   
#   
##   
#   
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=500, n_batch=64):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    n_steps = bat_per_epo * n_epochs
    half_batch = int(n_batch / 2)
    
    # Lists to store metrics
    train_accuracies = []
    train_losses = []

    for i in range(n_steps):
        [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
        d_loss_real, d_r1, d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
        
        [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss_fake, d_f, d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
        
        [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
        y_gan = ones((n_batch, 1))
        g_loss, g_1, g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
        
        if (i+1) % bat_per_epo == 0:
            # Calculate and store metrics at the end of each epoch
            epoch = (i + 1) // bat_per_epo
            accuracy = evaluate_discriminator_accuracy(d_model, dataset, n_batch)
            train_accuracies.append(accuracy)
            train_losses.append(d_loss_real + d_loss_fake + g_loss)
            print('Epoch: %d, Discriminator Accuracy: %.3f' % (epoch, accuracy))
            summarize_performance(epoch, g_model, d_model, latent_dim, dataset, n_samples=500)

        # Print loss for the batch
        print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_r1, d_r2, d_f, d_f2, g_1, g_2))
#
#    plot_metrics(train_accuracies, train_losses, n_epochs)
#
#def plot_metrics(accuracies, losses, epochs):
#    epochs_range = range(1, len(accuracies) + 1)
#    
#    plt.figure(figsize=(12, 6))
#    
#    plt.subplot(1, 2, 1)
#    plt.plot(epochs_range, accuracies, label='Training Accuracy')
#    plt.xlabel('Epochs')
#    plt.ylabel('Accuracy')
#    plt.title('Training Accuracy vs. Epochs')
#    plt.legend()
#    
#    plt.subplot(1, 2, 2)
#    plt.plot(epochs_range, losses, label='Training Loss', color='orange')
#    plt.xlabel('Epochs')
#    plt.ylabel('Loss')
#    plt.title('Training Loss vs. Epochs')
#    plt.legend()
#    
#    plt.tight_layout()
#    plt.show()


#def load_real_samples(trainX, trainy):
#    X = np.expand_dims(trainX, axis=-1)  # Add channel dimension
#    X = X.astype('float32')
#    X = (X - 127.5) / 127.5  # Scale to [-1, 1]
#    return [X, trainy]
#
## Function to generate real samples
#def generate_real_samples(dataset, n_samples):
#    images, labels = dataset
#    ix = randint(0, images.shape[0], n_samples)
#    X, labels = images[ix], labels[ix]
#    y = np.ones((n_samples, 1))
#    return [X, labels], y
#
## Generate points in latent space as input for the generator
#def generate_latent_points(latent_dim, n_samples, n_classes=2):
#    x_input = randn(latent_dim * n_samples)
#    z_input = x_input.reshape(n_samples, latent_dim)
#    labels = randint(0, n_classes, n_samples)
#    return [z_input, labels]
#
## Generate fake samples
#def generate_fake_samples(generator, latent_dim, n_samples):
#    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
#    images = generator.predict([z_input, labels_input])
#    y = np.zeros((n_samples, 1))
#    return [images, labels_input], y
#
## Evaluate the discriminator's accuracy
#def evaluate_discriminator_accuracy(d_model, dataset, batch_size):
#    images, labels = dataset
#    predictions = d_model.predict(images, batch_size=batch_size)
#    y_pred_labels = np.argmax(predictions[1], axis=1)
#    y_true_labels = labels
#    accuracy = np.mean(y_pred_labels == y_true_labels)
#    return accuracy
#
## Evaluate the discriminator's loss on test set
#def evaluate_discriminator_loss(d_model, dataset, batch_size):
#    images, labels = dataset
#    loss = d_model.test_on_batch(images, [labels, labels])[0]
#    return loss
#
## Train the GAN model
#def train(g_model, d_model, gan_model, dataset, latent_dim, test_dataset, n_epochs=200, n_batch=64):
#    trainX, trainy = dataset
#    testX, testy = test_dataset
#    bat_per_epo = int(trainX.shape[0] / n_batch)
#    n_steps = bat_per_epo * n_epochs
#    half_batch = int(n_batch / 2)
#
#    train_accuracies = []
#    train_losses = []
#    test_accuracies = []
#    test_losses = []
#
#    for i in range(n_steps):
#        [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
#        d_loss_real, d_r1, d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
#
#        [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
#        d_loss_fake, d_f, d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
#
#        [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
#        y_gan = np.ones((n_batch, 1))
#        g_loss, g_1, g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
#
#        # Calculate training accuracy and loss
#        accuracy = evaluate_discriminator_accuracy(d_model, dataset, n_batch)
#        train_loss = d_loss_real + d_loss_fake + g_loss
#        train_accuracies.append(accuracy)
#        train_losses.append(train_loss)
#
#        # Calculate test accuracy and loss
#        test_accuracy = evaluate_discriminator_accuracy(d_model, test_dataset, n_batch)
#        test_loss = evaluate_discriminator_loss(d_model, test_dataset, n_batch)
#        test_accuracies.append(test_accuracy)
#        test_losses.append(test_loss)
#
#        # Print progress every batch
#        if (i + 1) % bat_per_epo == 0:
#            epoch = (i + 1) // bat_per_epo
#            print(f'Epoch: {epoch}, Discriminator Accuracy: {accuracy:.3f}, Test Accuracy: {test_accuracy:.3f}')
#            summarize_performance(epoch, g_model, d_model, latent_dim)
#
#        print(f'>{i+1}, dr[{d_r1:.3f},{d_r2:.3f}], df[{d_f:.3f},{d_f2:.3f}], g[{g_1:.3f},{g_2:.3f}]')
#
#    # Average metrics per epoch
#    avg_train_accuracy = np.mean(train_accuracies[-bat_per_epo:])
#    avg_train_loss = np.mean(train_losses[-bat_per_epo:])
#    avg_test_accuracy = np.mean(test_accuracies[-bat_per_epo:])
#    avg_test_loss = np.mean(test_losses[-bat_per_epo:])
#
#    # Plot metrics after all epochs are finished
#    plot_metrics(train_accuracies, train_losses, test_accuracies, test_losses, n_epochs)
#
#    print(f"Final Train Accuracy: {avg_train_accuracy:.3f}, Final Train Loss: {avg_train_loss:.3f}")
#    print(f"Final Test Accuracy: {avg_test_accuracy:.3f}, Final Test Loss: {avg_test_loss:.3f}")
#
#def plot_metrics(train_accuracies, train_losses, test_accuracies, test_losses, n_epochs):
#    # Calculate average per epoch
#    epochs_range = range(1, n_epochs + 1)
#    avg_train_accuracies = [np.mean(train_accuracies[i * len(train_accuracies) // n_epochs : (i+1) * len(train_accuracies) // n_epochs]) for i in range(n_epochs)]
#    avg_train_losses = [np.mean(train_losses[i * len(train_losses) // n_epochs : (i+1) * len(train_losses) // n_epochs]) for i in range(n_epochs)]
#    avg_test_accuracies = [np.mean(test_accuracies[i * len(test_accuracies) // n_epochs : (i+1) * len(test_accuracies) // n_epochs]) for i in range(n_epochs)]
#    avg_test_losses = [np.mean(test_losses[i * len(test_losses) // n_epochs : (i+1) * len(test_losses) // n_epochs]) for i in range(n_epochs)]
#
#    plt.figure(figsize=(12, 6))
#
#    # Plot Accuracy
#    plt.subplot(1, 2, 1)
#    plt.plot(epochs_range, avg_train_accuracies, label='Train Accuracy')
#    plt.plot(epochs_range, avg_test_accuracies, label='Test Accuracy')
#    plt.xlabel('Epochs')
#    plt.ylabel('Accuracy')
#    plt.title('Train/Test Accuracy vs Epochs')
#    plt.legend()
#
#    # Plot Loss
#    plt.subplot(1, 2, 2)
#    plt.plot(epochs_range, avg_train_losses, label='Train Loss', color='orange')
#    plt.plot(epochs_range, avg_test_losses, label='Test Loss', color='red')
#    plt.xlabel('Epochs')
#    plt.ylabel('Loss')
#    plt.title('Train/Test Loss vs Epochs')
#    plt.legend()
#
#    plt.tight_layout()
#    plt.show()
#
## Function to save generated samples and model performance
#def summarize_performance(step, g_model, d_model, latent_dim, n_samples=200):
#    [X, _], _ = generate_fake_samples(g_model, latent_dim, n_samples)
#    X = (X + 1) / 2.0
#    for i in range(4):
#        plt.subplot(2, 2, 1 + i)
#        plt.axis('off')
#        plt.imshow(X[i, :, :])
#    filename1 = f'generated_plot_{step+1:04d}.png'
#    plt.savefig(filename1)
#    plt.close()
#    filename2 = f'model_{step+1:04d}.h5'
#    g_model.save(filename2)
#    filename3 = f'discriminator_{step+1:04d}.h5'
#    d_model.save(filename3)
#    print(f'>Saved: {filename1}, {filename2}, {filename3}')

# Example for running the training
latent_dim = 100
discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan_model = define_gan(generator, discriminator)

# Use train and test data
train_data = load_real_samples(trainX, trainy)
test_data = load_real_samples(testX, testy)

# Train the model
train(generator, discriminator, gan_model, train_data, latent_dim, n_epochs=500)


#model = load_model('discriminator_0047.h5')
# #load the dataset
#
##expand to 3d, e.g. add channels
#trainX = expand_dims(trainX, axis=-1)
#testX = expand_dims(testX, axis=-1)
# #convert from ints to floats
#trainX = trainX.astype('float32')
#testX = testX.astype('float32')
## scale from [0,255] to [-1,1]
#trainX = (trainX - 127.5) / 127.5
#testX = (testX - 127.5) / 127.5
#predictions = model.predict(testX, batch_size=64)
#y_pred_labels = np.argmax(predictions[1], axis=1)
#y_true_labels = testy
#accuracy = np.mean(y_pred_labels == y_true_labels)
#print('Test Accuracy: %.3f%%' % (accuracy * 100))
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


