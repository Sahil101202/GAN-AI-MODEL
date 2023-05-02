
from google.colab import drive
import os
drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/CELEBa')

import numpy as np # linear algebra

from PIL import Image
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from keras import layers, models
from keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D,Dropout,Dense,Flatten,Conv2DTranspose,BatchNormalization,LeakyReLU,Reshape,MaxPooling2D

all_images_path=[]
full_image_train_path="/content/drive/MyDrive/CELEBa"

for path in os.listdir(full_image_train_path):
  if '.jpg' in path:
    all_images_path.append(os.path.join(full_image_train_path,path))
image_path=all_images_path[0:500]
len(image_path)

training_images = [np.array(Image.open(path)) for path in image_path]
#normalized = (x-min(x))/(max(x)-min(x))

for i in range(len(training_images)):
    training_images[i] = ((training_images[i] - training_images[i].min())/(255 - training_images[i].min()))
    
training_images = np.array(training_images) 

print(training_images.shape)

plt.figure(figsize=(40,10))
fig,ax=plt.subplots(2,5)
fig.suptitle("Real Images")
idx=8

for i in range(2):
    for j in range(5):
            ax[i,j].imshow(training_images[idx].reshape(64,64,3))            
            idx+=6
            
plt.tight_layout()
plt.show()

noise_shape = (100,)

#  Generator will upsample our seed using convolutional transpose layers (upsampling layers)



optimizer = Adam(0.00007, 0.5)

def generator_model(img_shape, noise_shape = (100,)):
    
    #noise_shape : the dimension of the input vector for the generator
    #img_shape   : the dimension of the output
    
    ## latent variable as input
    input_noise = layers.Input(shape=noise_shape) 
    #d = layers.Dense(1024, activation="relu")(input_noise) 
    d = layers.Dense(1024, activation="relu")(input_noise) 
    d = layers.Dense(128*8*8, activation="relu")(d)
    d = layers.Reshape((8,8,128))(d)
    
    d = layers.Conv2DTranspose(128, kernel_size=(2,2) ,  strides=(2,2) , use_bias=False)(d)
    d = layers.Conv2D( 64  , ( 1 , 1 ) , activation='relu' , padding='same', name="block_4")(d) ## 16,16,128


    d = layers.Conv2DTranspose(256, kernel_size=(2,2) ,  strides=(2,2) , use_bias=False)(d)
    d = layers.Conv2D( 64  , ( 1 , 1 ) , activation='relu' , padding='same', name="block_5")(d) ## 32,32,64
    
    if img_shape[0] == 64:
        d = layers.Conv2DTranspose(512, kernel_size=(2,2) ,  strides=(2,2) , use_bias=False)(d)
        d = layers.Conv2D( 64  , ( 1 , 1 ) , activation='relu' , padding='same', name="block_6")(d) ## 64,64,32
    
    img = layers.Conv2D( 3 , ( 1 , 1 ) , activation='sigmoid' , padding='same', name="final_block")(d) ## 64,64,3
    model = models.Model(input_noise, img)
    model.summary() 
    return(model)

## Set the dimension of latent variables to be 100
noise_shape = (100,)

generator = generator_model((64,64,3), noise_shape = noise_shape)

generator.compile(loss='binary_crossentropy', optimizer=optimizer)

def discriminator_model(img_shape,noutput=1):
    input_img = layers.Input(shape=img_shape)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_img)
    #x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    #x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    #x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(1, 1), name='block4_pool')(x)

    
    x         = layers.Flatten()(x)
    x         = layers.Dense(1024,      activation="relu")(x)
    out       = layers.Dense(noutput,   activation='sigmoid')(x)
    model     = models.Model(input_img, out)
    
    return model

discriminator  = discriminator_model((64,64,3))
discriminator.compile(loss      = 'binary_crossentropy', 
                      optimizer = optimizer,
                      metrics   = ['accuracy'])

discriminator.summary()

z = layers.Input(shape=noise_shape)
img = generator(z)


# For the combined model we will only train the generator
discriminator.trainable = False

# The valid takes generated images as input and determines validity
valid = discriminator(img)

# The combined model  (stacked generator and discriminator) takes
# noise as input => generates images => determines validity 
GAN = models.Model(z, valid)
GAN.compile(loss='binary_crossentropy', optimizer=optimizer)
GAN.summary()

epochs = 2000
batch_size = 64

def get_noise(nsample=1, nlatent_dim=100):
    noise = np.random.normal(0, 1, (nsample,nlatent_dim))
    return(noise)

def plot_generated_images(noise,path_save=None):
    imgs = generator.predict(noise)
    fig = plt.figure(figsize=(10,10))
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(2,5,i+1)
        ax.imshow(img)
    fig.suptitle("Generated images ",fontsize=30)
    
    #if path_save is not None:
    #    plt.savefig(path_save,
    #                bbox_inches='tight',
     #               pad_inches=0)
    #    plt.close()
    #else:
    plt.show()


noise_shape = (100,)
nsample = 10

noise = get_noise(nsample=nsample, nlatent_dim=noise_shape[0])



def train(models, X_train, noise_plot, epochs=10000, batch_size=128):
        #models     : tuple containins three tensors, (combined, discriminator, generator)
        #X_train    : np.array containing images (Nsample, height, width, Nchannels)
        #noise_plot : np.array of size (Nrandom_sample_to_plot, hidden unit length)
        #dir_result : the location where the generated plots for noise_plot are saved 
        
        combined, discriminator, generator = models
        nlatent_dim = noise_plot.shape[1]
        half_batch  = int(batch_size / 2)
        history = []
        for epoch in range(epochs):
          print(f"Currently training on Epoch {epoch+1}")
          for i in range(training_images.shape[0]//batch_size):
          

              # ---------------------
              #  Train Discriminator
              # ---------------------

              # Select a random half batch of images
              idx = np.random.randint(0, X_train.shape[0], half_batch)
              imgs = X_train[idx]
              noise = get_noise(half_batch, nlatent_dim)

              # Generate a half batch of new images
              gen_imgs = generator.predict(noise)

              
              # Train the discriminator q: better to mix them together?
              d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
              d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
              d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


              # ---------------------
              #  Train Generator
              # ---------------------

              noise = get_noise(batch_size, nlatent_dim)

              # The generator wants the discriminator to label the generated samples
              # as valid (ones)
              valid_y = (np.array([1] * batch_size)).reshape(batch_size,1)
              
              # Train the generator
              g_loss = combined.train_on_batch(noise, valid_y)

              history.append({"D":d_loss[0],"G":g_loss})
              
              if i % 7 == 0:
                  #Plot the progress
                  print ("Epoch {:05.0f} [D loss: {:4.3f}, acc.: {:05.1f}%] [G loss: {:4.3f}]".format(epoch, d_loss[0], 100*d_loss[1], g_loss))
                  plot_generated_images(noise_plot)
                  plt.show()
                  
                        
        return(history)


_models = (GAN, discriminator, generator) 

history = train(_models, training_images, noise,epochs=epochs, batch_size=batch_size)

print('Training completed with all epochs')\