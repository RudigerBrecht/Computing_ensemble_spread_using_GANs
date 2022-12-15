import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import time

import netCDF4 as nc

from tensorflow.keras.layers import Input, Dense, Conv3D, Conv3DTranspose, Reshape, Flatten, \
  LeakyReLU, Activation, Dropout, Concatenate, BatchNormalization, \
  ConvLSTM2D, ZeroPadding3D, Cropping3D, RepeatVector

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras import metrics, losses

###

plt.rcParams['image.cmap'] = 'coolwarm'
plt.rcParams.update({'font.size': 18})

#tf.keras.backend.set_floatx('float16')

###


###

###
# define the encoder block
def convolution_block(layer_in, n_filters, kernel_size, stride, batchnorm=True):
	
  # weight initialization
	init = tf.random_normal_initializer(0., 0.02)
	
  # add downsampling layer
	g = Conv3D(n_filters, kernel_size=kernel_size, strides=stride, padding='same', 
            kernel_initializer=init, use_bias=False)(layer_in)
	
  # conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g)#, training=True)
	# leaky relu activation
	g = LeakyReLU()(g)
	return g
 
# define the decoder block
def deconvolution_concat_block(layer_in, skip_in, n_filters, kernel_size, stride, cropping, dropout=True):
	
  # weight initialization
	init = tf.random_normal_initializer(0., 0.02)
	
  # add upsampling layer
	g = Conv3DTranspose(n_filters, kernel_size=kernel_size, strides=stride, padding='same', 
                     kernel_initializer=init, use_bias=False)(layer_in)
	# add cropping
	g = Cropping3D(cropping=cropping)(g)

  # add batch normalization
	g = BatchNormalization()(g)#, training=True)
 
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g)#, training=True)

  # relu activation
	g = Activation('relu')(g)

  # merge with skip connection
	g = Concatenate()([g, skip_in])
	#lm = ConvLSTM2D(256, (2,2), padding='same', return_sequences=True,
	#                dropout=0.2, recurrent_dropout=0.2)(skip_in, training=True)
	#g = Concatenate()([g, lm])    
	
	return g

# define the decoder block
def deconvolution_block(layer_in, n_filters, kernel_size, stride, cropping, dropout=True):
	
  # weight initialization
	init = tf.random_normal_initializer(0., 0.02)
	
  # add upsampling layer
	g = Conv3DTranspose(n_filters, kernel_size=kernel_size, strides=stride, padding='same', 
                     kernel_initializer=init, use_bias=False)(layer_in)
	# add cropping
	g = Cropping3D(cropping=cropping)(g)

  # add batch normalization
	g = BatchNormalization()(g)#, training=True)

  # relu activation
	g = Activation('relu')(g)

	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g)#, training=True)
	
	return g

# define the standalone generator model
def define_generator(image_shape):
	
  # weight initialization
	init = tf.random_normal_initializer(0., 0.02)
	
  # image input
	in_image = Input(shape=image_shape)
	
  # encoder model:
	e1 = convolution_block(in_image, 16, (4,4,4), (2,2,2), batchnorm=False)
 	# e1 = Concatenate(axis=1)([e1, e1, e1, e1])

	e2 = convolution_block(e1, 32, (4,4,4), (1,2,2))
	e3 = convolution_block(e2, 64, (4,4,4), (1,2,2))
	e4 = convolution_block(e3, 128, (3,4,4), (1,2,2))
	e5 = convolution_block(e4, 128, (3,4,4), (1,2,2))
	e6 = convolution_block(e5, 256, (3,4,4), (1,2,2)) 
	b = convolution_block(e6, 256, (3,4,4), (1,1,1))  

	# bottleneck LSTM 
	#b = ConvLSTM2D(256, (2,2), padding='same', return_sequences=True,
	#               dropout=0.2, recurrent_dropout=0.2)(b, training=True)
	#b = ConvLSTM2D(256, (2,2), padding='same', return_sequences=True,
	#               dropout=0.2, recurrent_dropout=0.2)(b, training=True)
	
  # decoder model:
	d1 = deconvolution_concat_block(b, e6, 256, (3,4,4), (1,1,1), ((0,0),(0,0),(0,0)))
	#d1= Cropping3D(cropping=((0,0),(0,0),(0,0)))(d1)      
	d2 = deconvolution_concat_block(d1, e5, 128, (3,4,4), (1,2,2), ((0,0),(0,0),(1,0)))
      
	d3 = deconvolution_concat_block(d2, e4, 128, (3,4,4), (1,2,2), ((0,0),(1,0),(1,0)))
	d4 = deconvolution_concat_block(d3, e3, 64, (4,4,4), (1,2,2), ((0,0),(1,0),(0,0))) 
	d5 = deconvolution_concat_block(d4, e2, 32, (4,4,4), (1,2,2), ((0,0),(0,0),(0,0)))
	d6 = deconvolution_concat_block(d5, e1, 16, (4,4,4), (1,2,2), ((0,0),(0,0),(0,0)), dropout=False)
	d6 = deconvolution_block(d6, 16, (4,4,4), (2,1,1), ((6,0),(0,0),(0,0)), dropout=False)
	#d6= Cropping3D(cropping=((1,0),(0,0),(0,0)))(d6)        
	d6 = deconvolution_block(d6, 16, (4,4,4), (2,1,1), ((10,0),(0,0),(0,0)), dropout=False)
	#d6= Cropping3D(cropping=((2,0),(0,0),(0,0)))(d6)    
	# d6 = deconvolution_block(d6, 32, (4,4,4), (2,1,1), ((0,0),(0,0),(0,0)), dropout=False)

  # # output
	out_image = Conv3DTranspose(nr_levels, (4,4,4), strides=(2,2,2), padding='same', 
                     kernel_initializer=init)(d6)
	out_image= Cropping3D(cropping=((4,0),(0,0),(0,0)))(out_image)    
    
	out_image = Activation('relu')(out_image)
 	
  # define model
	model = Model(in_image, out_image)
 
	return model


###

input_shape=(16,360,720,1)
nr_levels=1
generator = define_generator(input_shape)
generator.summary()
output_shape=(16,360,720,1)

###

LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # MAE
  l1_loss = tf.reduce_mean(tf.abs(target-gen_output))

  total_gen_loss = gan_loss + LAMBDA * l1_loss

  return total_gen_loss, gan_loss, l1_loss
###

def define_discriminator(in_shape=input_shape, out_shape=output_shape):

  # Weight initialization
  init = tf.random_normal_initializer(0., 0.02)

  inp_X = Input(shape=in_shape) # Input layer past data
  inp_Y = Input(shape=out_shape) # Input layer future data

  # Add some noise
  h1 = tf.keras.layers.GaussianNoise(0.01)(inp_X)
  h2 = tf.keras.layers.GaussianNoise(0.01)(inp_Y)  

  # Bring data into same shape
  h1 = convolution_block(h1, 32, (4,4,4), (1,2,2), batchnorm=False)
  #h1 = convolution_block(h1, 64, (4,4,4), (2,2,2))
  #h1 = convolution_block(h1, 128, (4,4,4), (2,2,2))  

  h2 = convolution_block(h2, 32, (4,4,4), (1,2,2), batchnorm=False)
  #h2 = convolution_block(h2, 64, (4,4,4), (2,2,2))
  #h2 = convolution_block(h2, 128, (4,4,4), (2,2,2))

  # Concatenate data
  h = Concatenate()([h1, h2])

  h = convolution_block(h, 256, (4,4,4), (2,2,2))
  # h = convolution_block(h, 256, (4,4,4), (1,2,2))
  # h = convolution_block(h, 256, (3,4,4), (1,2,2))

  # h = ZeroPadding3D()(h)
  h = Conv3D(256, (4,4,4), strides=(1,2,2), padding='same',
                                kernel_initializer=init,
                                use_bias=False)(h)
  h = BatchNormalization()(h)
  h = LeakyReLU()(h)

  # h = ZeroPadding3D()(h)
  outp = Conv3D(1, (4,4,4), strides=(2,1,1), padding='same',
                                kernel_initializer=init)(h)

  model = Model([inp_X, inp_Y], outp)
  
  return model

###

discriminator = define_discriminator(input_shape)
discriminator.summary()

###

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.math.scalar_mul(0.9, tf.ones_like(disc_real_output)), disc_real_output)

  generated_loss = loss_object(tf.math.add(0.1, tf.zeros_like(disc_generated_output)), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

###

###
@tf.function
def train_step(input_image, target, epoch):

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
    
###

def fit(cf,spread, epoch):
    for i in range(epoch):
        
        now = time.time()
        
        step = tf.constant(1)
        
        inds=np.arange(len(cf))
        np.random.shuffle(inds)
        for ind in inds:
            train_step(cf[None,ind,0:16,:,:,None].astype(np.float32), spread[None,ind,0:16,:,:,None].astype(np.float32), step)
            step += tf.constant(1)
                #print(step)
            
        later = time.time()
        print('time for epoch: '+str(i)+' took '+str(int(later - now)) + ' sec')
        val_err=0
        for kk in range(200):
            nnspread=generator(cf_val[None,kk,0:16,:,:,None])
            val_err+= rmse(spread_val[kk,0:16,:,:],nnspread[0,0:16,:,:,0].numpy())
        print(val_err/200)
        
        generator.save('modelsPix/new/generator_A_e_'+str(i))
        discriminator.save('modelsPix/new/discriminator_A_e_'+str(i))
        
        
ds20_s = nc.Dataset('10-20_es.nc')
ds20_c = nc.Dataset('10-20_cf.nc')

def mynormalize1(x):
    return np.float16(2)*(x-np.min(x))/(np.max(x)-np.min(x))-np.float16(1)
def mynormalize2(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def rmse(x, y):
    return np.sqrt(((x - y)**2).mean())

print('load data')
###

# cf is the control run of the geopotential 
# spread is the standard deviation of the control run to the ensemble 
# We use about 10 years for training
# the format of the data cf[i,j,k,l] where
# i - is the day where the cf/spread starts
# j - is the snapshot time of the simulation
# k - resolution of the latitude
# l  - resolution of the longitude

cf=mynormalize1(np.float16(ds20_c['z'][0:3400,0:17,0:360,0:720])) # 85% of set
spread=mynormalize2(np.float16(ds20_s['z'][0:3400,0:17,0:360,0:720]))
cf_val=mynormalize1(np.float16(ds20_c['z'][3400:3600,0:17,0:360,0:720]))
spread_val=mynormalize2(np.float16(ds20_s['z'][3400:3600,0:17,0:360,0:720])) # 5% of set

print('start training')

###

fit(cf,spread,10)
