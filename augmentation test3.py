# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:47:14 2019

@author: Roman Prochazka
"""

# import libraries
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import os
import glob

# how many output images will be generated per one input image
N = 2

#directory with raw images
datadir = "../image-augmentation/test"
#directory after augmentation
save_dir = datadir + "/aug"

# create save_dir directory if doesnot exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# load images and data augmantation for each input image
    
# empty array  
images = []
#loading file names (/*.png) in directory 
for img in glob.glob(datadir + '/*.png'):
    #loading image
    n= load_img(img)
    #append image
    images.append(n)
    #convert image to array
    data = img_to_array(n)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(brightness_range=[0.1,1.2],
                                 width_shift_range=[-10,10],
                                 height_shift_range=[-10,10],
                                 rotation_range=2,
                                 zoom_range=[0.95, 1.05])
    # prepare iterator and data saving to save_dir 
    it = datagen.flow(samples, batch_size=1, save_to_dir= save_dir, save_prefix='aug', save_format='png')
    for i in range(N):
    	# generate batch of images
    	batch = it.next()
    	# convert to unsigned integers for viewing
    	image = batch[0].astype('uint8')
