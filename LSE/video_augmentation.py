# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:57:35 2022

@author: SNT
"""

import tensorflow as tf
import tensorflow_addons as tfa


def augment_video(video_input, p = .3):
        ih, iw = video_input.shape[1:3]
        pa =  tf.random.uniform((5,),
            minval=0,
            maxval=1,
            dtype=tf.dtypes.float16)
    
        if pa[0] < p:
            angle = tf.random.uniform((1,), minval=0, maxval=.1, dtype=tf.dtypes.float32)
            video_input = tfa.image.rotate(video_input, angle)
            # print('1')
    
        if pa[1] < p:
            central_fraction = tf.random.uniform((1,), minval=0.85, maxval=.99, dtype=tf.dtypes.float32)
            video_input = tf.image.central_crop(video_input, central_fraction)
            video_input = tf.image.resize(video_input, (ih, iw))
            # print('2')
            
        # if pa[2] < p:
        #     delta = tf.random.uniform((1,), minval=1e-4, maxval=1e-3, dtype=tf.dtypes.float32)
        #     video_input = tf.image.adjust_brightness(
        #     video_input, delta)
        #     # print('3')
    
        if pa[3] < p:
            factor = tf.random.uniform((1,), minval=0.5, maxval=2, dtype=tf.dtypes.float32)
            video_input = tf.image.adjust_saturation (
            video_input, factor[0])
            # print('4')
      
        if pa[4] < p:
            factor = tf.random.uniform((1,), minval=0.5, maxval=2, dtype=tf.dtypes.float32)
            video_input = tf.image.adjust_contrast(
            video_input, factor[0])
            # print('5')
        return video_input

class AugmentationLayer(tf.keras.layers.Layer):
    
    def __init__(self, p = .25, **kwargs):
        super(AugmentationLayer, self).__init__(**kwargs)
        self.p = p
        self.aug_fn = augment_video
    
    def __call__(self, video_input, training = False):
        if training: 
            video_input = tf.map_fn(lambda x: self.aug_fn(x, p = self.p), video_input)
        return video_input
    
    
    
#%%
# from signClassification_utils import _parse_dataset, _pad_img_batch, _parse_image
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split


# ''' PREPARING DATASET '''
# dataset_folder = 'ngt_3sings'
# list_images, labels = _parse_dataset(dataset_folder)
# iw,ih = list_images[0][-1].shape[:2]

# le = LabelEncoder()

# y = le.fit_transform(labels)
# nsamples = len(y)
# nclasses = len(set(y))
# y = tf.keras.utils.to_categorical(y)
# x_train, x_val, y_train, y_val = train_test_split(list_images, y, test_size = int(nsamples*.3),
#                                                   stratify = y)


#%%
''' PAD IMAGE SECUENCES '''


# img_r = tf.keras.layers.experimental.preprocessing.Resizing(256, 192)
# img_scaler = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / 255)

# with tf.device("/cpu:0"):
#     x_train = tf.stack(_pad_img_batch(x_train, iw, ih))
#     x_val = tf.stack(_pad_img_batch(x_val, iw, ih))
#     im_r = tf.keras.layers.TimeDistributed(img_r)
#     im_scaler = tf.keras.layers.TimeDistributed(img_scaler)    
#     x_val = im_scaler(im_r(x_val))
#     x_train = im_scaler(im_r(x_train))    
# #%%    
# import matplotlib.pyplot as plt
# with tf.device("/cpu:0"):
#     video_input = tf.stack(x_train[:16])
#     video_augmented = tf.map_fn(augment_video, video_input)
# #%%
# plt.figure()
# plt.imshow(video_augmented[1,0, ::])
# plt.title('augmented_1')

# plt.figure()
# plt.imshow(video_augmented[1,2, ::])
# plt.title('augmented_2')


# plt.figure()
# plt.imshow(video_input[1,3, ::])
# #%%


# aug_layer = AugmentationLayer()
# with tf.device("/cpu:0"):
#     video_augmented2 = aug_layer(video_input, training = True)

# vidx = 5

# plt.figure()
# plt.imshow(video_augmented2[vidx,0, ::])
# plt.title('augmented_1')

# plt.figure()
# plt.imshow(video_augmented2[vidx,2, ::])
# plt.title('augmented_2')

# plt.figure()
# plt.imshow(video_input[vidx,3, ::])
# plt.title('Ori')
