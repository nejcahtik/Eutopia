# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:58:50 2022

@author: SNT
"""

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from video_augmentation import AugmentationLayer
# from transformers.models.mbart.modeling_tf_mbart import 




# def get_classifier_model(pose_model, model_type, n_classes  = 3, aug_p = .25):
    
#     if model_type in RUN_EAGERLY:
#         tf.config.run_functions_eagerly(True)
    
#     input_var = tf.keras.layers.Input((9, 256, 192, 3,), name = 'input')
#     pose_model_input = AugmentationLayer(p = aug_p)(input_var)
#     pose_heatmaps = pose_model(pose_model_input)
    
    


        
#     logits = tf.keras.layers.Dense(units=n_classes, activation="softmax")(last_layer_input)       
#     return tf.keras.Model(input_var, logits)





    
def _compute_seq_mask(img_seq):
    return (tf.reduce_sum(img_seq, axis = [2,3,4]) > 0)


#%%%
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.utils import shuffle
# from signClassification_utils import _parse_dataset, _pad_img_batch
# from tf2cv.model_provider import get_model as tf2cv_get_model

# dataset_folder = 'ngt_3sings'
# list_images, labels = _parse_dataset(dataset_folder)
# iw,ih = list_images[0][-1].shape[:2]
# seed = 0
# le = LabelEncoder()

# y = le.fit_transform(labels)
# nsamples = len(y)
# nclasses = len(set(y))
# y = tf.keras.utils.to_categorical(y)
# x_train, x_val, y_train, y_val = train_test_split(list_images, y, test_size = int(nsamples*.3),
#                                                   stratify = y, random_state = seed)
# ''' PAD IMAGE SECUENCES '''
# img_r = tf.keras.layers.experimental.preprocessing.Resizing(256, 192)
# img_scaler = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / 255)

# with tf.device("/cpu:0"):
#     x_train = tf.stack(_pad_img_batch(x_train, iw, ih))
#     x_val = tf.stack(_pad_img_batch(x_val, iw, ih))
#     im_r = tf.keras.layers.TimeDistributed(img_r)
#     im_scaler = tf.keras.layers.TimeDistributed(img_scaler)    
#     x_val = im_scaler(im_r(x_val))
#     x_train = im_scaler(im_r(x_train))    


    
# batch = x_train[:2]

# pose_model_name = 'simplepose_resnet18_coco'
# pose_model = tf2cv_get_model(pose_model_name, pretrained=True, data_format="channels_last", return_heatmap = True)

# td_pose_model = tf.keras.layers.TimeDistributed(pose_model)

#%%
# tf.config.run_functions_eagerly(True)
# input_var = tf.keras.layers.Input((9, 256, 192, 3,), name = 'input')
# pose_model_input = AugmentationLayer()(input_var)
# pose_heatmaps = td_pose_model(pose_model_input)

# cnn3d_output = tf.keras.layers.Conv3D(128, (1,64,48), activation = 'relu')(pose_heatmaps)
# cnn3d_output_r = tf.keras.layers.Lambda(lambda x: x[:,:,0,0,:])(cnn3d_output)
# mask = tf.keras.layers.Lambda(lambda x:  _expand_mask(_compute_seq_mask(x)))(input_var)
# encoder1_output = TFMBartEncoderLayer(128, 4, 256)(cnn3d_output_r, mask)
# last_layer_input = tf.keras.layers.Lambda(lambda x: tf.gather(x, 8, axis = 1))(encoder1_output[0])


# model = tf.keras.Model(input_var, last_layer_input)
# optimizer = tf.keras.optimizers.Adam()
# model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics = 'accuracy')


# o_test = model(batch)
# print(o_test.shape)
# print(o_test[0].shape)
# print(o_test[1].shape)

# print(o_test[0][0].shape)
# print(o_test[0][1].shape)