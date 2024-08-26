# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 12:05:39 2022

@author: SNT
"""

import tensorflow as tf

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''                        LANDMARK-LEVEL AUGMENTATION                      '''
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def _apply_mirroring(lms):
    mirrored = tf.stack([1-lms[:,:,0], lms[:,:,1], lms[:,:,2]], -1)
    return mirrored 

def _apply_horizonal_shift(lms, random_h_shift):
    shifted = tf.stack([lms[:,:,0]+random_h_shift, lms[:,:,1], lms[:,:,2]], -1)
    return shifted

def _apply_vertical_shift(lms, random_v_shift):
    shifted = tf.stack([lms[:,:,0], lms[:,:,1]+random_v_shift, lms[:,:,2]], -1)
    return shifted

def _apply_lms_horizonal_shift(lms, stddev = .01):
    random_h_shift = tf.random.normal([1], stddev = stddev, dtype = tf.float64)
    shifted = tf.stack([lms[:,:,0]+random_h_shift, lms[:,:,1], lms[:,:,2]], -1)
    return shifted

def _apply_lms_vertical_shift(lms, stddev = .01):
    random_v_shift = tf.random.normal([1], stddev = stddev, dtype = tf.float64)
    shifted = tf.stack([lms[:,:,0], lms[:,:,1]+random_v_shift, lms[:,:,2]], -1)
    return shifted



class GlobalLandmarksAugmentationLayer(tf.keras.layers.Layer):
    def __init__(self, p = .5, **kwargs):
        super(GlobalLandmarksAugmentationLayer, self).__init__(**kwargs)
        self.p = p
    
    def __call__(self, list_of_lms, training = False):
        if training: 
            # lms = tf.map_fn(lambda x: self.aug_fn(x, p = self.p), lms)
            pa =  tf.random.uniform((5,), minval=0, maxval=1, dtype=tf.dtypes.float16)
            if pa[0] < self.p: # MIRRONING
                list_of_lms = [_apply_mirroring(lms)
                               if tf.reduce_sum(lms) > 0  and tf.reduce_mean(lms) != -1
                               else lms for lms in list_of_lms]
            if pa[1] < self.p: # H-SHIFT
                random_h_shift = tf.random.normal([1], stddev = .05, dtype = tf.float64)
                list_of_lms = [_apply_horizonal_shift(lms, random_h_shift)
                                if tf.reduce_sum(lms) > 0  and tf.reduce_mean(lms) != -1
                                else lms for lms in list_of_lms]                
            if pa[2] < self.p: # V-SHIFT
                random_v_shift = tf.random.normal([1], stddev = .05, dtype = tf.float64)
                list_of_lms = [_apply_vertical_shift(lms, random_v_shift)
                                if tf.reduce_sum(lms) > 0 and tf.reduce_mean(lms) != -1
                                else lms for lms in list_of_lms]
            # TODO: ZOOM IN - ZOOM OUT - SOFT ROTATIONS
        return list_of_lms



class IntraLandmarksAugmentationLayer(tf.keras.layers.Layer):
    def __init__(self, p = .5, **kwargs):
        super(IntraLandmarksAugmentationLayer, self).__init__(**kwargs)
        self.p = p
    
    def __call__(self, list_of_lms, training = False):
        if training: 
            # lms = tf.map_fn(lambda x: self.aug_fn(x, p = self.p), lms)
            pa =  tf.random.uniform((2,), minval=0, maxval=1, dtype=tf.dtypes.float16)
            if pa[0] < self.p: # H-SHIFT
                list_of_lms = [_apply_lms_horizonal_shift(lms, stddev = .01)
                               if tf.reduce_sum(lms) > 0 and tf.reduce_mean(lms) != -1
                               else lms for lms in list_of_lms]
            if pa[1] < self.p: # V-SHIFT
                list_of_lms = [_apply_lms_vertical_shift(lms, stddev = .01)
                                if tf.reduce_sum(lms) > 0 and tf.reduce_mean(lms) != -1
                                else lms for lms in list_of_lms]
        return list_of_lms



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''                        SEQUENCE-LEVEL AUGMENTATION                      '''
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def _down_up_sampling_sequence(list_of_sequences):
    seq_len = list_of_sequences[0].shape[0]
    delta_factor = float(tf.random.uniform([1], minval=0.8, maxval=1.2, dtype=tf.dtypes.float16))
    indices = tf.cast(tf.math.floor(tf.range(0, limit = seq_len, delta = delta_factor)), dtype = tf.int32)
    list_of_sequences = [tf.gather(lms, indices, axis = 0)
                         for lms in list_of_sequences]
    return list_of_sequences

def _cut_seq_beginning(list_of_sequences):
    seq_len = list_of_sequences[0].shape[0]
    starting_point = int(tf.random.uniform([1], minval=0, maxval=seq_len*0.1))
    indices = tf.cast(tf.math.round(tf.range(starting_point, limit = seq_len, delta = 1)), dtype = tf.int32)
    list_of_sequences = [tf.gather(lms, indices, axis = 0)
                         for lms in list_of_sequences]
    return list_of_sequences

def _cut_seq_ending(list_of_sequences):
    seq_len = list_of_sequences[0].shape[0]
    ending_point = int(tf.random.uniform([1], minval=0, maxval=seq_len*0.1))
    indices = tf.cast(tf.math.round(tf.range(0, limit = int(seq_len-ending_point), delta = 1)), dtype = tf.int32)    
    list_of_sequences = [tf.gather(lms, indices, axis = 0)
                         for lms in list_of_sequences]
    return list_of_sequences

def _random_shuffle(list_of_sequences):
    seq_len = list_of_sequences[0].shape[0]
    margin = int(0.1*seq_len)
    center_point = int(tf.random.uniform([1], minval=margin, maxval=seq_len-margin))   
    indices = tf.concat([
        tf.range(0, limit = center_point-margin), # BEGGINING
        tf.random.shuffle(tf.range(center_point-margin, limit = center_point+margin)), # SHUFFLE
        tf.range(center_point+margin, limit = seq_len), # END
        ], axis = 0)
    list_of_sequences = [tf.gather(lms, indices, axis = 0)
                          for lms in list_of_sequences]
    return list_of_sequences

def _random_reverse(list_of_sequences):
    seq_len = list_of_sequences[0].shape[0]
    margin = int(0.1*seq_len)
    center_point = int(tf.random.uniform([1], minval=margin, maxval=seq_len-margin))   
    indices = tf.concat([
        tf.range(0, limit = center_point-margin), # BEGGINING
        tf.reverse(tf.range(center_point-margin, limit = center_point+margin), [0]), # REVERSE
        tf.range(center_point+margin, limit = seq_len), # END
        ], axis = 0)
    list_of_sequences = [tf.gather(lms, indices, axis = 0)
                          for lms in list_of_sequences]
    return list_of_sequences

def _down_up_sampling_seq_middle(list_of_sequences):
    seq_len = list_of_sequences[0].shape[0]
    margin = int(0.1*seq_len)
    center_point = int(tf.random.uniform([1], minval=margin, maxval=seq_len-margin))
    delta = float(tf.random.uniform([1], minval=0.5, maxval=1.5, dtype=tf.dtypes.float16))
    indices = tf.concat([
        tf.range(0, limit = center_point-margin, delta = 1), # BEGGINING
        tf.cast(tf.range(center_point-margin, limit = center_point+margin, delta = delta), dtype = tf.int32), # SAMPLING
        tf.cast(tf.range(center_point+margin, limit = seq_len, delta = 1), dtype = tf.int32), # END
        ], axis = 0)
    list_of_sequences = [tf.gather(lms, indices, axis = 0)
                          for lms in list_of_sequences]
    return list_of_sequences

# TODO : Mask random samples to -1
class SequenceLevelAugmentationLayer(tf.keras.layers.Layer):
    def __init__(self, p = .5, **kwargs):
        super(SequenceLevelAugmentationLayer, self).__init__(**kwargs)
        self.p = p
    
    def __call__(self, list_of_sequences, training = False):
        if training: 
            pa =  tf.random.uniform((6,), minval=0, maxval=1, dtype=tf.dtypes.float16)
            if pa[0] < self.p: 
                list_of_sequences = _cut_seq_beginning(list_of_sequences)
            if pa[1] < self.p: 
                list_of_sequences = _cut_seq_ending(list_of_sequences)
            if pa[2] < self.p:
                list_of_sequences = _down_up_sampling_sequence(list_of_sequences)
            if pa[3] < self.p: 
                list_of_sequences = _down_up_sampling_seq_middle(list_of_sequences)    
            if pa[4] < self.p: 
                list_of_sequences = _random_shuffle(list_of_sequences)    
            if pa[4] < self.p: 
                list_of_sequences = _random_reverse(list_of_sequences)  
            return list_of_sequences


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''                          TESTING AUGMENTATION                           '''
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os, sys
import matplotlib.pyplot as plt
sys.path.insert(1, '..')
from signClassification_utils import _deserialize_data

ds_folder = "../data_SSL/videos"
data_fns = [os.path.join(ds_folder,f) for f in os.listdir(ds_folder)]
# list_of_data = [_deserialize_data(fn) for fn in data_fns]
sample = _deserialize_data(data_fns[0]) 



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''                            ORIGINAL IMAGE                               '''
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

seq_idx = 10
img = sample['image'][seq_idx]
hand_r = tf.convert_to_tensor(sample['mp_hands (R,L)'][1][seq_idx])


h,w = img.shape[:2]

plt.figure()
plt.imshow(img)
plt.scatter(hand_r[:,0]*h, hand_r[:,1]*w, s=0.2, c = 'r')
plt.title('Original')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''                            MIRRORED IMAGE                               '''
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hand_r_m= _apply_mirroring(hand_r)

plt.figure()
plt.imshow(img[:,::-1,:])
plt.scatter((hand_r_m[:,0])*h, hand_r[:,1]*w, s=0.2, c = 'r')
plt.title('Mirrored')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''                             H SHIFT IMAGE                               '''
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# hand_r_sh = _apply_horizonal_shift(hand_r)

# plt.figure()
# plt.imshow(img)
# plt.scatter((hand_r_sh[:,0])*h, hand_r_sh[:,1]*w, s=0.2, c = 'r')
# plt.title('TOTAL H SHIFT')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''                             V SHIFT IMAGE                               '''
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# hand_v_sh = _apply_vertical_shift(hand_r)
# plt.figure()
# plt.imshow(img)
# plt.scatter((hand_v_sh[:,0])*h, hand_v_sh[:,1]*w, s=0.2, c = 'r')
# plt.title('TOTAL V SHIFT')






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''                    GlobalLandmarksAugmentationLayer                     '''
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# global_aug_layer = GlobalLandmarksAugmentationLayer()
# intra_aug_layer = IntraLandmarksAugmentationLayer()

# rhand_seq = tf.convert_to_tensor(sample['mp_hands (R,L)'][0])
# lhand_seq = tf.convert_to_tensor(sample['mp_hands (R,L)'][1])
# pose_seq = tf.convert_to_tensor(sample['mp_pose'])[:,:22,:]
# face_seq = tf.convert_to_tensor(sample['mp_face'])

# # SEQ LENGTH; NUMBER OF LANDMARKS; COORDINATES
# pose_seq_1, face_seq_1, lhand_seq_1, rhand_seq_1 = intra_aug_layer([pose_seq, face_seq, lhand_seq, rhand_seq], training = True)


# pose_seq_2, face_seq_2, lhand_seq_2, rhand_seq_2 = global_aug_layer([pose_seq, face_seq,
#                                                               lhand_seq, rhand_seq],
#                                                             training = True)

# pose_seq_3, face_seq_3, lhand_seq_3, rhand_seq_3 = global_aug_layer([pose_seq_1, face_seq_1,
                                                              # lhand_seq_1, rhand_seq_1],
                                                            # training = True)

#%%
# def plot_img_lms(seq_idx):
#     h,w = sample['image'][seq_idx].shape[:2]
#     plt.figure()
#     plt.imshow(sample['image'][seq_idx])
#     plt.scatter((rhand_seq[seq_idx, :,0])*w, rhand_seq[seq_idx, :,1]*h, s=0.3, c = 'r')
#     plt.scatter((lhand_seq[seq_idx, :,0])*w, lhand_seq[seq_idx, :,1]*h, s=0.3, c = 'b')
#     plt.scatter((face_seq[seq_idx, :,0])*w, face_seq[seq_idx, :,1]*h, s=0.3, c = 'y')
#     plt.scatter((pose_seq[seq_idx, :,0])*w, pose_seq[seq_idx, :,1]*h, s=0.3, c = 'k')
#     plt.title('Input')

#     plt.figure()
#     plt.imshow(sample['image'][seq_idx])
#     plt.scatter((rhand_seq_1[seq_idx, :,0])*w, rhand_seq_1[seq_idx, :,1]*h, s=0.3, c = 'r')
#     plt.scatter((lhand_seq_1[seq_idx, :,0])*w, lhand_seq_1[seq_idx, :,1]*h, s=0.3, c = 'b')
#     plt.scatter((face_seq_1[seq_idx, :,0])*w, face_seq_1[seq_idx, :,1]*h, s=0.3, c = 'y')
#     plt.scatter((pose_seq_1[seq_idx, :,0])*w, pose_seq_1[seq_idx, :,1]*h, s=0.3, c = 'k')
#     plt.title('IntraLandmarksAugmentationLayer OUTPUT')
    
#     plt.figure()
#     plt.imshow(sample['image'][seq_idx])
#     plt.scatter((rhand_seq_2[seq_idx, :,0])*w, rhand_seq_2[seq_idx, :,1]*h, s=0.3, c = 'r')
#     plt.scatter((lhand_seq_2[seq_idx, :,0])*w, lhand_seq_2[seq_idx, :,1]*h, s=0.3, c = 'b')
#     plt.scatter((face_seq_2[seq_idx, :,0])*w, face_seq_2[seq_idx, :,1]*h, s=0.3, c = 'y')
#     plt.scatter((pose_seq_2[seq_idx, :,0])*w, pose_seq_2[seq_idx, :,1]*h, s=0.3, c = 'k')
#     plt.title('GlobalLandmarksAugmentationLayer OUTPUT')


#     plt.figure()
#     plt.imshow(sample['image'][seq_idx])
#     plt.scatter((rhand_seq_3[seq_idx, :,0])*w, rhand_seq_3[seq_idx, :,1]*h, s=0.3, c = 'r')
#     plt.scatter((lhand_seq_3[seq_idx, :,0])*w, lhand_seq_3[seq_idx, :,1]*h, s=0.3, c = 'b')
#     plt.scatter((face_seq_3[seq_idx, :,0])*w, face_seq_3[seq_idx, :,1]*h, s=0.3, c = 'y')
#     plt.scatter((pose_seq_3[seq_idx, :,0])*w, pose_seq_3[seq_idx, :,1]*h, s=0.3, c = 'k')
#     plt.title('All together OUTPUT')


# plot_img_lms(50)
# plot_img_lms(70)


#%%


# pose_seq_1, face_seq_1, lhand_seq_1, rhand_seq_1 = _down_up_sample_sequence([pose_seq, face_seq,
#                                                              lhand_seq, rhand_seq])

# pose_seq_2, face_seq_2, lhand_seq_2, rhand_seq_2 = _cut_seq_beginning([pose_seq, face_seq,
#                                                              lhand_seq, rhand_seq])

# pose_seq_3, face_seq_3, lhand_seq_3, rhand_seq_3 = _cut_seq_ending([pose_seq, face_seq,
#                                                               lhand_seq, rhand_seq])

# pose_seq_4, face_seq_4, lhand_seq_4, rhand_seq_4 = _down_up_sample_seq_middle([pose_seq, face_seq, lhand_seq, rhand_seq])
#%%
# seq_aug_layer = SequenceLevelAugmentationLayer()
#%%
# for _ in range(10):
#     img_seq = tf.convert_to_tensor(sample['image'])
#     rhand_seq = tf.convert_to_tensor(sample['mp_hands (R,L)'][0])
#     lhand_seq = tf.convert_to_tensor(sample['mp_hands (R,L)'][1])
#     pose_seq = tf.convert_to_tensor(sample['mp_pose'])[:,:22,:]
#     face_seq = tf.convert_to_tensor(sample['mp_face'])
    
    
#     a = seq_aug_layer([img_seq], training = True)
#     print(a[0].shape)

#%%

# samples = list_of_data 


# global_aug_layer = GlobalLandmarksAugmentationLayer()
# intra_aug_layer = IntraLandmarksAugmentationLayer()
# seq_aug_layer = SequenceLevelAugmentationLayer()

# rhand_seq = [tf.convert_to_tensor(s['mp_hands (R,L)'][0]) for s in samples]
# lhand_seq = [tf.convert_to_tensor(s['mp_hands (R,L)'][1]) for s in samples]
# pose_seq = [tf.convert_to_tensor(s['mp_pose'])[:,:22,:] for s in samples]
# face_seq = [tf.convert_to_tensor(s['mp_face']) for s in samples]

# def augment_sequences(seq_batch):
#     seq_batch_1 = [intra_aug_layer(t, training = True) for t in seq_batch]
#     seq_batch_2 = [global_aug_layer(t, training = True) for t in seq_batch_1]
#     seq_batch_3 = [seq_aug_layer(t, training = True) for t in seq_batch_2]
#     return seq_batch_3

# aligned_seq = [s for s in zip(pose_seq, face_seq, lhand_seq, rhand_seq)]

# augmented_batch = augment_sequences(aligned_seq)










