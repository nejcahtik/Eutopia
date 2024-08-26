# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:50:36 2022

@author: SNT
"""
import os, sys
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from signClassification_utils import _deserialize_data
from augmentation_layer import GlobalLandmarksAugmentationLayer, IntraLandmarksAugmentationLayer, SequenceLevelAugmentationLayer

import numpy as np
from helpers import *


#%%
''' EXPERIMENT PARAMETERS '''

bs = 8
nepochs = 200

lr = 1e-4
dataset_folder = 'data/features_test4' if '-d' not in sys.argv else float(sys.argv[sys.argv.index('-d')+1]) 

dirs = os.listdir('./runs')
pattern = 'test'

count = [1 if pattern in fn else 0 for fn in dirs]
WORKING_DIR = 'runs/'+pattern+'_{}'.format(sum(count)+1)
                    
handled_dirs = create_working_environment(WORKING_DIR, subdirs = [])
tf.print(WORKING_DIR)

script_name = os.path.basename(__file__)
settings_dict = {'BATCH_SIZE': bs, 'N_EPOCHS': nepochs, 'LR': lr,   
    'WORKING_DIR': WORKING_DIR, 'script_name': script_name,
    'DATASET' : dataset_folder, #'AUG_P' : aug_p,
    }


data_fns = [os.path.join(dataset_folder,f) for f in os.listdir(dataset_folder)]
list_of_data = [_deserialize_data(fn, exclude=['image']) for fn in data_fns]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''                         SPLIT-PAD-MASK DATASET                          '''
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def zero_to_one(x):
    x[x == 0] = -1
    return x

train_data = [d for d in list_of_data if 'mp4' in d['filename']]
test_data = [d for d in list_of_data if 'mov' in d['filename']]
del list_of_data
#%%
''' DOWNSAMPLING LONG VIDEOS '''

max_seq_len = 80

train_rhands = [d['mp_hands (R,L)'][0][np.linspace(0, int(d['info']['frame_count'])-1, max_seq_len, dtype = int), ::]
                for d in train_data]
train_rhands = list(map(zero_to_one, train_rhands))                            # MAKE NON DETECTED KP TO BE -1

test_rhands = [d['mp_hands (R,L)'][0][np.linspace(0, int(d['info']['frame_count'])-1, max_seq_len, dtype = int), ::]
               for d in test_data]
test_rhands = list(map(zero_to_one, test_rhands))                              # MAKE NON DETECTED KP TO BE -1

train_lhands = [d['mp_hands (R,L)'][1][np.linspace(0, int(d['info']['frame_count'])-1, max_seq_len, dtype = int), ::]
                for d in train_data]
train_lhands = list(map(zero_to_one, train_lhands))                            # MAKE NON DETECTED KP TO BE -1

test_lhands = [d['mp_hands (R,L)'][1][np.linspace(0, int(d['info']['frame_count'])-1, max_seq_len, dtype = int), ::]
               for d in test_data]
test_lhands = list(map(zero_to_one, test_lhands))                              # MAKE NON DETECTED KP TO BE -1

train_face = [d['mp_face'][np.linspace(0, int(d['info']['frame_count'])-1, max_seq_len, dtype = int), ::]
              for d in train_data]
train_face = list(map(zero_to_one, train_face))                                # MAKE NON DETECTED KP TO BE -1

test_face = [d['mp_face'][np.linspace(0, int(d['info']['frame_count'])-1, max_seq_len, dtype = int), ::]
             for d in test_data]
test_face = list(map(zero_to_one, test_face))                                  # MAKE NON DETECTED KP TO BE -1

train_pose = [d['mp_pose'][np.linspace(0, int(d['info']['frame_count'])-1, max_seq_len, dtype = int), ::]
              for d in train_data]
train_pose = list(map(zero_to_one, train_pose))                                # MAKE NON DETECTED KP TO BE -1

test_pose = [d['mp_pose'][np.linspace(0, int(d['info']['frame_count'])-1, max_seq_len, dtype = int), ::]
             for d in test_data]
test_pose = list(map(zero_to_one, test_pose))                                  # MAKE NON DETECTED KP TO BE -1

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''                        AUGMENTATION METHOD                              '''
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global_aug_layer = GlobalLandmarksAugmentationLayer()
intra_aug_layer = IntraLandmarksAugmentationLayer()
seq_aug_layer = SequenceLevelAugmentationLayer()

def augment_sequences(seq_batch):
    seq_batch_1 = [intra_aug_layer(t, training = True) for t in seq_batch]
    seq_batch_2 = [global_aug_layer(t, training = True) for t in seq_batch_1]
    #seq_batch_3 = [seq_aug_layer(t, training = True) for t in seq_batch_2]
    return seq_batch_2


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''                             CREATING MODEL                              '''
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def flat_last_dims(x):
    x_shape = x.shape
    return tf.reshape(x, x_shape[:-2]+(x_shape[-2]*x_shape[-1],))

hand_processor = tf.keras.Sequential([
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, 'linear')),])

face_processor = tf.keras.Sequential([
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, 'relu')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64))])

pose_processor = tf.keras.Sequential([
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, 'relu')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64))])

lstm_model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation = 'relu', return_sequences=True)),
    tf.keras.layers.LSTM(64, activation = 'linear'),
    ])


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''                             TRAINING MODEL                              '''
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from losses import (_dot_simililarity_dim1 as sim_func_dim1,
                    _dot_simililarity_dim2 as sim_func_dim2, get_negative_mask)

negative_mask = get_negative_mask(bs)
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                          reduction=tf.keras.losses.Reduction.SUM)
temperature = 1.0
optimizer = tf.keras.optimizers.Adam(lr)
def forward(batch, training = True):
    # PROCESSING ORIGINAL INPUT
    rhand_f = flat_last_dims(batch[0])
    lhand_f = flat_last_dims(batch[1])
    latent_rhand = hand_processor(rhand_f, training)
    latent_lhand = hand_processor(lhand_f, training)
    latent_hands = (latent_rhand+latent_lhand)/2.0
    embeddings = lstm_model(latent_hands, training)
    return embeddings 

#@tf.function
def train_step(input1, input2, trainable_vars):
    with tf.GradientTape() as tape:
        enc_ori = forward(input1)
        enc_aug = forward(input2)
        
        enc_ori = tf.math.l2_normalize(enc_ori, axis = 1)
        enc_aug = tf.math.l2_normalize(enc_aug, axis = 1)
        
        l_pos = sim_func_dim1(enc_ori, enc_aug)
        l_pos = tf.reshape(l_pos, (bs, 1))
        l_pos /= temperature
    
        negatives = tf.concat([enc_aug, enc_ori], axis=0)
        
        loss = 0
        
        for positives in [enc_ori, enc_aug]:
            l_neg = sim_func_dim2(positives, negatives)
        
            labels = tf.zeros(bs, dtype=tf.int32)
        
            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (bs, -1))
            l_neg /= temperature
        
            logits = tf.concat([l_pos, l_neg], axis=1) 
            loss += criterion(y_pred=logits, y_true=labels)
        
        loss = loss / (2 * bs)
        
    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))    
    return loss




#%%
import time 

test_embeddings = []
train_embeddings = []
            # K = 1; K = 2; K = 5, K = 10
metrics = np.zeros(shape = (nepochs, 5))



n_train_samples = len(train_data)
idx = tf.range(n_train_samples)
n_test_samples = len(test_data)
idx_test = tf.range(n_test_samples)

for ep in range(1, nepochs+1):
    idx = tf.random.shuffle(idx)
    idx_batches = tf.reshape(tf.gather(idx, tf.range(int(n_train_samples/bs)*bs)),
                             (int(n_train_samples/bs), bs))
    
    for ib in idx_batches:
        batch = [[train_rhands[i] for i in ib],
             [train_lhands[i] for i in ib]]
        
        # AUGMENTATION - 1
        aligned_seq_batch = list(zip(batch[0], batch[1]))
        batch_aug_1 = augment_sequences(aligned_seq_batch)
        batch_aug_1 = list(zip(*batch_aug_1))

        # AUGMENTATION - 2
        batch_aug_2 = augment_sequences(aligned_seq_batch)
        batch_aug_2 = list(zip(*batch_aug_2))

    
        input_1 = tf.convert_to_tensor(batch_aug_1)
        input_2 = tf.convert_to_tensor(batch_aug_2)
        # TRAIN STEP
        if any([not hand_processor.built, not lstm_model.built]):
            batch = tf.convert_to_tensor(batch)
            _ = forward(batch)

        trainable_vars = hand_processor.trainable_variables + \
                                lstm_model.trainable_variables
        t0 = time.time()
        step_wise_loss = train_step(input_1, input_2, trainable_vars)
        print(time.time()-t0, step_wise_loss)


    batch_test = [[test_rhands[i] for i in idx_test],
         [test_lhands[i] for i in idx_test]]

    batch_train = [[train_rhands[i] for i in idx_test],
         [train_lhands[i] for i in idx_test]]

    
    
    max_batch_len = max([t.shape[0] for t in batch_test[0]])
    batch_test = tf.convert_to_tensor([[np.concatenate([m, np.zeros((max_batch_len-m.shape[0], m.shape[1],
                                      m.shape[2]))], 0) for m in b] for b in batch_test])
    
    max_batch_len = max([t.shape[0] for t in batch_train[0]])
    batch_train = tf.convert_to_tensor([[np.concatenate([m, np.zeros((max_batch_len-m.shape[0], m.shape[1],
                                          m.shape[2]))], 0) for m in b] for b in batch_train])
    ep_test_embeddings = forward(batch_test, training = False)
    ep_train_embeddings = forward(batch_train, training = False)
    
    test_embeddings.append(ep_test_embeddings.numpy().copy())
    train_embeddings.append(ep_train_embeddings.numpy().copy())





#%%


from sklearn.metrics import pairwise_distances, top_k_accuracy_score

epochs_to_eval = range(0,nepochs,1)
metrics = np.zeros(shape = (len(epochs_to_eval),3))
y_true = np.arange(0, n_train_samples)
for i, ep in enumerate(epochs_to_eval):

    pair_dist_l2 = pairwise_distances(tf.math.l2_normalize(train_embeddings[ep], axis = 1),
                      tf.math.l2_normalize(test_embeddings[ep], axis = 1))

    # pair_dist_l2 = pairwise_distances(train_embeddings[ep],
    #                  test_embeddings[ep])
    
    inverse_pair_dist = tf.math.softmax(1-pair_dist_l2).numpy()
    metrics[i, :] = [top_k_accuracy_score(y_true, inverse_pair_dist, k = 1),
                     top_k_accuracy_score(y_true, inverse_pair_dist, k = 2),
                     top_k_accuracy_score(y_true, inverse_pair_dist, k = 5)]



import matplotlib.pyplot as plt

plt.figure()
plt.plot(epochs_to_eval, metrics)
plt.legend(['Top-1', 'Top-2', 'Top-5'])
plt.title('')



