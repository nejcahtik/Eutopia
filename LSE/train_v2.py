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
nepochs = 30

lr = 1e-5
# aug_p = .3 if '-p' not in sys.argv else float(sys.argv[sys.argv.index('-p')+1]) 
dataset_folder = 'data/features_test1' if '-d' not in sys.argv else float(sys.argv[sys.argv.index('-d')+1]) 
# classifier_model = 'h1' if '-c' not in sys.argv else float(sys.argv[sys.argv.index('-c')+1]) 

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

train_rhands = [d['mp_hands (R,L)'][0] for d in train_data]
train_rhands = list(map(zero_to_one, train_rhands))                            # MAKE NON DETECTED KP TO BE -1

test_rhands = [d['mp_hands (R,L)'][0] for d in test_data]
test_rhands = list(map(zero_to_one, test_rhands))                              # MAKE NON DETECTED KP TO BE -1

train_lhands = [d['mp_hands (R,L)'][1] for d in train_data]
train_lhands = list(map(zero_to_one, train_lhands))                            # MAKE NON DETECTED KP TO BE -1

test_lhands = [d['mp_hands (R,L)'][1] for d in test_data]
test_lhands = list(map(zero_to_one, test_lhands))                              # MAKE NON DETECTED KP TO BE -1

train_face = [d['mp_face'] for d in train_data]
train_face = list(map(zero_to_one, train_face))                                # MAKE NON DETECTED KP TO BE -1

test_face = [d['mp_face'] for d in test_data]
test_face = list(map(zero_to_one, test_face))                                  # MAKE NON DETECTED KP TO BE -1

train_pose = [d['mp_pose'] for d in train_data]
train_pose = list(map(zero_to_one, train_pose))                                # MAKE NON DETECTED KP TO BE -1

test_pose = [d['mp_pose'] for d in test_data]
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
    seq_batch_3 = [seq_aug_layer(t, training = True) for t in seq_batch_2]
    return seq_batch_3


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''                             CREATING MODEL                              '''
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def flat_last_dims(x):
    x_shape = x.shape
    return tf.reshape(x, x_shape[:-2]+(x_shape[-2]*x_shape[-1],))

hand_processor = tf.keras.Sequential([
    #tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, 'relu')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64))])

face_processor = tf.keras.Sequential([
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, 'relu')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64))])

pose_processor = tf.keras.Sequential([
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, 'relu')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64))])

lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32)
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
def train_step(input1, input2):
    enc_ori = forward(input1)
    enc_aug = forward(input2)
    
    enc_l2_ori = tf.math.l2_normalize(enc_ori, axis = 1)
    enc_l2_aug = tf.math.l2_normalize(enc_aug, axis = 1)
    
    l_pos = sim_func_dim1(enc_l2_ori, enc_l2_aug)
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
    epoch_loss = 0.0
    with tf.GradientTape() as tape:
        for ib in idx_batches:
            batch = [[train_rhands[i] for i in ib],
                 [train_lhands[i] for i in ib]]
            
            # AUGMENTATION
            aligned_seq_batch = list(zip(batch[0], batch[1]))
            batch_aug = augment_sequences(aligned_seq_batch)
            batch_aug = list(zip(*batch_aug))
           
            # PAD
            max_batch_len = max([t.shape[0] for t in batch[0]+list(batch_aug[0])])
            batch = tf.convert_to_tensor([[np.concatenate([m, np.zeros((max_batch_len-m.shape[0], m.shape[1],
                                              m.shape[2]))], 0) for m in b] for b in batch])
            batch_aug = tf.convert_to_tensor([[np.concatenate([m, np.zeros((max_batch_len-m.shape[0], m.shape[1],
                                              m.shape[2]))], 0) for m in b] for b in batch_aug])
        
            # TRAIN STEP
            if any([not hand_processor.built, not lstm_model.built]):
                _ = forward(batch)
    

            t0 = time.time()
            step_wise_loss = train_step(batch, batch_aug)
            print(time.time()-t0, step_wise_loss)
            epoch_loss += step_wise_loss
 
   
    trainable_vars = hand_processor.trainable_variables + \
                           lstm_model.trainable_variables
    gradients = tape.gradient(epoch_loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))    

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
    
    test_embeddings.append(ep_test_embeddings.numpy())
    train_embeddings.append(ep_train_embeddings.numpy())

    metrics[ep-1 , 0] = np.mean(tf.keras.metrics.top_k_categorical_accuracy(ep_train_embeddings, ep_test_embeddings, k = 1))
    metrics[ep-1 , 1] = np.mean(tf.keras.metrics.top_k_categorical_accuracy(ep_train_embeddings, ep_test_embeddings, k = 2))
    metrics[ep-1 , 2] = np.mean(tf.keras.metrics.top_k_categorical_accuracy(ep_train_embeddings, ep_test_embeddings, k = 5))
    metrics[ep-1 , 3] = np.mean(tf.keras.metrics.top_k_categorical_accuracy(ep_train_embeddings, ep_test_embeddings, k = 10))
    metrics[ep-1 , 4] = tf.keras.losses.CosineSimilarity()(ep_train_embeddings, ep_test_embeddings)

#%%

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


sel_epoch = 4
tsne = TSNE(n_components=2)
new_features_train  = tsne.fit_transform(train_embeddings[sel_epoch])
plt.figure()
plt.scatter(new_features_train[:,0], new_features_train[:,1])


new_features_test  = tsne.fit_transform(test_embeddings[sel_epoch])
plt.figure()
plt.scatter(new_features_test[:,0], new_features_test[:,1])


new_features_both  = tsne.fit_transform(np.concatenate([train_embeddings[sel_epoch],
                                                        test_embeddings[sel_epoch]]))
plt.figure()
plt.scatter(new_features_both[:n_test_samples,0], new_features_both[:n_test_samples,1])
plt.scatter(new_features_both[n_test_samples:,0], new_features_both[n_test_samples:,1])






#%%
from sklearn.decomposition import PCA


sel_epoch = 4
pca = PCA(n_components=2)
new_features_train  = pca.fit_transform(train_embeddings[sel_epoch])

import matplotlib.pyplot as plt
plt.scatter(new_features_train[:,0], new_features_train[:,1])


new_features_test  = pca.fit_transform(test_embeddings[sel_epoch])

import matplotlib.pyplot as plt
plt.scatter(new_features_test[:,0], new_features_test[:,1])

#%%
from sklearn.metrics.pairwise import cosine_similarity
sel_epoch = 0

cosine_sim = cosine_similarity(train_embeddings[sel_epoch],
                  test_embeddings[sel_epoch]) 



#%%













