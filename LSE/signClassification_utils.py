# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:19:02 2022

@author: SNT
"""

import tensorflow as tf
import os
import numpy as np
import cv2

def _parse_video(fn):
    cap = cv2.VideoCapture(fn)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    
    list_of_frames = [None for _ in range(frame_count)]    
    frame_counter = 0
    while True:
        ret_val, frame = cap.read()
        if frame is None: break
        if cv2.waitKey(1) == 27:
            break  # esc to quit
        list_of_frames[frame_counter] = frame
        frame_counter += 1
    return list_of_frames, {'frame_count' : frame_count, 'fps' : fps}

def _parse_dataset(ds_folder):
    video_fns = [os.path.join(ds_folder,f) for f in os.listdir(ds_folder)]
    list_images = [None for _ in range(len(video_fns))]
    for i, fn in enumerate(video_fns):
        seq_of_images, video_info = _parse_video(fn) 
        list_images[i] = {'data' : seq_of_images, 'filename' : fn, 'info' : video_info}
    return list_images


def _serialize_data(data, output_folder = 'data/subset_tensors'):
    fn = data['filename']
    videoname = os.path.split(fn)[-1]
    outfn = os.path.join(output_folder,
                         videoname.replace('.', '_')+'.pickle')
    with open(outfn, 'wb') as f:
        pickle.dump(data,f)
        
        
def _deserialize_data(fn, exclude = []):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    if len(exclude) > 0:
        for d in exclude:
            del data[d]
    return data

def _get_hand_landmarks(model, img):
    result = model.process(img)
    if result.multi_handedness is None: # No hands detected
        return None
    
    left_hand_detected = any([True for i in result.multi_handedness if 'Left' in str(i)])
    right_hand_detected = any([True for i in result.multi_handedness if 'Right' in str(i)])
    
    left_hand_landmarks = np.zeros((21, 3))
    if left_hand_detected:
        left_idx = [i for i in range(len(result.multi_handedness))
                              if 'Left' in str(result.multi_handedness[i])][0]
        c = 0
        for lm in result.multi_hand_landmarks[left_idx].landmark:
            left_hand_landmarks[c,:] = [lm.x, lm.y, lm.z]
            c += 1

    right_hand_landmarks = np.zeros((21, 3))
    if right_hand_detected:
        right_idx = [i for i in range(len(result.multi_handedness))
                              if 'Right' in str(result.multi_handedness[i])][0]
        c = 0
        for lm in result.multi_hand_landmarks[right_idx].landmark:
            right_hand_landmarks[c,:] = [lm.x, lm.y, lm.z]
            c += 1

    return left_hand_landmarks, right_hand_landmarks

def _get_face_landmarks(model, img):
    result = model.process(img)
    if result.multi_face_landmarks is not None:
        result = result.multi_face_landmarks[0].landmark
        face_landmarks = np.zeros(shape = (len(result), 3))
        for i, lm in enumerate(result):
            face_landmarks[i,:] = [lm.x, lm.y, lm.z]
    else: 
        face_landmarks = np.zeros(shape=(468, 3))
    return face_landmarks


def _get_pose_landmarks(pose, img):
    result = pose.process(img)
    if result.pose_landmarks is not None:
        result = result.pose_landmarks.landmark
        pose_landmarks = np.zeros(shape = (len(result), 3))
        for i, lm in enumerate(result):
            pose_landmarks[i,:] = [lm.x, lm.y, lm.z]
    else:
        pose_landmarks = np.zeros(shape=(33, 3))
    return pose_landmarks

def _pad_img_batch(img_batch, iw, ih, lmax = None):
    lens = [s.shape[0] for s in img_batch]
    lmax = max(lens) if lmax == None else lmax
    
    img_batch = [ s for s,l in zip(img_batch,lens)]
    return [tf.concat([s, tf.zeros([lmax-l, iw,ih,3], dtype = 'uint8')], axis = 0)
            if lmax-l > 0 else s[:lmax,::] for s,l in zip(img_batch,lens)]


def _pad_kps_batch(kps_batch,lmax = None):
    lens = [s.shape[0] for s  in kps_batch]
    lmax = max(lens) if lmax == None else lmax
    kps_batch = [s for s,l in zip(kps_batch,lens)]
    
    return [tf.concat([s, tf.zeros((lmax-l, s.shape[1], s.shape[2]), dtype = tf.float64)], axis = 0)
            if lmax-l > 0 else s[:lmax,::] for s,l in zip(kps_batch,lens)]    
    
    # return [tf.concat([s]+[s[-1:] for _ in range(lmax-l)], axis = 0)
    #         if lmax-l > 0 else s[:lmax,::] for s,l in zip(kps_batch,lens)]

def _parse_dataset_hands(ds_folder, retun_filenames = False):
    classes = os.listdir(ds_folder)
    n_samples = []
    labels = []
    for c in classes:
        sample_folders = os.listdir(os.path.join(ds_folder, c))
        n_samples.append(len(sample_folders))
        labels += [c for _ in range(len(sample_folders))]

    list_hands = [None for _ in range(sum(n_samples))]
    counter = 0
    fn_list = []
    for c, n_s in zip(classes, n_samples):
        for n in range(1, n_s+1):
            seq_of_hands = tf.stack([np.genfromtxt(os.path.join(ds_folder, c, str(n), i),
                                          skip_header = True, delimiter = ',')
                        for i in os.listdir(os.path.join(ds_folder, c, str(n))) if '.csv' in i])
            if retun_filenames: fn_list.append([os.path.join(ds_folder, c, str(n), i)
                                                for i in os.listdir(os.path.join(ds_folder, c, str(n)))
                                                if '.csv' in i])
            list_hands[counter] = seq_of_hands
            counter += 1
    if retun_filenames: return list_hands, labels, fn_list
    return list_hands, labels



import pickle
def _parse_dataset_hands_heatmaps(ds_folder):
    classes = os.listdir(ds_folder)
    n_samples = []
    labels = []
    for c in classes:
        sample_folders = os.listdir(os.path.join(ds_folder, c))
        n_samples.append(len(sample_folders))
        labels += [c for _ in range(len(sample_folders))]
    list_heatmaps = [None for _ in range(sum(n_samples))]
    counter = 0
    for c, n_s in zip(classes, n_samples):
        for n in range(1, n_s+1):
            with open(os.path.join(ds_folder, c, str(n), 'hands.heatmaps'), 'rb') as f:
                list_heatmaps[counter] = pickle.load(f)
            counter += 1
    return list_heatmaps, labels




def generate_heatmap(xland, yland, iw, ih, width = 1):
    xland = int(xland*ih) if xland < 1.0 else int(xland)
    yland = int(yland*iw) if yland < 1.0 else int(yland)
    hm = np.zeros((iw, ih))
    if xland+yland > 0:
        if width == 0:
            hm[yland, xland] = 1
        else:
            hm[yland-width:yland+width, xland-width:xland+width] = 1
    return hm

def landmarks2heatmaps(landmarks, iw, ih, width = 1):
    landmarks = tf.reshape(landmarks, ((2*landmarks.shape[0], 2)))  
    hms = [generate_heatmap(l[0], l[1],
                     iw, ih, width = width) for l in landmarks.numpy()]
    return np.stack(hms)

#%%
import itertools

compute_mag = lambda x, a : np.sqrt(np.sum(x*x, axis = a))
def polar_distances(array):
    dims = array.shape
    gravity_center = array.mean(axis = 1)
    vectors = np.broadcast_to(np.expand_dims(gravity_center, axis = 1), (dims[0], dims[1], 3))-array

    inner_dist = np.stack([np.arctan(vectors[:,:,0]/(vectors[:,:,1]+1e-10)),
                                  np.arctan(vectors[:,:,2]/(vectors[:,:,1]+1e-10)),
                                  compute_mag(vectors, -1)], axis = -1)
    return inner_dist, gravity_center 
            
def pair_polar_distances(x):
    dims = x.shape
    indexes = range(dims[-1])
    diff = np.stack([x[:,:,i]-x[:,:,j]
                     for i,j in itertools.combinations(indexes, 2)], axis = -1)
    # (SeqLen, Npairs, Azimut-elevation-modulo)
    d = np.stack([np.arctan(diff[:,0,:]/(diff[:,1,:]+1e-10),),
                  np.arctan(diff[:,2,:]/(diff[:,1,:]+1e-10)),
                  compute_mag(diff, 1)], axis = -1)
    return d 
    