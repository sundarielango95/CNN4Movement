# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:56:09 2020

@author: Sundari Elango
"""

from keras.models import Input
from keras.layers import Concatenate
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model#, load_model
from keras.callbacks import ModelCheckpoint
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from random import shuffle
import random
import pickle

def feature_map_percent_lesion(X):  

    if percent == 0:
        return X

    les_arr_fm = ft_arr[lay_num]
    # Set the selected nodes' activation to 0.
    les_fm = overlap_strength*np.ones((1,X.shape[1],X.shape[2],X.shape[3]),dtype=np.float32)
    for ft in range(len(les_arr_fm)):
        lesion_array_fm = les_arr_fm[ft]
        for i_j in lesion_array_fm:
            les_fm[:,i_j[0],i_j[1],ft] = 0
   
    les_fm = tf.convert_to_tensor(les_fm,dtype=tf.float32)
    X = tf.math.multiply(X,les_fm)
       
    return X

def fully_connected_percent_lesion(X):
    
    if percent == 0:
        return X
    
    les = overlap_strength*np.ones((1,30),dtype=np.float32)
    for fc in range(len(fc_arr)):
        i_j = fc_arr[fc]
        les[:,i_j] = 0
        
    les = tf.convert_to_tensor(les,dtype=tf.float32)
    X = tf.math.multiply(X,les)
    return X

def page_dr_shepherd_FC_left(X):
    X = keras.activations.sigmoid(X)
    if les_ind_l == []:
        return X
    les = np.ones((1,30),dtype=np.float32)
    for ii in range(len(les_ind_l)):
        j = les_ind_l[ii]
        les[:,j] = 0
    les = tf.convert_to_tensor(les,dtype=tf.float32)
    X = tf.math.multiply(X,les)
    return X

def page_dr_shepherd_FC_right(X):
    X = keras.activations.sigmoid(X)
    if les_ind_r == []:
        return X
    les = np.ones((1,30),dtype=np.float32)
    for ii in range(len(les_ind_r)):
        j = les_ind_r[ii]
        les[:,j] = 0
    les = tf.convert_to_tensor(les,dtype=tf.float32)
    X = tf.math.multiply(X,les)
    return X

def CCM():
    
    global lay_num
    
    # defining input and parameters of network
    img_shape=(1,45,45,3,1)
    l2_reg=0
    input_left = Input(shape=(45,45,3))
    input_right= Input(shape=(45,45,3))
    hc_l_sheet = Input(shape=(22,22,1))
    hc_r_sheet = Input(shape=(22,22,1))
#    com_sheet = Input(shape=(22,22,1))
    
#    eye = Concatenate(axis=-1)([input_left, input_right])
    
    # Channelised conv layers
    # Left Eye
    left_vc1 = Conv2D(2, (5, 5), strides=(1,1), input_shape=img_shape,padding='same', kernel_regularizer=l2(l2_reg))(input_left)
    left_vc1 = BatchNormalization()(left_vc1)
    left_vc1 = Activation('relu')(left_vc1)
    left_vc1l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(left_vc1) # direct activity
#    VC1_l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(left_vc1)
    lay_num = 0
    left_vc1r = Activation(feature_map_percent_lesion)(left_vc1l) # proportional activity
    # Right Eye
    right_vc1 = Conv2D(2, (5, 5), strides=(1,1), input_shape=img_shape,padding='same', kernel_regularizer=l2(l2_reg))(input_right)
    right_vc1 = BatchNormalization()(right_vc1)
    right_vc1 = Activation('relu')(right_vc1)
    right_vc1r = MaxPooling2D(pool_size=(2,2), strides=(2,2))(right_vc1) # direct activity
#    VC1_r = MaxPooling2D(pool_size=(2,2), strides=(2,2))(right_vc1)
    right_vc1l = Activation(feature_map_percent_lesion)(right_vc1r) # proportional activity
    
    # Combining output from stage 1 
    VC1_l = Concatenate(axis=-1)([left_vc1l, right_vc1l]) # input to left 
    VC1_r = Concatenate(axis=-1)([right_vc1r, left_vc1r]) # input to right

    # Left layer 2
    left_vc2 = Conv2D(4, (5, 5), strides=(1,1),padding='same')(VC1_l)
    left_vc2 = BatchNormalization()(left_vc2)
#    left_vc2l = Activation('relu')(left_vc2)
    VC2l = Activation('relu')(left_vc2)
    lay_num = 1
#    left_vc2r = Activation(feature_map_percent_lesion)(left_vc2l)
    # Right layer 2
    right_vc2 = Conv2D(4, (5, 5), strides=(1,1),padding='same')(VC1_r)
    right_vc2 = BatchNormalization()(right_vc2)
#    right_vc2r = Activation('relu')(right_vc2)
    VC2r = Activation('relu')(right_vc2)
#    right_vc2l = Activation(feature_map_percent_lesion)(right_vc2r)
    
    # Combining output from stage 2 and the Modality of hand movement inputs 
#    VC2_l = Concatenate(axis=-1)([left_vc2l, right_vc2l, hc_l_sheet, hc_r_sheet,com_sheet])# 
#    VC2_r = Concatenate(axis=-1)([right_vc2r, left_vc2r, hc_l_sheet, hc_r_sheet,com_sheet])#
    
    VC2_l = Concatenate(axis=-1)([VC2l, hc_l_sheet, hc_r_sheet])# 
    VC2_r = Concatenate(axis=-1)([VC2r, hc_l_sheet, hc_r_sheet])#
   
    # Left layer 3
    left_vc3 = Conv2D(8, (5, 5), strides=(1,1),padding='same')(VC2_l)
    left_vc3 = BatchNormalization()(left_vc3)
#    left_vc3l = Activation('relu')(left_vc3)
    VC3_l = Activation('relu')(left_vc3)
    lay_num = 2
#    left_vc3r = Activation(feature_map_percent_lesion)(left_vc3l)
    # Right layer 3
    right_vc3 = Conv2D(8, (5, 5), strides=(1,1),padding='same')(VC2_r)
    right_vc3 = BatchNormalization()(right_vc3)
#    right_vc3r = Activation('relu')(right_vc3)
    VC3_r = Activation('relu')(right_vc3)
#    right_vc3l = Activation(feature_map_percent_lesion)(right_vc3r)
    
    # Combining output from stage 3
#    VC3_l = Concatenate(axis=-1)([left_vc3l, right_vc3l])
#    VC3_r = Concatenate(axis=-1)([right_vc3r, left_vc3r])
    
    # Left layer 4    
    left_vc4 = Conv2D(4, (5, 5), strides=(1,1),padding='same')(VC3_l)
    left_vc4 = BatchNormalization()(left_vc4)
#    left_vc4l = Activation('relu')(left_vc4)
    VC4_l = Activation('relu')(left_vc4)
    lay_num = 3
#    left_vc4r = Activation(feature_map_percent_lesion)(left_vc4l)
    # Right layer 4
    right_vc4 = Conv2D(4, (5, 5), strides=(1,1),padding='same')(VC3_r)
    right_vc4 = BatchNormalization()(right_vc4)
#    right_vc4r = Activation('relu')(right_vc4)
    VC4_r = Activation('relu')(right_vc4)
#    right_vc4l = Activation(feature_map_percent_lesion)(right_vc4r)
    
    # Combining output from stage 4
#    VC4_l = Concatenate(axis=-1)([left_vc4l, right_vc4l])
#    VC4_r = Concatenate(axis=-1)([right_vc4r, left_vc4r])
    
    # Left layer 5
    left_vc5 = Conv2D(2, (5, 5), strides=(1,1),padding='same')(VC4_l)
    left_vc5 = BatchNormalization()(left_vc5)
    left_vc5l = Activation('relu')(left_vc5)
    lay_num = 4
#    left_vc5r = Activation(feature_map_percent_lesion)(left_vc5l)
    # Right layer 5
    right_vc5 = Conv2D(2, (5, 5), strides=(1,1),padding='same')(VC4_r)
    right_vc5 = BatchNormalization()(right_vc5)
    right_vc5r = Activation('relu')(right_vc5)
#    right_vc5l = Activation(feature_map_percent_lesion)(right_vc5r)
    
    # Combining ouput from stage 5
#    VC5_l = Concatenate(axis=-1)([left_vc5l, right_vc5l])
#    VC5_l = Flatten()(VC5_l)
    VC5_l = Flatten()(left_vc5l)
#    VC5_r = Concatenate(axis=-1)([right_vc5r, left_vc5r])
#    VC5_r = Flatten()(VC5_r)
    VC5_r = Flatten()(right_vc5r)
    
    # Channelised fc layers
    # Left layer 1
    left_mc1l = Dense(50, activation = 'sigmoid')(VC5_l)
    left_mc1r = Activation(fully_connected_percent_lesion)(left_mc1l)
#    MC1_l = Dense(30, activation = page_dr_shepherd_FC_left)(VC5_l)
    # Right layer 1 
    right_mc1r = Dense(50, activation = 'sigmoid')(VC5_r)
    right_mc1l = Activation(fully_connected_percent_lesion)(right_mc1r)
#    MC1_r = Dense(30, activation = page_dr_shepherd_FC_right)(VC5_r)
    # Combining output from layer 1
    MC1_l = Concatenate(axis=-1)([left_mc1l,right_mc1l])
    MC1_r = Concatenate(axis=-1)([right_mc1r,left_mc1r])
    # Output layer
    # Left
    MC2_l = Dense(30, activation = page_dr_shepherd_FC_left)(MC1_l)
    # Right
    MC2_r = Dense(30, activation = page_dr_shepherd_FC_right)(MC1_r)
    
    left_spinal = Dense(6, activation = 'sigmoid')(MC2_l)
    
    right_spinal = Dense(6, activation = 'sigmoid')(MC2_r)
    
    spinal_activation = Concatenate(axis=-1)([left_spinal,right_spinal])
   
    spinal_cord = Model([input_left,input_right,hc_l_sheet,hc_r_sheet],spinal_activation)
 
    return spinal_cord
    
def input_load_images(pdir,ind,length):
    images = []
    for i in range(length):
        print("LOADING", pdir+"/"+str(int(i+1))+"_"+str(ind)+".png")
        path = pdir+"/"+str(int(i+1))+"_"+str(ind)+".png"
        img = cv2.imread(path)
        if np.any(img == None):
            continue
        images.append(np.array(img, dtype=np.uint8))
    images=np.array(images)
    return images

def desired_output_array_generator(dataset,Y_col):
    Y=dataset[:,Y_col]
    return Y

def train_test_shuffle_split(I1,I2,I3,I4,Y):
    I1 = list(I1)
    I2 = list(I2)
    I3 = list(I3)
    I4 = list(I4)
#    I5 = list(I5)
    Y = list(Y)
    inp1 = []
    inp2 = []
    inp3 = []
    inp4 = []
#    inp5 = []
    y = []
   
    ind = list(range(len(Y)))
    shuffle(ind)
   
    for i in ind:
        inp1.append(I1[i])
        inp2.append(I2[i])
        inp3.append(I3[i])
        inp4.append(I4[i])
#        inp5.append(I5[i])
        y.append(Y[i])
   
    inp1 = np.array(inp1)
    inp2 = np.array(inp2)
    inp3 = np.array(inp3)
    inp4 = np.array(inp4)
#    inp5 = np.array(inp5)
    y = np.array(y)
   
    return inp1,inp2,inp3,inp4,y

num_object = ["one_obj","two_obj"]
excel_directories = ["01","10","11","00"]
#ed = "01"
#cc = 0
nEpisodes = 1
#healthy_score = np.zeros((nEpisodes,1))

input_images_left_all = []
input_images_right_all = []
hc_l_inputs_all = []
hc_r_inputs_all = []
com_inputs_all = []
MN_all = []

sizes = 3
for size in range(sizes):
    for num_obj in num_object:
        # Load Left eye inputs - train data
        directory_left = "D:/MS/Codes/Python/CNN_model_Stroke/Inputs/INPUTS TWO BITS/Different_sizes_n_colors/Size_"+str(size+1)+"_"+num_obj
        alphabet = 'a'
        input_images_left = input_load_images(directory_left,alphabet,1174)
        input_images_left = np.reshape(input_images_left,[len(input_images_left),45,45,3])
        
        # Load Right eye inputs - train data
        directory_right = "D:/MS/Codes/Python/CNN_model_Stroke/Inputs/INPUTS TWO BITS/Different_sizes_n_colors/Size_"+str(size+1)+"_"+num_obj
        alphabet = 'b'
        input_images_right = input_load_images(directory_right,alphabet,1174)
        input_images_right = np.reshape(input_images_right,[len(input_images_right),45,45,3])
        
        for excel_directory in excel_directories:
            # train data
            input_images_left_mod = []
            input_images_right_mod = []
            hc_l_inputs_mod = []
            hc_r_inputs_mod = []
            com_inputs_mod = []
            MN_mod = []
            arm_data = pd.read_excel("D:/MS/Codes/Python/CNN_model_Stroke/Excel Sheet/2_bit_Modality/Left_n_Right/Train_Data/"+num_obj+"_"+excel_directory+".xlsx")
            arm_data = np.array(arm_data)
            input_images_left_num_obj = input_images_left
            input_images_right_num_obj = input_images_right
            MN_left = desired_output_array_generator(arm_data,[7,8,9,10,11,12])#MN activations
            MN_right = desired_output_array_generator(arm_data,[19,20,21,22,23,24])#MN activations
            hc_l = desired_output_array_generator(arm_data,[25])
            hc_r = desired_output_array_generator(arm_data,[26])
#            com = desired_output_array_generator(arm_data,[27])
            MN = np.concatenate((MN_left,MN_right),axis=-1)
            indices = desired_output_array_generator(arm_data,[0]) 
            hc_l_inputs = np.zeros((hc_l.shape[0],22,22,1))
            hc_r_inputs = np.zeros((hc_r.shape[0],22,22,1))
#            com_inputs = np.zeros((com.shape[0],22,22,1))
            
            for i in range(hc_l.shape[0]):
                hc_l_inputs[i,:,:,:] = np.ones((22,22,1))*hc_l[i,:]
                hc_r_inputs[i,:,:,:] = np.ones((22,22,1))*hc_r[i,:]
#                com_inputs[i,:,:,:] = np.ones((22,22,1))*com[i,:]
                
            input_images_left_mod,input_images_right_mod,hc_l_inputs_mod,hc_r_inputs_mod,MN_mod = train_test_shuffle_split(input_images_left_num_obj,input_images_right_num_obj,hc_l_inputs,hc_r_inputs,MN)
            input_images_left_all.append(input_images_left_mod)
            input_images_right_all.append(input_images_right_mod)
            hc_l_inputs_all.append(hc_l_inputs_mod)
            hc_r_inputs_all.append(hc_r_inputs_mod)
            com_inputs_all.append(com_inputs_mod)
            MN_all.append(MN_mod)
            
       
input_images_left_all = np.array([ item for elem in input_images_left_all for item in elem])
input_images_right_all = np.array([ item for elem in input_images_right_all for item in elem])
hc_l_inputs_all = np.array([ item for elem in hc_l_inputs_all for item in elem])
hc_r_inputs_all = np.array([ item for elem in hc_r_inputs_all for item in elem])
#com_inputs_all = np.array([ item for elem in com_inputs_all for item in elem])
MN_all = np.array([ item for elem in MN_all for item in elem])

les_ind_l = []
les_ind_r = []
nn = 5
#for i_p in range(7):
#    cc = 0
for nEp in range(nEpisodes):
#    for percent in percentages:
    percent = 0
    overlap_strength = 1
    fm_num = [2,4,8,4,2]
    fm_dim = np.array([[22,22],[22,22],[22,22],[22,22],[22,22]])
    # creates n% lesion in every feature map of a given layer
    ft_arr = []
    # For each feature map, choose n% of nodes randomly to be lesioned.
    for f_n in range(len(fm_num)):
        fm_arr =[]
        for ft in range(fm_num[f_n]):
            arr=[]
            for i in range(fm_dim[f_n,0]):
                for j in range(fm_dim[f_n,1]):
                    arr.append([i,j]) # creating m*n array with positions of the feature map.
            n_of_nodes = int(len(arr)*percent/100)
            nodes = random.sample(range(len(arr)),n_of_nodes) # randomly choosing elements to be zero.  
            fm_arr.append([arr[n] for n in nodes])
        ft_arr.append(fm_arr)
    fc_arr = []
    arr_fc = []
    for j in range(50):
        arr_fc.append([j])
    n_fc = int((len(arr_fc)*percent)/100)
    nodes_fc = random.sample(range(len(arr_fc)),n_fc)
    fc_arr.append([arr_fc[n_fc] for n_fc in nodes_fc])
    input_images_left_train,input_images_right_train,hc_l_inputs_train,hc_r_inputs_train,MN_train = train_test_shuffle_split(input_images_left_all,input_images_right_all,hc_l_inputs_all,hc_r_inputs_all,MN_all)    
    # Build and train model
    Motor_Cortex = CCM()
    adam=keras.optimizers.Adam(lr=0.0001, beta_1=0.999, beta_2=0.999, amsgrad=False)
    Motor_Cortex.compile(loss="mse", optimizer=adam)
    filepath ="D:/V2_from_PC/V2/Models/Healthy_Models/CCM/Varying Overlaps/2_bit_modality_Healthy_Subject_"+str(nEp+nn)+"_try.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint1]   
    Motor_Cortex.fit([input_images_left_train,input_images_right_train,hc_l_inputs_train,hc_r_inputs_train], MN_train, epochs = 30,batch_size = 10,validation_split = 0.2,callbacks=callbacks_list)
    with open("D:/V2_from_PC/V2/Models/Healthy_Models/CCM/Varying Overlaps/2_bit_modality_Healthy_Subject_"+str(nEp+nn)+"_FM_indices_try.txt", "wb") as fp:
            pickle.dump(ft_arr, fp)
    with open("D:/V2_from_PC/V2/Models/Healthy_Models/CCM/Varying Overlaps/2_bit_modality_Healthy_Subject_"+str(nEp+nn)+"_FC_indices_try.txt", "wb") as fp:
            pickle.dump(fc_arr, fp)
#        if i_p == 0:
#            healthy = load_model("D:/V2_from_PC/V2/Models/Healthy_Models/CCM/Varying Overlaps/Ip to Op/no_overlapped_layers_Healthy_Subject_"+str(nEp+1)+"_100_percent_overlap.h5",custom_objects={'feature_map_percent_lesion': feature_map_percent_lesion,'fully_connected_percent_lesion': fully_connected_percent_lesion,'page_dr_shepherd_FC_left':page_dr_shepherd_FC_left , 'page_dr_shepherd_FC_right':page_dr_shepherd_FC_right})
#        else:
#    healthy = load_model("D:/V2_from_PC/V2/Models/Healthy_Models/CCM/BMT_Cong_overlap_at_layer_4_Healthy_Subject_"+str(nEp+1)+"_100_percent_overlap.h5",custom_objects={'feature_map_percent_lesion': feature_map_percent_lesion,'fully_connected_percent_lesion': fully_connected_percent_lesion,'page_dr_shepherd_FC_left':page_dr_shepherd_FC_left , 'page_dr_shepherd_FC_right':page_dr_shepherd_FC_right})
#    wts = healthy.get_weights()
#    patient = CCM()
#    patient.set_weights(wts)
#    healthy_score[nEp,cc] = healthy.evaluate([input_images_left_all, input_images_right_all, hc_l_inputs_all, hc_r_inputs_all], MN_all)   
#        print(nEp,i_p)
#    cc = cc+1

#            