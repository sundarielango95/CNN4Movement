# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 13:30:53 2020

@author: Sundari Elango
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:49:19 2020

@author: Sundari Elango
"""

import keras
import tensorflow as tf
from keras.models import load_model
from keras.models import Input
from keras.layers import Concatenate
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model
import numpy as np
import cv2
import math
from math import cos,sin
import pandas as pd
import pickle

def feature_map_percent_lesion(X):  
    if percent == 0:
        return X
    les_arr_fm = ft_arr[lay_num]
    # Set the selected nodes' activation to 0.
    les_fm = np.ones((1,X.shape[1],X.shape[2],X.shape[3]),dtype=np.float32)
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
    les = np.ones((1,30),dtype=np.float32)
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
    lay_num = 0
    left_vc1r = Activation(feature_map_percent_lesion)(left_vc1l) # proportional activity
    # Right Eye
    right_vc1 = Conv2D(2, (5, 5), strides=(1,1), input_shape=img_shape,padding='same', kernel_regularizer=l2(l2_reg))(input_right)
    right_vc1 = BatchNormalization()(right_vc1)
    right_vc1 = Activation('relu')(right_vc1)
    right_vc1r = MaxPooling2D(pool_size=(2,2), strides=(2,2))(right_vc1) # direct activity
    right_vc1l = Activation(feature_map_percent_lesion)(right_vc1r) # proportional activity
    
    # Combining output from stage 1 
    VC1_l = Concatenate(axis=-1)([left_vc1l, right_vc1l]) # input to left 
    VC1_r = Concatenate(axis=-1)([left_vc1r, right_vc1r]) # input to right

    # Left layer 2
    left_vc2 = Conv2D(4, (5, 5), strides=(1,1),padding='same')(VC1_l)
    left_vc2 = BatchNormalization()(left_vc2)
    left_vc2l = Activation('relu')(left_vc2)
    lay_num = 1
    left_vc2r = Activation(feature_map_percent_lesion)(left_vc2l)
    # Right layer 2
    right_vc2 = Conv2D(4, (5, 5), strides=(1,1),padding='same')(VC1_r)
    right_vc2 = BatchNormalization()(right_vc2)
    right_vc2r = Activation('relu')(right_vc2)
    right_vc2l = Activation(feature_map_percent_lesion)(right_vc2r)
    
    # Combining output from stage 2 and the Modality of hand movement inputs 
    VC2_l = Concatenate(axis=-1)([left_vc2l, right_vc2l,hc_l_sheet,hc_r_sheet]) 
    VC2_r = Concatenate(axis=-1)([left_vc2r, right_vc2r,hc_l_sheet,hc_r_sheet])
   
    # Left layer 3
    left_vc3 = Conv2D(8, (5, 5), strides=(1,1),padding='same')(VC2_l)
    left_vc3 = BatchNormalization()(left_vc3)
    left_vc3l = Activation('relu')(left_vc3)
    lay_num = 2
    left_vc3r = Activation(feature_map_percent_lesion)(left_vc3l)
    # Right layer 3
    right_vc3 = Conv2D(8, (5, 5), strides=(1,1),padding='same')(VC2_r)
    right_vc3 = BatchNormalization()(right_vc3)
    right_vc3r = Activation('relu')(right_vc3)
    right_vc3l = Activation(feature_map_percent_lesion)(right_vc3r)
    
    # Combining output from stage 3
    VC3_l = Concatenate(axis=-1)([left_vc3l, right_vc3l])
    VC3_r = Concatenate(axis=-1)([left_vc3r, right_vc3r])
    
    # Left layer 4    
    left_vc4 = Conv2D(4, (5, 5), strides=(1,1),padding='same')(VC3_l)
    left_vc4 = BatchNormalization()(left_vc4)
    left_vc4l = Activation('relu')(left_vc4)
    lay_num = 3
    left_vc4r = Activation(feature_map_percent_lesion)(left_vc4l)
    # Right layer 4
    right_vc4 = Conv2D(4, (5, 5), strides=(1,1),padding='same')(VC3_r)
    right_vc4 = BatchNormalization()(right_vc4)
    right_vc4r = Activation('relu')(right_vc4)
    right_vc4l = Activation(feature_map_percent_lesion)(right_vc4r)
    
    # Combining output from stage 4
    VC4_l = Concatenate(axis=-1)([left_vc4l, right_vc4l])
    VC4_r = Concatenate(axis=-1)([left_vc4r, right_vc4r])
    
    # Left layer 5
    left_vc5 = Conv2D(2, (5, 5), strides=(1,1),padding='same')(VC4_l)
    left_vc5 = BatchNormalization()(left_vc5)
    left_vc5l = Activation('relu')(left_vc5)
    lay_num = 4
    left_vc5r = Activation(feature_map_percent_lesion)(left_vc5l)
    # Right layer 5
    right_vc5 = Conv2D(2, (5, 5), strides=(1,1),padding='same')(VC4_r)
    right_vc5 = BatchNormalization()(right_vc5)
    right_vc5r = Activation('relu')(right_vc5)
    right_vc5l = Activation(feature_map_percent_lesion)(right_vc5r)
    
    # Combining ouput from stage 5
    VC5_l = Concatenate(axis=-1)([left_vc5l, right_vc5l])
    VC5_l = Flatten()(VC5_l)
    VC5_r = Concatenate(axis=-1)([left_vc5r, right_vc5r])
    VC5_r = Flatten()(VC5_r)
    
    # Channelised fc layers
    # Left layer 1
    left_mc1l = Dense(30, activation = page_dr_shepherd_FC_left)(VC5_l)
    left_mc1r = Activation(fully_connected_percent_lesion)(left_mc1l)
    # Right layer 1 
    right_mc1r = Dense(30, activation = page_dr_shepherd_FC_right)(VC5_r)
    right_mc1l = Activation(fully_connected_percent_lesion)(right_mc1r)
    
    # Combining output from layer 1
    MC1_l = Concatenate(axis=-1)([left_mc1l,right_mc1l])
    MC1_r = Concatenate(axis=-1)([left_mc1r,right_mc1r])
    # Output layer
    # Left
    left_spinal = Dense(6, activation = 'sigmoid')(MC1_l)
    # Right
    right_spinal = Dense(6, activation = 'sigmoid')(MC1_r)
    
    spinal_activation = Concatenate(axis=-1)([left_spinal,right_spinal])
   
    spinal_cord = Model([input_left,input_right,hc_l_sheet,hc_r_sheet],spinal_activation)
 
    return spinal_cord

def end_effector_calculator(shoulder,mn):
    theta_s1 = (mn[0,0] - mn[0,1])*(math.pi/2) + (math.pi/2)
    theta_s2 = (mn[0,2] - mn[0,3])*(math.pi/2) + (math.pi/2)
    theta_e = (mn[0,4] - mn[0,5])*(math.pi/2) + (math.pi/2)
    
    hand_start = np.array([0,0,-((L_s-a_e)+L_e)])
    elbow_start = np.array([0,0,-(L_s-a_e)])
       
    rot_e = np.array([[cos(theta_e),0,-sin(theta_e)], [0,1,0], [sin(theta_e),0,cos(theta_e)]])
    rot_s1 = np.array([[cos(theta_s1),0,-sin(theta_s1)], [0,1,0], [sin(theta_s1),0,cos(theta_s1)]])
    rot_s2 = np.array([[cos(theta_s2),-sin(theta_s2),0], [sin(theta_s2),cos(theta_s2),0], [0,0,1]])
    
    end_effector_el_rot = np.matmul((hand_start-elbow_start),rot_e) + (elbow_start)
    end_effector_s1_rot = np.matmul(end_effector_el_rot,rot_s1)
    end_effector_s2_rot = np.matmul(end_effector_s1_rot,rot_s2)
    
    end_effector = np.reshape(end_effector_s2_rot,[3])+shoulder
    
    return end_effector

def mean_square_error(A,D):
    error = D-A
    mse = np.sum(error**2,axis = 1)
    return mse

def r_by_s_err(MN,re):
    re_left = re[:,0]
    re_right = re[:,1]
    stiffness_array = np.zeros((MN.shape[0],int(MN.shape[1]/2)))
    a = 0
    b = 1
    for i in range(stiffness_array.shape[1]):
        stiffness_array[:,i] = MN[:,a]+MN[:,b]
        a = a+2
        b = b+2
    stiffness_left = np.reshape(np.mean(stiffness_array[:,0:3],axis=-1),[len(MN),1])
    stiffness_right = np.reshape(np.mean(stiffness_array[:,3:],axis=-1),[len(MN),1])
    error_left = np.divide(re_left,stiffness_left)
    error_right = np.divide(re_right,stiffness_right)
    error = np.concatenate((error_left,error_right),axis=1)
    return error

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

# arm_params
arm_right = np.array([0,0,0])
arm_left = np.array([-0.3,0,0])
a_s = 0.04
b_s = 0.07
L_s = 0.3
a_e = 0.03
b_e = 0.08
L_e = 0.3

num_object = ["one_obj","two_obj"]
excel_directories = ["00.xlsx","01.xlsx","10.xlsx","11.xlsx"]

input_images_left_all = []
input_images_right_all = []
hc_l_inputs_all = []
hc_r_inputs_all = []
MN_all = []
targets_desired_left_train = []
targets_desired_right_train = []

input_images_left_all_test = []
input_images_right_all_test = []
hc_l_inputs_all_test = []
hc_r_inputs_all_test = []
MN_all_test = []
targets_desired_left_test = []
targets_desired_right_test = []

for num_obj in num_object:
    # Load Left eye inputs - train data
    directory_left = "D:/MS/Codes/Python/CNN_model_Stroke/Inputs/INPUTS TWO BITS/"+num_obj
    alphabet = 'a'
    input_images_left = input_load_images(directory_left,alphabet,1174)
    input_images_left = np.reshape(input_images_left,[len(input_images_left),45,45,3])
    
    # Load Right eye inputs - train data
    directory_right = "D:/MS/Codes/Python/CNN_model_Stroke/Inputs/INPUTS TWO BITS/"+num_obj
    alphabet = 'b'
    input_images_right = input_load_images(directory_right,alphabet,1174)
    input_images_right = np.reshape(input_images_right,[len(input_images_right),45,45,3])
    
    # Load Left eye inputs - test data
    directory_left_test = "D:/MS/Codes/Python/CNN_model_Stroke/Inputs/INPUTS TWO BITS/"+num_obj+"_testing"
    alphabet_test = 'a'
    if num_obj == "one_obj":
        len_data = 40
    else:
        len_data = 38
    input_images_left_test = input_load_images(directory_left,alphabet_test,len_data)
    input_images_left_test = np.reshape(input_images_left_test,[len(input_images_left_test),45,45,3])
    
    # Load Right eye inputs - test data
    directory_right_test = "D:/MS/Codes/Python/CNN_model_Stroke/Inputs/INPUTS TWO BITS/"+num_obj+"_testing"
    alphabet_test = 'b'
    input_images_right_test = input_load_images(directory_right_test,alphabet_test,len_data)
    input_images_right_test = np.reshape(input_images_right_test,[len(input_images_right_test),45,45,3])
    
    for excel_directory in excel_directories:
        # train data
        arm_data = pd.read_excel("D:/MS/Codes/Python/CNN_model_Stroke/Excel Sheet/2_bit_Modality/Train_Data/"+num_obj+"_"+excel_directory)
        arm_data = np.array(arm_data)
        input_images_left_num_obj = input_images_left
        input_images_right_num_obj = input_images_right
        MN_left = desired_output_array_generator(arm_data,[7,8,9,10,11,12])#MN activations
        MN_right = desired_output_array_generator(arm_data,[19,20,21,22,23,24])#MN activations
        hc_l = desired_output_array_generator(arm_data,[25])
        hc_r = desired_output_array_generator(arm_data,[26])
        MN = np.concatenate((MN_left,MN_right),axis=-1)
        indices = desired_output_array_generator(arm_data,[0]) 
        targets_desired_left_num_obj = desired_output_array_generator(arm_data,[1,2,3])
        targets_desired_right_num_obj = desired_output_array_generator(arm_data,[13,14,15])
        hc_l_inputs = np.zeros((hc_l.shape[0],22,22,1))
        hc_r_inputs = np.zeros((hc_r.shape[0],22,22,1))
        
        for i in range(hc_l.shape[0]):
            hc_l_inputs[i,:,:,:] = np.ones((22,22,1))*hc_l[i,:]
            hc_r_inputs[i,:,:,:] = np.ones((22,22,1))*hc_r[i,:]
            
#        input_images_left_mod,input_images_right_mod,hc_l_inputs_mod,hc_r_inputs_mod,MN_mod = train_test_shuffle_split(input_images_left_num_obj,input_images_right_num_obj,hc_l_inputs,hc_r_inputs,MN)
        input_images_left_all.append(input_images_left_num_obj)
        input_images_right_all.append(input_images_right_num_obj)
        hc_l_inputs_all.append(hc_l_inputs)
        hc_r_inputs_all.append(hc_r_inputs)
        MN_all.append(MN)
        targets_desired_left_train.append(targets_desired_left_num_obj)
        targets_desired_right_train.append(targets_desired_right_num_obj)
        
        # test data
        arm_data_test = pd.read_excel("D:/MS/Codes/Python/CNN_model_Stroke/Excel Sheet/2_bit_Modality/Test_Data/"+num_obj+"_test_"+excel_directory)
        arm_data_test = np.array(arm_data_test)
        input_images_left_num_obj_test = input_images_left_test
        input_images_right_num_obj_test = input_images_right_test
        MN_left_test = desired_output_array_generator(arm_data_test,[7,8,9,10,11,12])#MN activations
        MN_right_test = desired_output_array_generator(arm_data_test,[19,20,21,22,23,24])#MN activations
        hc_l_test = desired_output_array_generator(arm_data_test,[25])
        hc_r_test = desired_output_array_generator(arm_data_test,[26])
        MN_test = np.concatenate((MN_left_test,MN_right_test),axis=-1)
        indices_test = desired_output_array_generator(arm_data_test,[0]) 
        targets_desired_left_num_obj_test = desired_output_array_generator(arm_data_test,[1,2,3])
        targets_desired_right_num_obj_test = desired_output_array_generator(arm_data_test,[13,14,15])
        hc_l_inputs_test = np.zeros((hc_l_test.shape[0],22,22,1))
        hc_r_inputs_test = np.zeros((hc_r_test.shape[0],22,22,1))
        
        for i in range(hc_l_test.shape[0]):
            hc_l_inputs_test[i,:,:,:] = np.ones((22,22,1))*hc_l_test[i,:]
            hc_r_inputs_test[i,:,:,:] = np.ones((22,22,1))*hc_r_test[i,:]
            
#        input_images_left_mod_test,input_images_right_mod_test,hc_l_inputs_mod_test,hc_r_inputs_mod_test,MN_mod_test = train_test_shuffle_split(input_images_left_num_obj_test,input_images_right_num_obj_test,hc_l_inputs_test,hc_r_inputs_test,MN_test)
        input_images_left_all_test.append(input_images_left_num_obj_test)
        input_images_right_all_test.append(input_images_right_num_obj_test)
        hc_l_inputs_all_test.append(hc_l_inputs_test)
        hc_r_inputs_all_test.append(hc_r_inputs_test)
        MN_all_test.append(MN_test)
        targets_desired_left_test.append(targets_desired_left_num_obj_test)
        targets_desired_right_test.append(targets_desired_right_num_obj_test)
        
input_images_left_all = np.array([ item for elem in input_images_left_all for item in elem])
input_images_right_all = np.array([ item for elem in input_images_right_all for item in elem])
hc_l_inputs_all = np.array([ item for elem in hc_l_inputs_all for item in elem])
hc_r_inputs_all = np.array([ item for elem in hc_r_inputs_all for item in elem])
MN_all = np.array([ item for elem in MN_all for item in elem])
targets_desired_left_train = np.array([ item for elem in targets_desired_left_train for item in elem])
targets_desired_right_train = np.array([ item for elem in targets_desired_right_train for item in elem])

input_images_left_all_test = np.array([ item for elem in input_images_left_all_test for item in elem])
input_images_right_all_test = np.array([ item for elem in input_images_right_all_test for item in elem])
hc_l_inputs_all_test = np.array([ item for elem in hc_l_inputs_all_test for item in elem])
hc_r_inputs_all_test = np.array([ item for elem in hc_r_inputs_all_test for item in elem])
MN_all_test = np.array([ item for elem in MN_all_test for item in elem])
targets_desired_left_test = np.array([ item for elem in targets_desired_left_test for item in elem])
targets_desired_right_test = np.array([ item for elem in targets_desired_right_test for item in elem])

input_left_all = np.concatenate((input_images_left_all,input_images_left_all_test))
input_right_all = np.concatenate((input_images_right_all,input_images_right_all_test))
hc_l_all = np.concatenate((hc_l_inputs_all,hc_l_inputs_all_test))
hc_r_all = np.concatenate((hc_r_inputs_all,hc_r_inputs_all_test))
MN_desired_all = np.concatenate((MN_all,MN_all_test))
targets_desired_left_all = np.concatenate((targets_desired_left_train,targets_desired_left_test))
targets_desired_right_all = np.concatenate((targets_desired_right_train,targets_desired_right_test))

percent = 50
#lesion_indices = np.array([]) 

#for les_ind in lesion_indices:
les_ind_l = []
les_ind_r = []

lay_num = 0
with open("Healthy_Subject_1_50_percent_overlap_FM_indices_w_reg_l1_00005.txt", "rb") as fp:
    ft_arr = pickle.load(fp)
with open("Healthy_Subject_1_50_percent_overlap_FC_indices_w_reg_l1_00005.txt", "rb") as fp:
    fc_arr = pickle.load(fp)
model = load_model("Healthy_Subject_1_50_percent_overlap_w_reg_l1_00005.h5",custom_objects={'feature_map_percent_lesion': feature_map_percent_lesion,'fully_connected_percent_lesion': fully_connected_percent_lesion,'page_dr_shepherd_FC_left':page_dr_shepherd_FC_left,'page_dr_shepherd_FC_right':page_dr_shepherd_FC_right})
wts = model.get_weights()
patient = CCM()
patient.set_weights(wts)
print("Model loaded")
MN_predicted_all = []
activations_cnn_L = []
for i in range(len(input_images_left_all)):
    print(i) #"Plasticity:",plasticity,"Therapy:",therapy,
    l = np.reshape(input_images_left_all[i,:,:,:],[1,45,45,3]) 
    r = np.reshape(input_images_right_all[i,:,:,:],[1,45,45,3])
    hc_l = np.reshape(hc_l_all[i,:,:,:],[1,22,22,1])
    hc_r = np.reshape(hc_r_all[i,:,:,:],[1,22,22,1])
    MN_predicted = patient.predict([l,r,hc_l,hc_r])   
    
# Activations filtering
    cnn_layers = [4]
    activations_cnn = []
    layer_outputs = [layer.output for layer in patient.layers[2:35]] # Extracts the outputs of the top n layers
    activation_model = Model(inputs=patient.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
    activations_all = activation_model.predict([l,r,hc_l,hc_r])
    for l in cnn_layers:
        activations_cnn.append(activations_all[l])
    activations_cnn_L.append(activations_cnn)
    
with open("activations_cnn_R_Blank_Subject_1_50_percent_overlap.txt", "wb") as fp:
    pickle.dump(activations_cnn_L, fp)
    