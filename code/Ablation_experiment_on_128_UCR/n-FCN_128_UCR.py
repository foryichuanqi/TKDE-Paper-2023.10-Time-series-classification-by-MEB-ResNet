# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 21:52:44 2022

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 09:49:08 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:26:41 2020

@author: Administrator
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 20:11:19 2016

@author: stephen
"""

import os
print(os.path.abspath(os.path.join(os.getcwd(), "../..")))
last_last_path=os.path.abspath(os.path.join(os.getcwd(), "../.."))

print(os.path.abspath(os.path.join(os.getcwd(), "..")))
last_path=os.path.abspath(os.path.join(os.getcwd(), ".."))
import tensorflow as tf
 
from tensorflow import keras
# import keras
import numpy as np
import pandas as pd
import os
import pickle
import datetime
import scipy as sp
import datetime
#data = sp.genfromtxt("filename.tsv", delimiter="/t")

def readucr(filename):
    data = np.loadtxt(filename, delimiter = '\t')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y





def transform_to_shrink_value(datas,shrink_value):
    

    print(datas.shape)
    
    if (len(datas[0])//8)>=shrink_value:
        
        ture_shrink_value=shrink_value
    
   
            
    else:
        
        shrink_value=len(datas[0])//8
            
            
        ture_shrink_value=shrink_value
            
            
    return ture_shrink_value
        

                

                
        
            
            
             
            
             
             
        
        
# nb_epochs = 2
# run_times=2   



nb_epochs = 2000
run_times=10
method_name='n-FCN_128_UCR'

run_begin_index=0

shrink_value=5

num_filter1=128
num_filter2=256
num_filter3=128
num_filter4=64



# path = last_last_path + r"/dataset/UCRArchive_2018/UCRArchive_2018_128"
path = last_last_path + r"/dataset/UCRArchive_2018/UCRArchive_2018_128"
flist = os.listdir(path)
flist.sort()
print(flist[run_begin_index:])

#flist=path_list.sort()
flist_44 = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'CricketX', 'CricketY', 'CricketZ', 'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'GunPoint', 'Haptics', 'InlineSkate', 'ItalyPowerDemand', 'Lightning2', 'Lightning7', 'Mallat', 'MedicalImages', 'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'Trace', 'TwoLeadECG', 'TwoPatterns', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'WordSynonyms', 'Yoga']

#flist  = ['Adiac']
# print(flist)
error_record=[]
loss_record=[]



if os.path.exists(last_last_path + r"/experiments_result/method_error_txt/{}.txt".format(method_name)):os.remove(last_last_path + r"/experiments_result/method_error_txt/{}.txt".format(method_name))

# for (num,each) in enumerate(flist[run_begin_index:run_begin_index+1]):
for (num,each) in enumerate(flist[run_begin_index:]):
    print('aaaaaaaaaaa')
    
 
    
    for i in range(run_times):

        print('xxx')
        
        fname = each
    
        x_train, y_train = readucr(last_last_path + r"/dataset/UCRArchive_2018/UCRArchive_2018_128"+"//"+fname+"//"+fname+'_TRAIN.tsv')
        x_test, y_test = readucr(last_last_path + r"/dataset/UCRArchive_2018/UCRArchive_2018_128"+"//"+fname+"//"+fname+'_TEST.tsv')


        
    #    x_train, y_train = readucr(last_last_path + r"/dataset/UCRArchive_2018/UCRArchive_2018"+"//"+'Adiac'+"//"+'Adiac'+'_TRAIN.tsv')
    #    x_test, y_test = readucr(last_last_path + r"/dataset/UCRArchive_2018/UCRArchive_2018"+"//"+'Adiac'+"//"+'Adiac'+'_TEST.tsv')
        
    #    x_train, y_train = readucr(fname+'/'+fname+'_TRAIN')
    #    x_test, y_test = readucr(fname+'/'+fname+'_TEST')
        
        # print(x_train, y_train)
        nb_classes = len(np.unique(y_test))
        batch_size = min(x_train.shape[0]//10, 16)
        
        y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
        y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
        
        
        Y_train = keras.utils.to_categorical(y_train, nb_classes)
        Y_test = keras.utils.to_categorical(y_test, nb_classes)
        
        x_train_mean = x_train.mean()
        x_train_std = x_train.std()
        x_train = (x_train - x_train_mean)/(x_train_std)
         
        x_test = (x_test - x_train_mean)/(x_train_std)
        x_train = x_train.reshape(x_train.shape + (1,1,))
        x_test = x_test.reshape(x_test.shape + (1,1,))
        print(x_train, y_train)
        print(x_train.shape)
        
        ture_shrink_value=transform_to_shrink_value(x_train,shrink_value)  
        ture_shrink_value=transform_to_shrink_value(x_test,shrink_value) 
        
        if ture_shrink_value==5:
            
            input_shape=x_train[0].shape
            input_shape=x_train.shape[1:]
            print('aaaaa')
            print(input_shape)
#            exec('x_{}={}'.format(i,keras.layers.Input((65,1,1,))),globals())
            x_0=keras.layers.Input(input_shape)

            
            
            
            conv1 = keras.layers.Conv2D(num_filter1, 8, strides=1, padding='same')(x_0)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.Activation('relu')(conv1)
            
        #    drop_out = Dropout(0.2)(conv1)
            conv1 = keras.layers.Conv2D(num_filter2, 5, strides=1, padding='same')(conv1)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.Activation('relu')(conv1)
            
        #    drop_out = Dropout(0.2)(conv2)
            conv1 = keras.layers.Conv2D(num_filter3, 3, strides=1, padding='same')(conv1)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.Activation('relu')(conv1)
            full1 = keras.layers.GlobalAveragePooling2D()(conv1)
            
            full1 = keras.layers.Dense(nb_classes, activation='softmax')(full1)
            out1  = full1
            full1 = keras.layers.Reshape((1,-1,1))(full1)
            
            
            
            
            
            conv2 = keras.layers.Conv2D(num_filter1, 8, strides=1, padding='same')(x_0)
            conv2 = keras.layers.BatchNormalization()(conv2)
            conv2 = keras.layers.Activation('relu')(conv2)
            
        #    drop_out = Dropout(0.2)(conv1)
            conv2 = keras.layers.Conv2D(num_filter2, 5, strides=1, padding='same')(conv2)
            conv2 = keras.layers.BatchNormalization()(conv2)
            conv2 = keras.layers.Activation('relu')(conv2)
            
        #    drop_out = Dropout(0.2)(conv2)
            conv2 = keras.layers.Conv2D(num_filter3, 3, strides=1, padding='same')(conv2)
            conv2 = keras.layers.BatchNormalization()(conv2)
            conv2 = keras.layers.Activation('relu')(conv2)
            full2 = keras.layers.GlobalAveragePooling2D()(conv2)
            
            full2 = keras.layers.Dense(nb_classes, activation='softmax')(full2)
            out2  = full2
            full2 = keras.layers.Reshape((1,-1,1))(full2)
            
            
            
          

            conv3 = keras.layers.Conv2D(num_filter1, 8, strides=1, padding='same')(x_0)
            conv3 = keras.layers.BatchNormalization()(conv3)
            conv3 = keras.layers.Activation('relu')(conv3)
            
        #    drop_out = Dropout(0.2)(conv1)
            conv3 = keras.layers.Conv2D(num_filter2, 5, strides=1, padding='same')(conv3)
            conv3 = keras.layers.BatchNormalization()(conv3)
            conv3 = keras.layers.Activation('relu')(conv3)
            
        #    drop_out = Dropout(0.2)(conv2)
            conv3 = keras.layers.Conv2D(num_filter3, 3, strides=1, padding='same')(conv3)
            conv3 = keras.layers.BatchNormalization()(conv3)
            conv3 = keras.layers.Activation('relu')(conv3) 
            full3 = keras.layers.GlobalAveragePooling2D()(conv3)
            
            full3 = keras.layers.Dense(nb_classes, activation='softmax')(full3)
            out3  = full3
            full3 = keras.layers.Reshape((1,-1,1))(full3)




            conv4 = keras.layers.Conv2D(num_filter1, 8, strides=1, padding='same')(x_0)
            conv4 = keras.layers.BatchNormalization()(conv4)
            conv4 = keras.layers.Activation('relu')(conv4)
            
        #    drop_out = Dropout(0.2)(conv1)
            conv4 = keras.layers.Conv2D(num_filter2, 5, strides=1, padding='same')(conv4)
            conv4 = keras.layers.BatchNormalization()(conv4)
            conv4 = keras.layers.Activation('relu')(conv4)
            
        #    drop_out = Dropout(0.2)(conv2)
            conv4 = keras.layers.Conv2D(num_filter3, 3, strides=1, padding='same')(conv4)
            conv4 = keras.layers.BatchNormalization()(conv4)
            conv4 = keras.layers.Activation('relu')(conv4) 
            full4 = keras.layers.GlobalAveragePooling2D()(conv4)
            
            full4 = keras.layers.Dense(nb_classes, activation='softmax')(full4)
            out4  = full4
            full4 = keras.layers.Reshape((1,-1,1))(full4)



            
            
            conv5 = keras.layers.Conv2D(num_filter1, 8, strides=1, padding='same')(x_0)
            conv5 = keras.layers.BatchNormalization()(conv5)
            conv5 = keras.layers.Activation('relu')(conv5)
            
        #    drop_out = Dropout(0.2)(conv1)
            conv5 = keras.layers.Conv2D(num_filter2, 5, strides=1, padding='same')(conv5)
            conv5 = keras.layers.BatchNormalization()(conv5)
            conv5 = keras.layers.Activation('relu')(conv5)
            
        #    drop_out = Dropout(0.2)(conv2)
            conv5 = keras.layers.Conv2D(num_filter3, 3, strides=1, padding='same')(conv5)
            conv5 = keras.layers.BatchNormalization()(conv5)
            conv5 = keras.layers.Activation('relu')(conv5)  
            full5 = keras.layers.GlobalAveragePooling2D()(conv5)
            # full5 = keras.layers.Reshape((-1,1))(full5)
            full5 = keras.layers.Dense(nb_classes, activation='softmax')(full5)
            out5  = full5
            full5 = keras.layers.Reshape((1,-1,1))(full5)
            
            
            out = keras.layers.average([out1,out2,out3,out4,out5])
            # print(full5)
            

            
            # concat = keras.layers.concatenate([full1,full2,full3,full4,full5],axis=1)
            # print(concat)
            # concat = keras.layers.Reshape((nb_classes*5,))(concat)
            # print(concat)
            # # concat = keras.layers.Reshape((-1,5,1))(concat)
            # # print(concat)
            # # concat = keras.layers.Flatten()(concat)
            # # print(concat)
            # out = keras.layers.Dense(nb_classes, activation='softmax')(concat)
            
            model = keras.models.Model(inputs=x_0, outputs=out)
            
            # out = keras.layers.Conv2D(1, (5,1), strides=1)(concat)
            # out = keras.layers.Reshape((-1,))(out)
            
            # out = keras.layers.Flatten()(out)
            # out = keras.layers.BatchNormalization()(out)
            # out = keras.layers.Activation('relu')(out)
            # print(out)
            
            # out = keras.layers.GlobalAveragePooling2D()(out)
            # out = keras.layers.Dense(nb_classes, activation='softmax')(out)
            
            
            
            
        if ture_shrink_value==4:
            
            input_shape=x_train[0].shape
            input_shape=x_train.shape[1:]
            print('aaaaa')
            print(input_shape)
#            exec('x_{}={}'.format(i,keras.layers.Input((65,1,1,))),globals())
            x_0=keras.layers.Input(input_shape)


            
            
            
            conv1 = keras.layers.Conv2D(num_filter1, 8, strides=1, padding='same')(x_0)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.Activation('relu')(conv1)
            
        #    drop_out = Dropout(0.2)(conv1)
            conv1 = keras.layers.Conv2D(num_filter2, 5, strides=1, padding='same')(conv1)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.Activation('relu')(conv1)
            
        #    drop_out = Dropout(0.2)(conv2)
            conv1 = keras.layers.Conv2D(num_filter3, 3, strides=1, padding='same')(conv1)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.Activation('relu')(conv1)
            full1 = keras.layers.GlobalAveragePooling2D()(conv1)
            
            full1 = keras.layers.Dense(nb_classes, activation='softmax')(full1)
            out1  = full1
            full1 = keras.layers.Reshape((1,-1,1))(full1)
            
            
            
            
            
            conv2 = keras.layers.Conv2D(num_filter1, 8, strides=1, padding='same')(x_0)
            conv2 = keras.layers.BatchNormalization()(conv2)
            conv2 = keras.layers.Activation('relu')(conv2)
            
        #    drop_out = Dropout(0.2)(conv1)
            conv2 = keras.layers.Conv2D(num_filter2, 5, strides=1, padding='same')(conv2)
            conv2 = keras.layers.BatchNormalization()(conv2)
            conv2 = keras.layers.Activation('relu')(conv2)
            
        #    drop_out = Dropout(0.2)(conv2)
            conv2 = keras.layers.Conv2D(num_filter3, 3, strides=1, padding='same')(conv2)
            conv2 = keras.layers.BatchNormalization()(conv2)
            conv2 = keras.layers.Activation('relu')(conv2)
            full2 = keras.layers.GlobalAveragePooling2D()(conv2)
            
            full2 = keras.layers.Dense(nb_classes, activation='softmax')(full2)
            out2  = full2
            full2 = keras.layers.Reshape((1,-1,1))(full2)
            
            
            
          

            conv3 = keras.layers.Conv2D(num_filter1, 8, strides=1, padding='same')(x_0)
            conv3 = keras.layers.BatchNormalization()(conv3)
            conv3 = keras.layers.Activation('relu')(conv3)
            
        #    drop_out = Dropout(0.2)(conv1)
            conv3 = keras.layers.Conv2D(num_filter2, 5, strides=1, padding='same')(conv3)
            conv3 = keras.layers.BatchNormalization()(conv3)
            conv3 = keras.layers.Activation('relu')(conv3)
            
        #    drop_out = Dropout(0.2)(conv2)
            conv3 = keras.layers.Conv2D(num_filter3, 3, strides=1, padding='same')(conv3)
            conv3 = keras.layers.BatchNormalization()(conv3)
            conv3 = keras.layers.Activation('relu')(conv3) 
            full3 = keras.layers.GlobalAveragePooling2D()(conv3)
            
            full3 = keras.layers.Dense(nb_classes, activation='softmax')(full3)
            out3  = full3
            full3 = keras.layers.Reshape((1,-1,1))(full3)




            conv4 = keras.layers.Conv2D(num_filter1, 8, strides=1, padding='same')(x_0)
            conv4 = keras.layers.BatchNormalization()(conv4)
            conv4 = keras.layers.Activation('relu')(conv4)
            
        #    drop_out = Dropout(0.2)(conv1)
            conv4 = keras.layers.Conv2D(num_filter2, 5, strides=1, padding='same')(conv4)
            conv4 = keras.layers.BatchNormalization()(conv4)
            conv4 = keras.layers.Activation('relu')(conv4)
            
        #    drop_out = Dropout(0.2)(conv2)
            conv4 = keras.layers.Conv2D(num_filter3, 3, strides=1, padding='same')(conv4)
            conv4 = keras.layers.BatchNormalization()(conv4)
            conv4 = keras.layers.Activation('relu')(conv4) 
            full4 = keras.layers.GlobalAveragePooling2D()(conv4)
            
            full4 = keras.layers.Dense(nb_classes, activation='softmax')(full4)
            out4  = full4
            full4 = keras.layers.Reshape((1,-1,1))(full4)


            out = keras.layers.average([out1,out2,out3,out4])
            # print(full5)
            

            
    
            
            model = keras.models.Model(inputs=x_0, outputs=out)
            
            

            

            
 
            
            
        if ture_shrink_value==3:
            
            input_shape=x_train[0].shape
            input_shape=x_train.shape[1:]
            print('aaaaa')
            print(input_shape)
#            exec('x_{}={}'.format(i,keras.layers.Input((65,1,1,))),globals())
            x_0=keras.layers.Input(input_shape)


            
            
            
            conv1 = keras.layers.Conv2D(num_filter1, 8, strides=1, padding='same')(x_0)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.Activation('relu')(conv1)
            
        #    drop_out = Dropout(0.2)(conv1)
            conv1 = keras.layers.Conv2D(num_filter2, 5, strides=1, padding='same')(conv1)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.Activation('relu')(conv1)
            
        #    drop_out = Dropout(0.2)(conv2)
            conv1 = keras.layers.Conv2D(num_filter3, 3, strides=1, padding='same')(conv1)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.Activation('relu')(conv1)
            full1 = keras.layers.GlobalAveragePooling2D()(conv1)
            
            full1 = keras.layers.Dense(nb_classes, activation='softmax')(full1)
            out1  = full1
            full1 = keras.layers.Reshape((1,-1,1))(full1)
            
            
            
            
            
            conv2 = keras.layers.Conv2D(num_filter1, 8, strides=1, padding='same')(x_0)
            conv2 = keras.layers.BatchNormalization()(conv2)
            conv2 = keras.layers.Activation('relu')(conv2)
            
        #    drop_out = Dropout(0.2)(conv1)
            conv2 = keras.layers.Conv2D(num_filter2, 5, strides=1, padding='same')(conv2)
            conv2 = keras.layers.BatchNormalization()(conv2)
            conv2 = keras.layers.Activation('relu')(conv2)
            
        #    drop_out = Dropout(0.2)(conv2)
            conv2 = keras.layers.Conv2D(num_filter3, 3, strides=1, padding='same')(conv2)
            conv2 = keras.layers.BatchNormalization()(conv2)
            conv2 = keras.layers.Activation('relu')(conv2)
            full2 = keras.layers.GlobalAveragePooling2D()(conv2)
            
            full2 = keras.layers.Dense(nb_classes, activation='softmax')(full2)
            out2  = full2
            full2 = keras.layers.Reshape((1,-1,1))(full2)
            
            
            
          

            conv3 = keras.layers.Conv2D(num_filter1, 8, strides=1, padding='same')(x_0)
            conv3 = keras.layers.BatchNormalization()(conv3)
            conv3 = keras.layers.Activation('relu')(conv3)
            
        #    drop_out = Dropout(0.2)(conv1)
            conv3 = keras.layers.Conv2D(num_filter2, 5, strides=1, padding='same')(conv3)
            conv3 = keras.layers.BatchNormalization()(conv3)
            conv3 = keras.layers.Activation('relu')(conv3)
            
        #    drop_out = Dropout(0.2)(conv2)
            conv3 = keras.layers.Conv2D(num_filter3, 3, strides=1, padding='same')(conv3)
            conv3 = keras.layers.BatchNormalization()(conv3)
            conv3 = keras.layers.Activation('relu')(conv3) 
            full3 = keras.layers.GlobalAveragePooling2D()(conv3)
            
            full3 = keras.layers.Dense(nb_classes, activation='softmax')(full3)
            out3  = full3
            full3 = keras.layers.Reshape((1,-1,1))(full3)



            out = keras.layers.average([out1,out2,out3])
            # print(full5)
            

            
          
            # out = keras.layers.Dense(nb_classes, activation='softmax')(concat)
            
            model = keras.models.Model(inputs=x_0, outputs=out)

            

            
      
                

        if ture_shrink_value==2:
            
            input_shape=x_train[0].shape
            input_shape=x_train.shape[1:]
            print('aaaaa')
            print(input_shape)
#            exec('x_{}={}'.format(i,keras.layers.Input((65,1,1,))),globals())
            x_0=keras.layers.Input(input_shape)



            
            
            
            conv1 = keras.layers.Conv2D(num_filter1, 8, strides=1, padding='same')(x_0)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.Activation('relu')(conv1)
            
        #    drop_out = Dropout(0.2)(conv1)
            conv1 = keras.layers.Conv2D(num_filter2, 5, strides=1, padding='same')(conv1)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.Activation('relu')(conv1)
            
        #    drop_out = Dropout(0.2)(conv2)
            conv1 = keras.layers.Conv2D(num_filter3, 3, strides=1, padding='same')(conv1)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.Activation('relu')(conv1)
            full1 = keras.layers.GlobalAveragePooling2D()(conv1)
            
            full1 = keras.layers.Dense(nb_classes, activation='softmax')(full1)
            out1  = full1
            full1 = keras.layers.Reshape((1,-1,1))(full1)
            
            
            
            
            
            conv2 = keras.layers.Conv2D(num_filter1, 8, strides=1, padding='same')(x_0)
            conv2 = keras.layers.BatchNormalization()(conv2)
            conv2 = keras.layers.Activation('relu')(conv2)
            
        #    drop_out = Dropout(0.2)(conv1)
            conv2 = keras.layers.Conv2D(num_filter2, 5, strides=1, padding='same')(conv2)
            conv2 = keras.layers.BatchNormalization()(conv2)
            conv2 = keras.layers.Activation('relu')(conv2)
            
        #    drop_out = Dropout(0.2)(conv2)
            conv2 = keras.layers.Conv2D(num_filter3, 3, strides=1, padding='same')(conv2)
            conv2 = keras.layers.BatchNormalization()(conv2)
            conv2 = keras.layers.Activation('relu')(conv2)
            full2 = keras.layers.GlobalAveragePooling2D()(conv2)
            
            full2 = keras.layers.Dense(nb_classes, activation='softmax')(full2)
            out2  = full2
            full2 = keras.layers.Reshape((1,-1,1))(full2)
            
            
            out = keras.layers.average([out1,out2])
            # print(full5)
            

       
            
            model = keras.models.Model(inputs=x_0, outputs=out)            
          







            

            
            
            
            
            
        if ture_shrink_value==1:
            
            input_shape=x_train[0].shape
            input_shape=x_train.shape[1:]
            print('aaaaa')
            print(input_shape)
#            exec('x_{}={}'.format(i,keras.layers.Input((65,1,1,))),globals())
            x_0=keras.layers.Input(input_shape)



            
            
            
            conv1 = keras.layers.Conv2D(num_filter1, 8, strides=1, padding='same')(x_0)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.Activation('relu')(conv1)
            
        #    drop_out = Dropout(0.2)(conv1)
            conv1 = keras.layers.Conv2D(num_filter2, 5, strides=1, padding='same')(conv1)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.Activation('relu')(conv1)
            
        #    drop_out = Dropout(0.2)(conv2)
            conv1 = keras.layers.Conv2D(num_filter3, 3, strides=1, padding='same')(conv1)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.Activation('relu')(conv1)
            full1 = keras.layers.GlobalAveragePooling2D()(conv1)
            
            full1 = keras.layers.Dense(nb_classes, activation='softmax')(full1)
            out1  = full1
            # full1 = keras.layers.Reshape((1,-1,1))(full1)
            
            
            
        
            out = keras.layers.Dense(nb_classes, activation='softmax')(out1)
            
            model = keras.models.Model(inputs=x_0, outputs=out)
            
            
        
            
        
        # model = keras.models.Model(inputs=x_0, outputs=[out1,out2,out3,out4,out5,out])
         
        optimizer = keras.optimizers.Adam()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
         
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                          patience=50, min_lr=0.0001) 

            
        hist = model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
                  verbose=1, validation_data=(x_test,Y_test ), callbacks = [reduce_lr])

        #Print the testing results which has the lowest training loss.
        print(hist.history.keys())
    
#     keras.backend.clear_session()
        log = pd.DataFrame(hist.history)
        log.to_excel(last_last_path + r"/experiments_result/log/{}_dataset_{}_log{}_time{}.xlsx".format(method_name,str(flist.index(each)),i,datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
        # print(log.items()) 
        print(log.loc[log['loss'].idxmin()]['loss'], log.loc[log['loss'].idxmin()]['val_acc'])# now idxmin()()  not dixmin print('Loss: ', log.loc[log['loss'].idxmin()()]['loss'])
        error_record.append(1-log.loc[log['loss'].idxmin()]['val_acc'])
        loss_record.append(log.loc[log['loss'].idxmin()]['loss'])
            
        file = open(last_last_path + r"/experiments_result/method_error_txt/{}.txt".format(method_name), 'a')
        # file.write( str(flist.index(each))+'error:'+'    '+str('%.5f'%(1-log.loc[log['loss'].idxmin()]['val_acc']))+'     ')
        file.write( str(flist.index(each))+'error:'+'    '+str('%.5f'%(1-log.loc[log['loss'].idxmin()]['val_acc']))+'     '+'loss:'+str('%.8f'%(log.loc[log['loss'].idxmin()]['loss']))+'     ')
    #        file.write( 'error:'+'   '+str('%.5f'%(1-log.loc[log['loss'].idxmin()]['val_acc']))+'        '+'corresponding_min_loss:'+'   '+str('%.5f'%log.loc[log['loss'].idxmin()]['loss']) +'        '+str(flist.index(each))+'        ' +each +'/n')
    #        file.write( 'error:'+'   '+str('%.5f'%(1-log.loc[log['loss'].idxmin()]['val_acc']))+'        '+'corresponding_min_loss:'+'   '+str('%.5f'%log.loc[log['loss'].idxmin()]['loss']) +'        '+str(flist.index(each))+'        ' +each +'/n')
    
        file.close()
    
        print('!!!!!!!!!!!!!!!!  {} {} {}::runtime:{}____min_error:{}'.format(method_name,num,each,i,'%.5f'%min(error_record)))
    
    file = open(last_last_path + r"/experiments_result/method_error_txt/{}.txt".format(method_name), 'a')
    file.write('min_error:'+'     '+ str('%.5f'%(min(error_record)))+'     '+'     '+str(flist.index(each))+'        ' +each +'\n')
    file.close()
    error_record=[]
    loss_record=[] 
    keras.backend.clear_session()

    
