# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 17:06:22 2021

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

 
from tensorflow import keras
# import keras
import numpy as np
import pandas as pd
import os
import pickle
import datetime
import scipy as sp
import datetime
#data = sp.genfromtxt("filename.tsv", delimiter="\t")
np.random.seed(813306)

def build_resnet(input_shape, n_feature_maps, nb_classes):

    conv_x = keras.layers.BatchNormalization()(input_shape)
    conv_x = keras.layers.Conv2D(n_feature_maps, 8, 1, padding='same')(conv_x)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
     
    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps, 1, 1,padding='same')(input_shape)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(input_shape)
    print ('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)
     
    print ('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv2D(n_feature_maps*2, 8, 1, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps*2, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps*2, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
     
    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps*2, 1, 1,padding='same')(x1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x1)
    print ('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)
     
    print ('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv2D(n_feature_maps*2, 8, 1, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps*2, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps*2, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps*2, 1, 1,padding='same')(x1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x1)
    print ('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)
     
    full = keras.layers.GlobalAveragePooling2D()(y)
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)
    print ('        -- model was built.')
    return  out


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
        

                

                
        
            
            
             
            
             
             
        



nb_epochs = 1500
run_times=10
method_name='MEB-ResNet_85_UCR'
     
run_begin_index=0

shrink_value=5

num_filter1=128
num_filter2=256
num_filter3=128
num_filter4=64

# num_filter1=32
# num_filter2=64
# num_filter3=32
# num_filter4=32


path = last_last_path + r"/dataset/UCRArchive_2018/UCRArchive_2018_128"
flist = os.listdir(path)
print(flist[run_begin_index:])

#flist=path_list.sort()

#flist  = ['Adiac']
# print(flist)
error_record=[]
loss_record=[]

if os.path.exists(last_last_path + r"/experiments_result/method_error_txt/{}.txt".format(method_name)):os.remove(last_last_path + r"/experiments_result/method_error_txt/{}.txt".format(method_name))


for (num,each) in enumerate(flist[run_begin_index:run_begin_index+2]):
    print('aaaaaaaaaaa')
    
  
    
    for i in range(run_times):
        print('xxx')
        
        fname = each
    
        x_train, y_train = readucr(path+"//"+fname+"//"+fname+'_TRAIN.tsv')
        x_test, y_test = readucr(path+"//"+fname+"//"+fname+'_TEST.tsv')



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
        
        ture_shrink_value=transform_to_shrink_value(x_train,shrink_value)  
        ture_shrink_value=transform_to_shrink_value(x_test,shrink_value) 
        
        if ture_shrink_value==5:
            
            input_shape=x_train[0].shape
            input_shape=x_train.shape[1:]
            print('aaaaa')
            print(input_shape)
#            exec('x_{}={}'.format(i,keras.layers.Input((65,1,1,))),globals())
            x_0=keras.layers.Input(input_shape)
            multi_scale_1=x_0
            multi_scale_2=keras.layers.AveragePooling2D((2,1), strides=2)(x_0)
            multi_scale_3=keras.layers.AveragePooling2D((3,1), strides=3)(x_0)
            multi_scale_4=keras.layers.AveragePooling2D((4,1), strides=4)(x_0)
            multi_scale_5=keras.layers.AveragePooling2D((5,1), strides=5)(x_0)
            
            full1 =build_resnet(multi_scale_1, 64, nb_classes)            
            out1  = full1
            full1 = keras.layers.Reshape((1,-1,1))(full1)
            
            
            full2 =build_resnet(multi_scale_2, 64, nb_classes)            
            out2  = full2
            full2 = keras.layers.Reshape((1,-1,1))(full2)     
            
            
            full3 =build_resnet(multi_scale_3, 64, nb_classes)            
            out3  = full3
            full3 = keras.layers.Reshape((1,-1,1))(full3)            

            
            
            full4 =build_resnet(multi_scale_4, 64, nb_classes)            
            out4  = full4
            full4 = keras.layers.Reshape((1,-1,1))(full4)            
          
            full5 =build_resnet(multi_scale_5, 64, nb_classes)            
            out5  = full5
            full5 = keras.layers.Reshape((1,-1,1))(full5)
            
            
            
        
            

            
            concat = keras.layers.concatenate([full1,full2,full3,full4,full5],axis=1)
            print(concat)
            concat = keras.layers.Reshape((nb_classes*5,))(concat)
            print(concat)
            # concat = keras.layers.Reshape((-1,5,1))(concat)
            # print(concat)
            # concat = keras.layers.Flatten()(concat)
            # print(concat)
            out = keras.layers.Dense(nb_classes, activation='softmax')(concat)
            
            model = keras.models.Model(inputs=x_0, outputs=[out1,out2,out3,out4,out5,out])
            
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
            multi_scale_1=x_0
            multi_scale_2=keras.layers.AveragePooling2D((2,1), strides=2)(x_0)
            multi_scale_3=keras.layers.AveragePooling2D((3,1), strides=3)(x_0)
            multi_scale_4=keras.layers.AveragePooling2D((4,1), strides=4)(x_0)

            
            
            
            full1 =build_resnet(multi_scale_1, 64, nb_classes)            
            out1  = full1
            full1 = keras.layers.Reshape((1,-1,1))(full1)
            
            
            full2 =build_resnet(multi_scale_2, 64, nb_classes)            
            out2  = full2
            full2 = keras.layers.Reshape((1,-1,1))(full2)     
            
            
            full3 =build_resnet(multi_scale_3, 64, nb_classes)            
            out3  = full3
            full3 = keras.layers.Reshape((1,-1,1))(full3)            

            
            
            full4 =build_resnet(multi_scale_4, 64, nb_classes)            
            out4  = full4
            full4 = keras.layers.Reshape((1,-1,1))(full4)            
          



            
            

            

            
            concat = keras.layers.concatenate([full1,full2,full3,full4],axis=1)
            print(concat)
            concat = keras.layers.Reshape((nb_classes*4,))(concat)
            print(concat)
            # concat = keras.layers.Reshape((-1,5,1))(concat)
            # print(concat)
            # concat = keras.layers.Flatten()(concat)
            # print(concat)
            out = keras.layers.Dense(nb_classes, activation='softmax')(concat)
            
            model = keras.models.Model(inputs=x_0, outputs=[out1,out2,out3,out4,out])
            
            
        if ture_shrink_value==3:
            
            input_shape=x_train[0].shape
            input_shape=x_train.shape[1:]
            print('aaaaa')
            print(input_shape)
#            exec('x_{}={}'.format(i,keras.layers.Input((65,1,1,))),globals())
            x_0=keras.layers.Input(input_shape)
            multi_scale_1=x_0
            multi_scale_2=keras.layers.AveragePooling2D((2,1), strides=2)(x_0)
            multi_scale_3=keras.layers.AveragePooling2D((3,1), strides=3)(x_0)

            
            
            
            full1 =build_resnet(multi_scale_1, 64, nb_classes)            
            out1  = full1
            full1 = keras.layers.Reshape((1,-1,1))(full1)
            
            
            full2 =build_resnet(multi_scale_2, 64, nb_classes)            
            out2  = full2
            full2 = keras.layers.Reshape((1,-1,1))(full2)     
            
            
            full3 =build_resnet(multi_scale_3, 64, nb_classes)            
            out3  = full3
            full3 = keras.layers.Reshape((1,-1,1))(full3)            

            
            






            

            
            concat = keras.layers.concatenate([full1,full2,full3],axis=1)
            print(concat)
            concat = keras.layers.Reshape((nb_classes*3,))(concat)
            print(concat)
            # concat = keras.layers.Reshape((-1,5,1))(concat)
            # print(concat)
            # concat = keras.layers.Flatten()(concat)
            # print(concat)
            out = keras.layers.Dense(nb_classes, activation='softmax')(concat)
            
            model = keras.models.Model(inputs=x_0, outputs=[out1,out2,out3,out])
                

        if ture_shrink_value==2:
            
            input_shape=x_train[0].shape
            input_shape=x_train.shape[1:]
            print('aaaaa')
            print(input_shape)
#            exec('x_{}={}'.format(i,keras.layers.Input((65,1,1,))),globals())
            x_0=keras.layers.Input(input_shape)
            multi_scale_1=x_0
            multi_scale_2=keras.layers.AveragePooling2D((2,1), strides=2)(x_0)


            
            
            
            full1 =build_resnet(multi_scale_1, 64, nb_classes)            
            out1  = full1
            full1 = keras.layers.Reshape((1,-1,1))(full1)
            
            
            full2 =build_resnet(multi_scale_2, 64, nb_classes)            
            out2  = full2
            full2 = keras.layers.Reshape((1,-1,1))(full2)     
            
         

            
            
            
          







            

            
            concat = keras.layers.concatenate([full1,full2],axis=1)
            print(concat)
            concat = keras.layers.Reshape((nb_classes*2,))(concat)
            print(concat)
            # concat = keras.layers.Reshape((-1,5,1))(concat)
            # print(concat)
            # concat = keras.layers.Flatten()(concat)
            # print(concat)
            out = keras.layers.Dense(nb_classes, activation='softmax')(concat)
            
            model = keras.models.Model(inputs=x_0, outputs=[out1,out2,out])
            
            
            
            
            
        if ture_shrink_value==1:
            
            input_shape=x_train[0].shape
            input_shape=x_train.shape[1:]
            print('aaaaa')
            print(input_shape)
#            exec('x_{}={}'.format(i,keras.layers.Input((65,1,1,))),globals())
            x_0=keras.layers.Input(input_shape)
            multi_scale_1=x_0
            multi_scale_2=keras.layers.AveragePooling2D((2,1), strides=2)(x_0)


            
            
            
            full1 =build_resnet(multi_scale_1, 64, nb_classes)            
            out1  = full1
        

            # full1 = keras.layers.Reshape((1,-1,1))(full1)
            
            
            
        
            out = keras.layers.Dense(nb_classes, activation='softmax')(out1)
            
            model = keras.models.Model(inputs=x_0, outputs=[out1,out])
            
            
        
            
        
        # model = keras.models.Model(inputs=x_0, outputs=[out1,out2,out3,out4,out5,out])
         
        optimizer = keras.optimizers.Adam()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
          
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                          patience=50, min_lr=0.0001) 
        
        list_Y_train=[]
        for j in range(ture_shrink_value+1):
            list_Y_train.append(Y_train)
            
        list_Y_test=[]
        for j in range(ture_shrink_value+1):
            list_Y_test.append(Y_test)
            
        hist = model.fit(x_train, list_Y_train, batch_size=batch_size, epochs=nb_epochs,
                  verbose=1, validation_data=(x_test,list_Y_test ), callbacks = [reduce_lr])





        #Print the testing results which has the lowest training loss.
        print(hist.history.keys())
        if list(hist.history.keys())[0]=='loss' :
            syn_val_acc=list(hist.history.keys())[-2]
            syn_loss=list(hist.history.keys())[int((len(list(hist.history.keys()))-2)/4)]    
            
        else:
            syn_loss=list(hist.history.keys())[int((len(list(hist.history.keys()))-3)/4*3+1)]
            syn_val_acc=list(hist.history.keys())[int((len(list(hist.history.keys()))-3)/2)]
        print(syn_loss,syn_val_acc)        
        

        
        
        
        log = pd.DataFrame(hist.history)
        log.to_excel(last_last_path + r"/experiments_result/log/{}_dataset_{}_log{}_time{}.xlsx".format(method_name,str(flist.index(each)),i,datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
        
        print(log.loc[log[syn_loss].idxmin][syn_loss], log.loc[log[syn_loss].idxmin][syn_val_acc])
        error_record.append(1-log.loc[log[syn_loss].idxmin][syn_val_acc])
        loss_record.append(log.loc[log[syn_loss].idxmin][syn_loss])
#        with open(last_last_path + r"/experiments_result/log/{}_dataset_{}_log{}.txt".format(method_name,str(flist.index(each)),i), 'wb') as file_txt:
##        with open(last_last_path + r"/experiments_result/log/{}_dataset_{}_log{}_time{}.txt".format(method_name,str(flist.index(each)),i,str(datetime.datetime.now())), 'wb') as file_txt:
#            pickle.dump(hist.history, file_txt)
        
        file = open(last_last_path + r"/experiments_result/method_error_txt/{}.txt".format(method_name), 'a')
        # file.write( str(flist.index(each))+'error:'+'    '+str('%.5f'%(1-log.loc[log[syn_loss].idxmin][syn_val_acc]))+'     ')
        file.write( str(flist.index(each))+'error:'+'    '+str('%.5f'%(1-log.loc[log[syn_loss].idxmin][syn_val_acc]))+'     '+'loss:'+str('%.8f'%(log.loc[log[syn_loss].idxmin][syn_loss]))+'     ')
#        file.write( 'error:'+'   '+str('%.5f'%(1-log.loc[log['loss'].idxmin]['val_acc']))+'        '+'corresponding_min_loss:'+'   '+str('%.5f'%log.loc[log['loss'].idxmin]['loss']) +'        '+str(flist.index(each))+'        ' +each +'/n')
#        file.write( 'error:'+'   '+str('%.5f'%(1-log.loc[log['loss'].idxmin]['val_acc']))+'        '+'corresponding_min_loss:'+'   '+str('%.5f'%log.loc[log['loss'].idxmin]['loss']) +'        '+str(flist.index(each))+'        ' +each +'/n')
    
        file.close()

        print('!!!!!!!!!!!!!!!!  {} {} {}::runtime:{}____min_error:{}'.format(method_name,num,each,i,'%.5f'%min(error_record)))

    file = open(last_last_path + r"/experiments_result/method_error_txt/{}.txt".format(method_name), 'a')
    file.write('min_error:'+'     '+ str('%.5f'%(min(error_record)))+'     '+'     '+str(flist.index(each))+'        ' +each +'\n')
    file.close()
    error_record=[]
    loss_record=[]

    
    
