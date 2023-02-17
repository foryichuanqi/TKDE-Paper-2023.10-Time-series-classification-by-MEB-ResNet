# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:26:16 2022

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:53:33 2020

@author: Administrator
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 20:11:19 2016

@author: stephen
"""
 

 
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import pickle
import scipy as sp
import datetime


#data = sp.genfromtxt("filename.tsv", delimiter="\t")


print(os.path.abspath(os.path.join(os.getcwd(), "../..")))
last_last_path=os.path.abspath(os.path.join(os.getcwd(), "../.."))

print(os.path.abspath(os.path.join(os.getcwd(), "..")))
last_path=os.path.abspath(os.path.join(os.getcwd(), ".."))

def readucr(filename):
    data = np.loadtxt(filename, delimiter = '\t')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

    
def divide_x_y_by_lenth(X_train,y_train,lenth):
    
    print('time series max_len_is {}'.format(X_train.shape[2]))
    print('time series len is {}'.format(lenth))
    X_train = np.transpose(X_train, (0, 2, 1))
    X_train=np.expand_dims(X_train, axis=2)
    x_tr=[]
    y_tr=[]
    for i in range(X_train.shape[0]):
        for j in range(int(X_train.shape[1]/lenth)):
            x_tr.append(X_train[i,j*lenth:(j+1)*lenth,:,:])
            y_tr.append(y_train[i])
    x_tr=np.array(x_tr)
    y_tr=np.array(y_tr)
    
    return x_tr, y_tr                   
        

nb_epochs = 250
run_times=10

run_begin_index=0
method_name='FCN_MTSC'


path = last_last_path + r"/dataset/MTSC/MTSC"
flist = os.listdir(path)
flist.sort(key=str.lower)
print(flist[run_begin_index:])


error_record=[]
loss_record=[]



if os.path.exists(last_last_path + r"/experiments_result/method_error_txt/{}.txt".format(method_name)):os.remove(last_last_path + r"/experiments_result/method_error_txt/{}.txt".format(method_name))

# for (num,each) in enumerate(flist[run_begin_index:run_begin_index+1]):
for (num,each) in enumerate(flist[run_begin_index:]):
    print('aaaaaaaaaaa')
    print('each')
    
    

    
    for i in range(run_times):

        print('xxx')
        
        fname = each

        
        
        X_train = np.load(last_last_path + r"/dataset/MTSC/MTSC"+"//" +fname+"//"+fname+"//"+ 'X_train.npy')
        y_train = np.load(last_last_path + r"/dataset/MTSC/MTSC"+"//" +fname+"//"+fname+ "//"+'y_train.npy')
        X_test  = np.load(last_last_path + r"/dataset/MTSC/MTSC"+"//" +fname+"//"+fname+"//"+'X_test.npy')
        y_test  = np.load(last_last_path + r"/dataset/MTSC/MTSC"+"//" +fname+"//"+fname+"//"+ 'y_test.npy')
        
        x_train,y_train=divide_x_y_by_lenth(X_train,y_train,X_train.shape[2])
        x_test,y_test=divide_x_y_by_lenth(X_test,y_test,X_train.shape[2])
        
        if np.any(y_test == 0):
            print('exist 0')
            
        else:
            y_train-=min(np.unique(y_test))

            y_test-=min(np.unique(y_test))
            # print(y_test)
        if  each==   'NetFlow':
            y_train[y_train==12]=1
            y_test[y_test==12]=1
            
        nb_classes = len(np.unique(y_test))
        batch_size = min(x_train.shape[0]//10, 16)
        
        # y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
        # y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
        
        
        Y_train = keras.utils.to_categorical(y_train, nb_classes)
        Y_test = keras.utils.to_categorical(y_test, nb_classes)
        
        # x_train_mean = x_train.mean()
        # x_train_std = x_train.std()
        # x_train = (x_train - x_train_mean)/(x_train_std)
         
        # x_test = (x_test - x_train_mean)/(x_train_std)
        # x_train = x_train.reshape(x_train.shape + (1,1,))
        # x_test = x_test.reshape(x_test.shape + (1,1,))
    
        x = keras.layers.Input(x_train.shape[1:])
        print(x_train.shape[1:])
    #    drop_out = Dropout(0.2)(x)
        conv1 = keras.layers.Conv2D(128, 8, 1, padding='same')(x)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation('relu')(conv1)
        
    #    drop_out = Dropout(0.2)(conv1)
        conv2 = keras.layers.Conv2D(256, 5, 1, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)
        
    #    drop_out = Dropout(0.2)(conv2)
        conv3 = keras.layers.Conv2D(128, 3, 1, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)
        
        full = keras.layers.GlobalAveragePooling2D()(conv3)
        # print(full)
        out = keras.layers.Dense(nb_classes, activation='softmax')(full)
        
        
        model = keras.models.Model(inputs=x, outputs=out)
         
        optimizer = keras.optimizers.Adam()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
         
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                          patience=50, min_lr=0.0001) 
        hist = model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
                  verbose=1, validation_data=(x_test, Y_test), callbacks = [reduce_lr])
        #Print the testing results which has the lowest training loss.


        log = pd.DataFrame(hist.history)
        log.to_excel(last_last_path + r"/experiments_result/log/{}_dataset_{}_log{}_time{}.xlsx".format(method_name,str(flist.index(each)),i,datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
             
            
        print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc'])
        error_record.append(1-log.loc[log['loss'].idxmin]['val_acc'])
        loss_record.append(log.loc[log['loss'].idxmin]['loss'])
        
        file = open(last_last_path + r"/experiments_result/method_error_txt/{}.txt".format(method_name), 'a')
        # file.write( str(flist.index(each))+'error:'+'    '+str('%.5f'%(1-log.loc[log['loss'].idxmin]['val_acc']))+'     ')
        file.write( str(flist.index(each))+'error:'+'    '+str('%.5f'%(1-log.loc[log['loss'].idxmin]['val_acc']))+'     '+'loss:'+str('%.8f'%(log.loc[log['loss'].idxmin]['loss']))+'     ')
#        file.write( 'error:'+'   '+str('%.5f'%(1-log.loc[log['loss'].idxmin]['val_acc']))+'        '+'corresponding_min_loss:'+'   '+str('%.5f'%log.loc[log['loss'].idxmin]['loss']) +'        '+str(flist.index(each))+'        ' +each +'/n')
#        file.write( 'error:'+'   '+str('%.5f'%(1-log.loc[log['loss'].idxmin]['val_acc']))+'        '+'corresponding_min_loss:'+'   '+str('%.5f'%log.loc[log['loss'].idxmin]['loss']) +'        '+str(flist.index(each))+'        ' +each +'/n')
    
        file.close()

        print('!!!!!!!!!!!!!!!!  {} {} {}::runtime:{}____min_error:{}'.format(method_name,num,each,i,'%.5f'%min(error_record)))

    file = open(last_last_path + r"/experiments_result/method_error_txt/{}.txt".format(method_name), 'a')
    file.write('min_error:'+'     '+ str('%.5f'%(min(error_record)))+'     '+'min_loss_id:'+'     '+ str('%.5f'%(loss_record.index(min(loss_record))))+'     '+'min_loss_error:'+'     '+ str('%.5f'%(error_record[loss_record.index(min(loss_record))]))+'     '+str(flist.index(each))+'        ' +each +'/n')
    file.close()
    error_record=[]
    loss_record=[]
    keras.backend.clear_session()


    
    
