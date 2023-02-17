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
 
import os
print(os.path.abspath(os.path.join(os.getcwd(), "../..")))
last_last_path=os.path.abspath(os.path.join(os.getcwd(), "../.."))

print(os.path.abspath(os.path.join(os.getcwd(), "..")))
last_path=os.path.abspath(os.path.join(os.getcwd(), ".."))
 
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import pickle
import scipy as sp
import datetime
#data = sp.genfromtxt("filename.tsv", delimiter="\t")

def readucr(filename):
    data = np.loadtxt(filename, delimiter = '\t')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y
  
nb_epochs = 2000
run_times=10

run_begin_index=0
method_name='FCN_128_UCR'


path = last_last_path + r"/dataset/UCRArchive_2018/UCRArchive_2018_128"


flist = os.listdir(path)
print(flist[run_begin_index:])




#flist  = ['Adiac']
# print(flist)
error_record=[]
loss_record=[]

if os.path.exists(last_last_path + r"/experiments_result/method_error_txt/{}.txt".format(method_name)):os.remove(last_last_path + r"/experiments_result/method_error_txt/{}.txt".format(method_name))


for (num,each) in enumerate(flist[run_begin_index:]):
    for i in range(run_times):
        
        fname = each
    
        x_train, y_train = readucr(last_last_path + r"/dataset/UCRArchive_2018/UCRArchive_2018_128"+"//"+fname+"//"+fname+'_TRAIN.tsv')
        x_test, y_test = readucr(last_last_path + r"/dataset/UCRArchive_2018/UCRArchive_2018_128"+"//"+fname+"//"+fname+'_TEST.tsv')
    
    #    x_train, y_train = readucr(r"F:\桌面11.17\project\dataset\UCRArchive_2018\UCRArchive_2018"+"\\"+'Adiac'+"\\"+'Adiac'+'_TRAIN.tsv')
    #    x_test, y_test = readucr(r"F:\桌面11.17\project\dataset\UCRArchive_2018\UCRArchive_2018"+"\\"+'Adiac'+"\\"+'Adiac'+'_TEST.tsv')
        
    #    x_train, y_train = readucr(fname+'/'+fname+'_TRAIN')
    #    x_test, y_test = readucr(fname+'/'+fname+'_TEST')
        
        print(x_train, y_train)
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
    file.write('min_error:'+'     '+ str('%.5f'%(min(error_record)))+'     '+'     '+str(flist.index(each))+'        ' +each +'\n')
    file.close()
    error_record=[]
    loss_record=[]


    
    
# https://blog.csdn.net/herr_kun/article/details/106997646    
############## Get CAM ################
# import matplotlib.pyplot as plt
# # from matplotlib.backends.backend_pdf import PdfPages

# get_last_conv = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [model.layers[-3].output])
# last_conv = get_last_conv([x_test[:100], 1])[0]

# get_softmax = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [model.layers[-1].output])
# softmax = get_softmax(([x_test[:100], 1]))[0]
# softmax_weight = model.get_weights()[-2]
# CAM = np.dot(last_conv, softmax_weight)


# pp = PdfPages('CAM.pdf')
# for k in range(20):
#     CAM = (CAM - CAM.min(axis=1, keepdims=True)) / (CAM.max(axis=1, keepdims=True) - CAM.min(axis=1, keepdims=True))
#     c = np.exp(CAM) / np.sum(np.exp(CAM), axis=1, keepdims=True)
#     plt.figure(figsize=(13, 7));
#     plt.plot(x_test[k].squeeze());
#     plt.scatter(np.arange(len(x_test[k])), x_test[k].squeeze(), cmap='hot_r', c=c[k, :, :, int(y_test[k])].squeeze(), s=100);
#     plt.title(
#         'True label:' + str(y_test[k]) + '   likelihood of label ' + str(y_test[k]) + ': ' + str(softmax[k][int(y_test[k])]))
#     plt.colorbar();
# #     pp.savefig()
#
# pp.close()