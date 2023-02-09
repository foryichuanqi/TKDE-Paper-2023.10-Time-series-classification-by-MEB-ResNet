# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 21:13:39 2022

@author: Administrator
"""


# 

import numpy as np
arem_path = r""


X_train = np.load(arem_path + 'X_train.npy')
y_train = np.load(arem_path + 'y_train.npy')
X_test  = np.load(arem_path + 'X_test.npy')
y_test  = np.load(arem_path + 'y_test.npy')



# np.squeeze()
print(X_test.shape)

print(X_train.shape)

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


lenth=120
X_train,y_train=divide_x_y_by_lenth(X_train,y_train,lenth)
X_test,y_test=divide_x_y_by_lenth(X_test,y_test,lenth)




print(X_train.shape)
print(y_train.shape)
    
print(X_test.shape)
print(y_test.shape)    