# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 11:57:24 2020

@author: Administrator
"""
##########   read datastream from xlsx

####3sani
# import xlrd
# import numpy as np




##############read  table_data from xlsx
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


import os
print(os.path.abspath(os.path.join(os.getcwd(), "../..")))
last_last_path=os.path.abspath(os.path.join(os.getcwd(), "../.."))

print(os.path.abspath(os.path.join(os.getcwd(), "..")))
last_path=os.path.abspath(os.path.join(os.getcwd(), ".."))


#######TABLE_5_comparison_experiment
path = last_last_path +r'\table\TABLE_5_comparison_experiment.xlsx'
num_classes=[37,3,5,2,2,4,3,3,4,2,2,12,12,12,4,3,2,6,2,2,5,2,7,14,4,14,50,7,2,2,2,2,2,5,2,7,11,2,3,2,7,8,3,10,3,2,6,2,42,42,4,6,2,39,7,3,2,6,3,3,2,60,3,2,2,3,2,15,6,6,2,2,4,2,4,8,8,8,8,2,2,25,5,2,2]
#######0_TABLE_5_comparison_experiment



# ###TABLE_3_Ablation_experiment
path = last_last_path +r'\table\TABLE_3_Ablation_experiment.xlsx'
num_classes=[10,37,10,10,10,3,5,2,2,3,4,3,2,3,4,2,2,12,12,12,24,4,3,2,6,7,2,2,2,2,5,2,7,12,12,4,14,4,14,50,7,2,2,2,2,18,26,26,26,6,6,2,2,2,2,2,2,5,2,2,7,3,3,11,2,3,2,7,8,3,10,10,3,2,6,5,5,2,42,42,4,6,2,39,10,52,52,52,11,7,2,3,2,6,3,4,3,2,6,5,10,2,60,3,3,2,2,3,2,15,6,6,2,2,4,2,4,3,8,8,8,8,2,2,25,5,2,2]
# ###TABLE_3_Ablation_experiment


# path = r'F:\桌面11.17\project\fluid_based_time_series_calssification\table\4.2.1comparison_table_0_to_128_ResNet.xlsx'
# path = r'F:\桌面11.17\project\fluid_based_time_series_calssification\table\4.2.1comparison_table_0_to_128_FCN.xlsx'
data = pd.DataFrame(pd.read_excel(path))#读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字，此时不用使用DataFram，会报错


print(data.columns)#获取列的索引名称


############average rank calculate
rank_list=[]
for i in range(len(data.index)):
    rank_list.append(np.array(data.loc[i].rank()))
rank_array=np.array(rank_list)
average_rank=rank_array.mean(axis=0)
print(average_rank)

sc = MinMaxScaler(feature_range=(0, 1))  
#转换
average_rank=np.array(average_rank).reshape(-1,1)
average_rank_transformed = sc.fit_transform(average_rank)



######### average rank calculate


# average_geometric_rank=rank_array.prod(axis=0)
# print(average_geometric_rank)

################  best rank num  
rank_list=[]
for i in range(len(data.index)):
    rank_list.append(np.array(data.loc[i].rank(method='min')))  ###########choice  min https://blog.csdn.net/weixin_42926612/article/details/90265032
    # rank_list.append(np.array(data.loc[i].rank()))  ###########choice  min https://blog.csdn.net/weixin_42926612/article/details/90265032


rank_array=np.array(rank_list)

rank_array[rank_array>1]=0
best_rank=rank_array.sum(axis=0)
print(best_rank)
num_best_rank=best_rank.sum(axis=0)
print(num_best_rank)

sc = MinMaxScaler(feature_range=(0, 1))  
#转换
best_rank=np.array(best_rank).reshape(-1,1)
best_rank_transformed = sc.fit_transform(best_rank)

################   MPCE  MACE

def T_test_PCE(x1,x2,num_classes):
    
    PCE_x1=[]
    for i in range(len(x1)):
        PCE_x1.append(x1[i]/num_classes[i])
        
    print(PCE_x1)

    PCE_x2=[]
    for i in range(len(x1)):
        PCE_x2.append(x2[i]/num_classes[i])
        
    print(PCE_x2)
        
    t, p = stats.ttest_rel(PCE_x1, PCE_x2)
    print(t,p)
    
    
    print('aaaaaaaaaaa')
    
    return t,p

def wilcoxon_PCE(x1,x2,num_classes):
    
    PCE_x1=[]
    for i in range(len(x1)):
        PCE_x1.append(x1[i]/num_classes[i])
        
    print(PCE_x1)

    PCE_x2=[]
    for i in range(len(x1)):
        PCE_x2.append(x2[i]/num_classes[i])
        
    print(PCE_x2)
        
    t, p = stats.wilcoxon(PCE_x1, PCE_x2)
    print(t,p)
    
    
    print('aaaaaaaaaaa')
    
    return t,p

def MPCE(x1,num_classes):
    
    PCE_x1=[]
    for i in range(len(x1)):
        PCE_x1.append(x1[i]/num_classes[i])
    # print(PCE_x1)    
    MPCE_value=0
    for i in range(len(x1)):
        MPCE_value+=PCE_x1[i]
        
    MPCE_value=MPCE_value/len(x1)
    
    return MPCE_value

        

    


def T_test_ACE(x1,x2,num_classes):
    
    ACE_x1=[]
    for i in range(len(x1)):
        ACE_x1.append(x1[i]*num_classes[i])
    print(ACE_x1)

    ACE_x2=[]
    for i in range(len(x1)):
        ACE_x2.append(x2[i]*num_classes[i])
    print( ACE_x2)
        
    t, p = stats.ttest_rel(ACE_x1, ACE_x2)
    print(t,p)
    
    return t,p

def wilcoxon_ACE(x1,x2,num_classes):
    
    ACE_x1=[]
    for i in range(len(x1)):
        ACE_x1.append(x1[i]*num_classes[i])
    print(ACE_x1)

    ACE_x2=[]
    for i in range(len(x1)):
        ACE_x2.append(x2[i]*num_classes[i])
    print( ACE_x2)
        
    t, p = stats.wilcoxon(ACE_x1, ACE_x2)
    print(t,p)
    
    return t,p

def MACE(x1,num_classes):
    
    ACE_x1=[]
    for i in range(len(x1)):
        ACE_x1.append(x1[i]*num_classes[i])
    MACE=0
    for i in range(len(x1)):
        MACE+=ACE_x1[i]
    sum_class_num=0
    for i in range(len(num_classes)):
        sum_class_num+=num_classes[i]
    MACE=MACE/sum_class_num    
    # print(MACE/sum_class_num)
    return MACE

MPCE_list=[]
for i in range(len(data.columns)):
    MPCE_list.append(MPCE(data[list(data.columns)[i]],num_classes))
print('MPCE:{}'.format(MPCE_list))


sc = MinMaxScaler(feature_range=(0, 1))

MPCE_list=np.array(MPCE_list).reshape(-1,1)  
#转换
MPCE_list_transformed = sc.fit_transform(MPCE_list)




MACE_list=[]
for i in range(len(data.columns)):
    MACE_list.append(MACE(data[list(data.columns)[i]],num_classes))
print('MACE:{}'.format(MACE_list))

sc = MinMaxScaler(feature_range=(0, 1))  
#转换
MACE_list=np.array(MACE_list).reshape(-1,1)
MACE_list_transformed = sc.fit_transform(MACE_list)
    
##################################### T_test    Wilcoxon    


###T_test_PCE
list_triangle=[]
list_triangle_name=[]
for i in range(len(data.columns)):
    list_i=[]
    list_i_name=[]
    for j in range(i+1,len(data.columns)):
        list_i.append(T_test_PCE(data[data.columns[i]],data[data.columns[j]],num_classes)[1])#############  t  p
        list_i_name.append([data.columns[i],data.columns[j]])
        
    list_triangle.append(list_i)
    list_triangle_name.append(list_i_name)
print('T_test_PCE:{}'.format(list_triangle))
print('T_test_PCE_name:{}'.format(list_triangle_name))



###T_test_ACE
list_triangle=[]
list_triangle_name=[]
for i in range(len(data.columns)):
    list_i=[]
    list_i_name=[]
    for j in range(i+1,len(data.columns)):
        list_i.append(T_test_ACE(data[data.columns[i]],data[data.columns[j]],num_classes)[1])
        list_i_name.append([data.columns[i],data.columns[j]])
        
    list_triangle.append(list_i)
    list_triangle_name.append(list_i_name)
print('T_test_ACE:{}'.format(list_triangle))
print('T_test_ACE_name:{}'.format(list_triangle_name))


###wilcoxon_PCE
list_triangle=[]
list_triangle_name=[]
for i in range(len(data.columns)):
    list_i=[]
    list_i_name=[]
    for j in range(i+1,len(data.columns)):
        list_i.append(wilcoxon_PCE(data[data.columns[i]],data[data.columns[j]],num_classes)[1])
        list_i_name.append([data.columns[i],data.columns[j]])
        
    list_triangle.append(list_i)
    list_triangle_name.append(list_i_name)
print('wilcoxon_PCE:{}'.format(list_triangle))
print('wilcoxon_PCE_name:{}'.format(list_triangle_name))



###wilcoxon_ACE
list_triangle=[]
list_triangle_name=[]
for i in range(len(data.columns)):
    list_i=[]
    list_i_name=[]
    for j in range(i+1,len(data.columns)):
        list_i.append(wilcoxon_ACE(data[data.columns[i]],data[data.columns[j]],num_classes)[1])
        list_i_name.append([data.columns[i],data.columns[j]])
        
    list_triangle.append(list_i)
    list_triangle_name.append(list_i_name)
print('wilcoxon_ACE:{}'.format(list_triangle))
print('wilcoxon_ACE_name:{}'.format(list_triangle_name))
        




    
    



