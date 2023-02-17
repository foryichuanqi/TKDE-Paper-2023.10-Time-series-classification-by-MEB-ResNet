# Multi-Scale-Ensemble-Booster-for-Improving-Existing-Time-Series-Data-Classifiers

We proposed a highly easy-to-use performance enhancement framework called multi-scale ensemble booster (MEB), helping existing time series data classifiers achieve performance leaps. Besides, we proposed a new metric for evaluating time series classification peformance of methods on mutiple datasets, called MACE.

Paper: Multi-Scale Ensemble Booster for Improving Existing TSD Classifiers

Website of the paper :https://ieeexplore.ieee.org/document/9994036

In MEB, we proposed an easy-to-combine network structure without changing any of their structure and hyperparameters, only needed to set one hyperparameter, consisting of multi-scale transformation and multi-output decision fusion. Then, a probability distribution co-evolution strategy is proposed to attain the optimal label probability distribution. We conducted numerous ablation experiments of MEB on 128 univariate datasets and 29 multivariate datasets and comparative experiments with 11 state-of-the-art methods, which demonstrated the significant performance improvement ability of MEB and the most advanced performance of the model enhanced by MEB, respectively. Furthermore, to figure out why MEB can improve model performance, we provided a chain of interpretability analyses.

![image](https://user-images.githubusercontent.com/48144488/218241520-796791d0-f732-4dc4-afe2-09c0aab02f34.png)
![image](https://user-images.githubusercontent.com/48144488/218240414-6f22bef8-f6ae-4205-9325-4cc44bb50e7b.png)
![image](https://user-images.githubusercontent.com/48144488/218240457-3f706b3f-677f-4f79-8730-a0cf8a053a84.png)

# Easy to run successfully
To make code easy to run successfully, we debug the files carefully. Generally speaking, if environments are satisfied, you can directly run all the xxx.py files inside after decompressing the compressed package without changing any code.

(1) Download and rename Multi-Scale-Ensemble-Booster-for-Improving-Existing-Time-Series-Data-Classifiers.zip to MEB.zip (Rename to avoid errors caused by long directories)

(2) Unzip MEB.zip

(2) Run any xxx.py directly        

NOTE:  complete datasets can be downloaded in https://github.com/foryichuanqi/Multi-Scale-Ensemble-Booster-for-Improving-Existing-Time-Series-Data-Classifiers/releases/tag/v1.0.0

Dataset reference website: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ (UCR repository: univariate time series datasets) and https://github.com/titu1994/MLSTM-FCN/releases (Multivariate time series datasets) 

# Paper of Code and Citation

(1) To better understand our code, please read our paper.

Paper: Multi-Scale Ensemble Booster for Improving Existing TSD Classifiers
The website of the paper :https://ieeexplore.ieee.org/document/9994036

(2) Please cite this paper and the original source of the dataset when using the code for academic purposes.

BibTex:

@ARTICLE{multi2022fan,
  author={Fan, Linchuan and Chai, Yi and Chen, Xiaolong},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Multi-Scale Ensemble Booster for Improving Existing TSD Classifiers}, 
  year={2022},
  volume={},
  number={},
  pages={1-17},
  doi={10.1109/TKDE.2022.3230709}}

GB/T 7714: 

Fan L, Chai Y, Chen X. Multi-Scale Ensemble Booster for Improving Existing TSD Classifiers[J]. IEEE Transactions on Knowledge and Data Engineering, 2022.

# Relationship between Code and Paper

Method:

(1) MEB-ResNet: code\Ablation_experiment_on_128_UCR\MEB-ResNet_128_UCR.py
(2) MEB-FCN:code\Ablation_experiment_on_128_UCR\MEB-FCN_128_UCR.py

Our proposed metric for evaluating time series classification peformance of methods on mutiple datasets:

(1) MACE: Section 4.1 : the def MACE(x1,num_classes) in code\table\Table_3_4_5_6.py

Experiment:

(1) Section 4.2.1. :code\Ablation_experiment_of_pooling_methods_on_MTSC_datasets 
(2) Section 4.2.2 and 4.2.3. :code\Ablation_experiment_on_128_UCR and code\Ablation_and_comparision_experiment_on_MTSC_datasets
(3) Section  4.3. :code\Comparision_experiment_on_85_UCR and  code\Ablation_and_comparision_experiment_on_MTSC_datasets
 
# Environment and Acknowledgement:


(1) Environment:

tensorflow-gpu            1.15.0
    
keras                     2.2.4
    
scipy                     1.5.2
    
pandas                    1.0.5
    
numpy                     1.19.1



(2) Acknowledgement: 
Thanks for the following references sincerely.
   
github: https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline
   
github: https://github.com/titu1994/MLSTM-FCN/releases

UCR:https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ 
   
