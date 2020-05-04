import os
import os.path
from os import path

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from sklearn.metrics import roc_curve,roc_auc_score
from datetime import datetime,timedelta

def DrawPics(tensor,nb=0,template='{}',classnumber=None):
    if (nb==0):
        N = tensor.shape[0]
    else:
        N = min(nb,tensor.shape[0])
    fig=plt.figure(figsize=(16,2))
    nbligne = floor(N/20)+1
    for m in range(N):
        subplot = fig.add_subplot(nbligne,min(N,20),m+1)
        axis("off")
        plt.imshow(tensor[m,:,:,0],vmin=0, vmax=1,cmap='Greys')
        if (classnumber!=None):
            subplot.title.set_text((template.format(classnumber)))
            
def compute_metrics(probs,yprobs):
    '''
    Returns
        fpr : Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i]
        tpr : Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
        thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
        auc : Area Under the ROC Curve metric
    '''
    # calculate AUC
    auc = roc_auc_score(yprobs, probs)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(yprobs, probs)
    
    return fpr, tpr, thresholds,auc

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1],idx-1
    else:
        return array[idx],idx
    
def draw_roc(fpr, tpr,thresholds,auc,n_iteration):
    #find threshold
    targetfpr=1e-3
    _, idx = find_nearest(fpr,targetfpr)
    threshold = thresholds[idx]
    recall = tpr[idx]
    
    
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.title('AUC: {0:.3f} @ {4} iterations\nSensitivity : {2:.1%} @FPR={1:.0e}\nThreshold={3})'.format(auc,targetfpr,recall,abs(threshold),n_iteration ))
    # show the plot
    plt.show()
    
    
    
def compute_cache(name,func,parameter_dict={},folder="./"):
    '''
    Execute the function func and save the numpy result on disk based on given name. If the exact file exists 
    from a previous execution, the file content is loaded from disk intead and returned
    
    Input:
        name : file base name
        func : function to execute
        parameter_dict : dictionnay containing additional context parameters. filename will be appended with its values
        folder : folder on disk where to store the npy file
    '''
    filename = name
    explain = name + " for "
    
    #Compute the filename by appending all parameters
    for key in parameter_dict:
        filename += "_" + str(parameter_dict[key])
        explain += "{0}={1}, ".format(key,parameter_dict[key])
        
    #Numpy extention
    filename += ".npy"
    
    #If that thing is already computed, load from disk
    if (path.isfile(folder+filename)):
        print("Reloading "+explain)
        res = np.load(folder+filename,allow_pickle=True)
    else:
        #else compute it
        print("Computing "+explain)
        res = func()
        np.save(folder+filename,res)
        
    return res