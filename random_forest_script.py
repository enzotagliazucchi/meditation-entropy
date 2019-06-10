# -*- coding: utf-8 -*-

# trains and saves machine learning model to classify controls from meditators
# the trained model is dumped in a pickle file to test the generalization accuracy

import numpy as np
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import random
import pickle

clfs = []
clfs_shuffled = []
aucs = []
aucs_shuffled = []

def imputer_enzo(data):
        
    for n in np.arange(data.shape[1]):
        temp = data[:,n]
        temp[temp==0] = np.mean(temp[temp!=0])
        data[:,n] = temp
        
    return data

   
def get_optimal_thr_diagonal_cm(probs, target, step): 
    difference = np.zeros((len(np.arange(0,1,step))))
    n=-1
    for thr in np.arange(0,1,step):
        preds_thresholded = np.zeros(len(probs))
        n=n+1
        preds_thresholded[np.where(probs>thr)[0]] = 1
        cm = confusion_matrix(target, preds_thresholded).astype(float)
        cm[0,:] = cm[0,:]/float(sum(cm[0,:]))
        cm[1,:] = cm[1,:]/float(sum(cm[1,:]))
        difference[n] = abs(cm[0,0] - cm[1,1])
    loc = np.where( difference==min(difference))[0]
    return np.arange(0,1,step)[loc][0]
    
def unfold_data(data_list): 
    output = np.zeros((len(data_list), len(data_list[0][np.triu_indices(data_list[0].shape[0],1)])))
    for i,matrix in enumerate(data_list):
        output[i,:] = matrix[np.triu_indices(data_list[0].shape[0],1)]
    return output
    
for banda in np.arange(1,6):
        
    print(banda)
    n_estimators = 1000 
    n_folds = 2 
    numiter = 1000
    
    
    data = np.genfromtxt('/Users/enzo/Documents/work/paper_meditacion_entropia/machine_learning/data_VIP_classifier_band_'+str(banda)+'.csv',delimiter=",")
    
    target1 = np.zeros(16)
    target2 = np.ones(16)
   	
    target = np.concatenate((target1, target2), axis=0)
    
    for itera in np.arange(1, numiter):
    
        print(itera)
        
        cv = StratifiedKFold(n_splits=n_folds) 
            
        cv_target = np.array([])
        cv_prediction = np.array([])
        cv_probas = np.array([])
        cv_importances = np.zeros((n_folds, data.shape[1] ))
            
        for train, test in cv.split(data,target):
                
            X_train = data[train]
            X_test = data[test]
            y_train = target[train]
            y_test = target[test]
            
            X_train = imputer_enzo(copy.deepcopy(X_train))
            X_test = imputer_enzo(copy.deepcopy(X_test))
            
            clf = RandomForestClassifier(n_estimators=n_estimators) 
            clf = clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            probas = clf.predict_proba(X_test)
                
            cv_target = np.concatenate((cv_target, y_test), axis=0) 
            cv_prediction = np.concatenate((cv_prediction, preds), axis=0)
            cv_probas = np.concatenate((cv_probas, probas[:,1]), axis=0)
        
        preds_thr = np.zeros(len(cv_target))
        thr_final = get_optimal_thr_diagonal_cm(cv_probas, cv_target, 0.01)
        preds_thr[np.where(cv_probas>thr_final)[0]] = 1
        cm = confusion_matrix(cv_target, preds_thr).astype(float)
        cm[0,:] = cm[0,:]/float(sum(cm[0,:])) 
        cm[1,:] = cm[1,:]/float(sum(cm[1,:]))
                
            
        fpr, tpr, thresholds = roc_curve(cv_target,  cv_probas) 
        
        print(auc(fpr,tpr)) 
        
        aucs.append(auc(fpr,tpr))
        clfs.append(clf)

        
        target_s  = copy.deepcopy(target)
        
        random.shuffle(target_s)
        
        
        cv = StratifiedKFold(n_splits=n_folds) 
            
        cv_target = np.array([])
        cv_prediction = np.array([])
        cv_probas = np.array([])
        cv_importances = np.zeros((n_folds, data.shape[1] ))
            
        for train, test in cv.split(data,target_s):
                
            X_train = data[train] 
            X_test = data[test]
            y_train = target_s[train]
            y_test = target_s[test]
            
            X_train = imputer_enzo(copy.deepcopy(X_train))
            X_test = imputer_enzo(copy.deepcopy(X_test))
            
            clf = RandomForestClassifier(n_estimators=n_estimators) 
            clf = clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            probas = clf.predict_proba(X_test)
                
            cv_target = np.concatenate((cv_target, y_test), axis=0) 
            cv_prediction = np.concatenate((cv_prediction, preds), axis=0)
            cv_probas = np.concatenate((cv_probas, probas[:,1]), axis=0)
        
        preds_thr = np.zeros(len(cv_target))
        thr_final = get_optimal_thr_diagonal_cm(cv_probas, cv_target, 0.01)
        preds_thr[np.where(cv_probas>thr_final)[0]] = 1
        cm = confusion_matrix(cv_target, preds_thr).astype(float)
        cm[0,:] = cm[0,:]/float(sum(cm[0,:])) 
        cm[1,:] = cm[1,:]/float(sum(cm[1,:]))
                
            
        fpr, tpr, thresholds = roc_curve(cv_target,  cv_probas) 
        
        print(auc(fpr,tpr)) 
        
        aucs_shuffled.append(auc(fpr,tpr))
        clfs_shuffled.append(clf)
        
        
    data_save = {"aucs": aucs,"aucs_shuffled": aucs_shuffled,"clfs": clfs, "clfs_shuffled": clfs_shuffled}
        
    pickle.dump( data_save, open( '/Users/enzo/Documents/work/paper_meditacion_entropia/machine_learning/VIP_classifier_band_'+str(banda)+'.p', 'wb' ) )
