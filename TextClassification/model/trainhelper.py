#!/usr/bin/env python

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

def cross_validate(X, Y, clf_base, param_grid, n_folds=5, verb=False):
    """
    Cross validate using n folds for sklearn classifiers.
    Inputs:
        X: training samples. N*M numpy array, N is sample number, M is feature dimension.
        Y: training labels.
        clf_base: Initial classifier.
        param_grid: user defined parameter grid based on the property of classifier, which is dict format.
        n_folds: number of folds. Default is 5.
        verb: verbose mode.
    """
    clf_best = None
    acc_best = 0.
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    batch = n // n_folds
    
    clf = clf_base
    param_comb = comb(param_grid)
    print("Total number of fits: ", len(param_comb))
    
    for p in comb(param_grid):
        param = {}
        for k, key in enumerate(param_grid.keys()):
            param[key] = p[k]
        clf.set_params(**param)
        
        if verb:
            print("Current parameter setting: ")
            print(param)
        acc_avg = 0
        for i in range(n_folds):
            
            if isinstance(X, coo_matrix):
                x_va = X.tocsr()[idx[i*batch : (i+1)*batch]]
                x_tr = vstack((X.tocsr()[idx[0*batch : i*batch]], X.tocsr()[idx[(i+1)*batch :]]))
            elif isinstance(X, np.ndarray):
                x_va = X[idx[i*batch : (i+1)*batch]]
                x_tr = np.concatenate((X[idx[0*batch : i*batch]], X[idx[(i+1)*batch :]]), axis=0)
            
            y_va = Y[idx[i*batch : (i+1)*batch]]
            y_tr = np.concatenate((Y[idx[0*batch : i*batch]], Y[idx[(i+1)*batch :]]), axis=0)
            
            clf.fit(x_tr, y_tr)
            y_pred = clf.predict(x_va)
            acc = len(np.where(y_pred == y_va)[0]) / len(y_va)
            acc_avg += acc
            if verb:
                print("No.{} fold's accuracy: {:.6f}".format(i+1, acc))
            
        acc_avg /= n_folds
        if acc_avg > acc_best:
            acc_best = acc_avg
            clf_best = clf
        
    return clf_best, acc_best

def comb(dictionary):
    """
    Helper function for generating parameter table given parameter grid using DFS.
    Inputs:
        dictionary: parameter grid.
    Return:
        list of combinations of parameter settings
    """
    def helper(a, group, temp):
        nonlocal res
        if group == len(a):
            res.append(temp.copy())
            return
        
        for i in range(len(a[group])):
            temp.append(a[group][i])
            helper(a, group+1, temp)
            temp.pop(-1)
        
        return
    
    arr = list(dictionary.values())
    res = []
    temp = []
    helper(arr, 0, temp)
    
    return res

def cal_classification_score(y, y_pred, verb=True):

    res = []
    unique_class = np.unique(y)
    for c in unique_class:
        tp = len(np.where(np.logical_and(y==c, y_pred==c))[0])
        fp = len(np.where(np.logical_and(y!=c, y_pred==c))[0])
        tn = len(np.where(np.logical_and(y!=c, y_pred!=c))[0])
        fn = len(np.where(np.logical_and(y==c, y_pred!=c))[0])
        
        precision = tp/(tp + fp)
        recall = tp / (tp + fn)
        f1 = 2*precision*recall/(precision + recall)
        res.append([precision, recall, f1, tp, fp, tn, fn])
        
    acc_all = len(np.where(y == y_pred)[0]) / len(y)
    res = np.array(res)
    res_summary = np.mean(res[:, 0:3], axis=0)
    report = "Overall accuracy: {:.4f}\n\n".format(acc_all)
    report += "Class\tPrecision\tRecall\t\tf1\ttp\tfp\ttn\tfn\n"
    for i, row in enumerate(res):
        precision, recall, f1, tp, fp, tn, fn = row
        report += "{:d}\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t{:d}\t{:d}\t{:d}\t{:d}\n".format(
            i, precision, recall, f1, int(tp), int(fp), int(tn), int(fn))
    
    report += "\n"
    report += "{}\t{:.4f}\t\t{:.4f}\t\t{:.4f}\n".format(
            "avg", res_summary[0], res_summary[1], res_summary[2])
    
    if verb:
        print(report)
    
    return report, res

def get_result(x, y, clf, save_path=None, verb=True):
    """
    Get confusion matrix results in each topic and accuracy for total predictions.
    Inputs:
        x: testing samples
        y: testing labels
        clf: an instance of sklearn classifier.
    """
    y_pred = clf.predict(x)
    score, res_class = cal_classification_score(y, y_pred, verb=verb)
    confmatrix = confusion_matrix(y, y_pred)
    
    if verb:
        plt.figure(figsize=(14, 8))
        sn.heatmap(confmatrix, annot=True, cmap="Blues", fmt="d")
        plt.xlabel('Predict')
        plt.ylabel('Actual')
        plt.title("Confusion matrix")
        plt.show()
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump((score, confmatrix), f)
