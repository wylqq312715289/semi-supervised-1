#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os,copy,math,time,h5py
import numpy as np
import pandas as pd
import shutil
from datetime import datetime
from datetime import timedelta
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import StratifiedKFold
#from svmutil import svm_read_problem

from config import config

# 随机划分训练集和测试集 (非按类别等分)
def split_vali( data, labels_cat, split_ratio ):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels_cat = labels_cat[indices]

    num_validation_samples = int( 1.0 * split_ratio * data.shape[0] )
    X_train = data[:-num_validation_samples]
    y_train = labels_cat[:-num_validation_samples]
    X_val = data[-num_validation_samples:]
    y_val = labels_cat[-num_validation_samples:]
    return X_train, y_train, X_val, y_val

# 一般矩阵归一化
def my_normalization( data_ary, axis=0 ):
    # axis = 0 按列归一化; 1时按行归一化
    if axis == 1:
        data_ary = np.matrix(data_ary).T
        ans = preprocessing.scale(data_ary)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1.0,1.0))
        ans = min_max_scaler.fit_transform(ans)
        ans = np.matrix(ans).T
        ans = np.array(ans)
    else:
        ans = preprocessing.scale(data_ary)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1.0,1.0))
        ans = min_max_scaler.fit_transform(ans)
    return ans

# 保证样本每一类中均匀切分出一定比率，我们随机取其中的一定比率作为unlabel数据 返回下标
def split_each_class( split_ratio, labels, is_shuffle, class_num ):
    print "begin to split_each_class ......"
    data = pd.DataFrame( labels, columns = ["label"] );
    data["idx"] = range(len(labels));
    if is_shuffle: data = shuffle(data)
    split_out_idx = [] # 分给半监督样本集的下标索引
    split_resi_idx = [] # 分给半监督样本集后剩下的集合下标索引
    for class_id in range(class_num):
        idx_list = data[data["label"]==class_id].reset_index(drop=True)
        end_idx = int(split_ratio * len(idx_list))
        split_out_idx.extend( idx_list["idx"].values[:end_idx] )
        split_resi_idx.extend( idx_list["idx"].values[end_idx:] )
    split_out_idx = shuffle(split_out_idx)
    split_resi_idx = shuffle(split_resi_idx)
    return split_resi_idx, split_out_idx

# 自学习中置信度最大的加入到训练集
def sort_confidence_and_get_best_idx( pred, label, retuen_idx_len ):
    confidence = 1.0 - np.sum( (pred-label)**2, axis=1 ) # 置信度用欧氏距离判断
    item = zip( range(len(confidence)), confidence ) # 索引与置信度组合 
    item = sorted(item, key = lambda x: x[1], reverse=True) # 按照置信度排序，降序排列
    best_idx = np.array(item)[:retuen_idx_len,0] # 取置信度最高的retuen_idx_len个元素的下标索引
    best_idx = best_idx.astype(np.int)
    return list(best_idx)


# 保证样本每一类中均匀切分为k折，我们随机取其中的1折作为unlabel数据
def k_fold_split_each_class( fords, labels, is_shuffle, class_num ):
    print "begin to k_fold_split_each_class ......"
    skf = StratifiedKFold( n_splits = fords, shuffle=False, random_state=2017 )
    data = pd.DataFrame( labels, columns = ["label"] );
    data["idx"] = range(len(labels));
    if is_shuffle: data = shuffle(data)
    idx_box = [ [[],[]] for i in range(fords)];
    for class_id in range(class_num):
        idx_list = data[data["label"]==class_id].reset_index(drop=True)
        assert len(idx_list.index) >= fords, "len(idx_list.index)=%d <= 0"%(len(idx_list.index))
        for i, (train_idx, vali_idx) in enumerate(skf.split(idx_list.values, np.zeros((len(idx_list.index),))), 0):
            idx_box[i][0].extend( list( idx_list["idx"].values[train_idx] ) )
            idx_box[i][1].extend( list( idx_list["idx"].values[vali_idx]  ) )
    for i in range(fords):
        idx_box[i][0] = shuffle(idx_box[i][0])
        idx_box[i][1] = shuffle(idx_box[i][1])
    return idx_box # 返回均匀类别下标

