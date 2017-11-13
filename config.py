#-*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
from easydict import EasyDict as edict

# 开始创建相应的文件
if not os.path.exists("./cache/models/"): os.makedirs("./cache/models/")
# if not os.path.exists("./cache/h5/"): os.makedirs("./cache/h5/")
# if not os.path.exists("./logs/"): os.makedirs("./logs/")
# if not os.path.exists("./cache/gen_feats/"): os.makedirs("./cache/gen_feats/")

config = edict()
config.model_file = "./cache/models/rcnn_model.h5" # 训练结束后模型保存地址
config.TRAIN_DATA_DIR = './data/20_newsgroupsori'
config.GLOVE_DIR = './data/'
config.input_length = 1000 # 语句的最长长度
config.MAX_NB_WORDS = 20000 # words个数阈值
config.w2vDimension = 50 # word2vec维度
config.VALIDATION_SPLIT = 0.2  # 整个 *样本集* 划分给验证集的大小
config.unSupervisedRatio = 0.4  # 整个从样本集划分出来的 *训练集* 划分给半监督集的比率
config.use_semi_supervised = True # 是否使用半监督训练模型(False时，使用整个训练集训练模型，效果对比)
# config.use_semi_supervised = False # 最后的版本其实不用False 修改这一行 程序开始时便使用整个训练集训练模型 然后使用切分半监督数据
config.semi_batch = 6 # 将所有半监督中使用的unlabel样本分semi_batch个批依次加入模型训练，值越大越慢
# semi_batch 这个参数是自学习的阈值，详细信息请看README.md


config.kfold = 5 # 总样本集5折交叉验证,这里设置交叉折数
config.batch_size = 128 # 深度模型 分批训练的批量大小
config.epochs = 100 # 总共训练的轮数（实际不会超过该轮次，因为有early_stop限制）
config.rcnn_early_stop = 3 # 最优epoch的置信epochs
config.class_num = 20 



