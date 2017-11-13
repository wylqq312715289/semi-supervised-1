#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import os,math,copy
from sklearn.utils import shuffle
import cv2
from tqdm import tqdm
import xml.etree.cElementTree as ET

from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

import keras
import tensorflow as tf
from keras import optimizers
from keras import backend as K
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, GlobalAveragePooling2D, concatenate
from keras.layers import Convolution1D, MaxPooling1D, Conv1D, GlobalAveragePooling1D
from keras.layers import Activation, Dropout, Flatten, Dense, Dropout, Input
from keras.layers import LSTM, Bidirectional, Lambda, TimeDistributed, Merge
from keras.layers.embeddings import Embedding
from keras.utils.data_utils import get_file
from keras.utils.np_utils import to_categorical

from utils import *
from config import config

class RcnnModel1(object):
	def __init__(self, n_symbols, embedding_weights, input_length, class_num):
		self.model_file = config.model_file
		self.early_stop = config.rcnn_early_stop
		self.n_symbols = n_symbols
		self.embedding_weights = embedding_weights
		self.input_length = input_length
		self.class_num = class_num
		self.hidden_dim_1 = 150
		self.hidden_dim_2 = 150

	def build_model(self):
		input_a = Input(shape=(self.input_length,))
		model = Sequential()
		emb =  Embedding(
			output_dim=config.w2vDimension,
			input_dim=self.n_symbols + 1,
			mask_zero=False,
			weights=[self.embedding_weights],
			input_length=self.input_length,
			trainable=False,)
		model.add( emb )
		model.add( Bidirectional(LSTM(self.hidden_dim_1, return_sequences=True)) )  
		model.add( TimeDistributed(Dense(self.hidden_dim_2, activation = "tanh")) )
		processed_a = model(input_a)    
		pool_rnn = Lambda(lambda x: K.max(x, axis = 1), output_shape = (self.hidden_dim_2, ))(processed_a)
		output = Dense(self.class_num, input_dim = self.hidden_dim_2, activation = "softmax")(pool_rnn)
		self.rcnn_model = Model(inputs = input_a, outputs = output)
		self.rcnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ["accuracy"])

	# 模型设置
	def fit_model(self, train_x, train_y, vali_x, vali_y):
		self.rcnn_model.fit(
			train_x, 
			train_y, 
			batch_size = config.batch_size, 
			epochs = config.epochs,
			validation_data = (vali_x, vali_y),
			callbacks = [
				# monitor=acc时 mode=max, patience是多少轮性能不好就停止
				EarlyStopping(monitor='val_acc', patience=self.early_stop, verbose=1 ),
				# ModelCheckpoint(filepath=self.model_file, monitor='val_acc', save_best_only=True),
				# LearningRateScheduler(lambda x: x/1.1),
			],
		)
		# self.rcnn_model = load_model(self.model_file)
	
	# 自学习模型
	def selfLearning( self, train_x, train_y, vali_x, vali_y ):
		# 随机取每一类样本一定比例作为无标签数据
		train_idx, semiS_idx = split_each_class(config.unSupervisedRatio, np.argmax(train_y, axis=1), True, self.class_num)
		print "train set size = %d. unlabel set size = %d."%(len(train_idx),len(semiS_idx))
		semiS_x, semiS_y = train_x[semiS_idx], train_y[semiS_idx]
		train_x, train_y = train_x[train_idx], train_y[train_idx]
		un_label_df = pd.DataFrame(np.array(range(len(semiS_idx))),columns=["idx"])
		un_label_df["unlabel"] = 0.0
		for k in range(config.semi_batch):
			self.build_model()
			self.fit_model( train_x, train_y, vali_x, vali_y )
			unlabel_idx = list(un_label_df[["idx"]][un_label_df["unlabel"]==0.0].values.reshape((-1,))) # 获取剩余unlabel集合
			print "#"*20+"  vali data set report  "+"#"*20
			pred_y = self.my_evaluate(vali_x,vali_y)
			print "#"*20+"  semiSupervised data set %d report  "%(k)+"#"*20
			pred_y = self.my_evaluate(semiS_x[unlabel_idx],semiS_y[unlabel_idx]) # 评测该轮半监督效果
			need_num = int( 1.0 * len(semiS_x) / config.semi_batch) # 将剩余unlabel集合中 need_num 个样本加入训练集
			best_batch_idx = sort_confidence_and_get_best_idx( pred_y, semiS_y[unlabel_idx], need_num )
			best_batch_idx = list( np.array(unlabel_idx)[best_batch_idx] )
			pred_y = np.argmax(pred_y, axis=1) # one-hot变成class_id
			pred_y = to_categorical(pred_y) # one-hot 处理
			train_x = np.concatenate((train_x,semiS_x[best_batch_idx]),axis=0)
			train_y = np.concatenate((train_y,semiS_y[best_batch_idx]),axis=0)
			un_label_df.ix[best_batch_idx,"unlabel"] = 1.0
			# self.rcnn_model.save(self.model_file)
			print "unlabel samples remains %d. "%(len(un_label_df[un_label_df["unlabel"]==0.0].index))

	# 训练整个模型
	def train_model( self, train_x, train_y, vali_x, vali_y ):
		print "train_model in MedlatModel ......"
		print "train num = %d."%(len(train_y))
		if os.path.exists(self.model_file): os.remove(self.model_file); print "Remove file %s."%(self.model_file)
		if os.path.exists(self.model_file): 
			self.rcnn_model = load_model(self.model_file)
		else:
			if config.use_semi_supervised:
				self.build_model()
				self.fit_model( train_x, train_y, vali_x, vali_y )
				pred_y = self.my_evaluate(vali_x,vali_y)
				self.selfLearning( train_x, train_y, vali_x, vali_y )
			else:
				self.build_model()
				self.fit_model( train_x, train_y, vali_x, vali_y )
				pred_y = self.my_evaluate(vali_x,vali_y)
	
	# 评估模型
	def my_evaluate( self, data_x, data_y ):
		pred_y = self.rcnn_model.predict(data_x)
		pred = np.argmax(pred_y, axis=1) # 求每行最大值索引
		data_y = np.argmax(data_y, axis=1)
		assert pred.shape==data_y.shape, "pred.shape!=data_y.shape"
		# print "confusion matrix: \n",metrics.confusion_matrix(data_y, pred)
		print "accuracy score: ",metrics.accuracy_score(data_y, pred)
		print "classification report:\n",metrics.classification_report(data_y, pred)
		# print "F1 score: ",metrics.f1_score(data_y, pred) # 多分类问题会报错
		return pred_y

class RcnnModel2(object):
	def __init__(self, n_symbols, embedding_weights, input_length, class_num):
		self.model_file = config.model_file
		self.early_stop = config.rcnn_early_stop
		self.n_symbols = n_symbols
		self.embedding_weights = embedding_weights
		self.input_length = input_length
		self.class_num = class_num

	def build_model(self):
		embedding_layer = Embedding(
			input_dim = self.n_symbols + 1,
            output_dim = config.w2vDimension,
            input_length = self.input_length,
            weights = [self.embedding_weights],
            trainable = True,)
		model_left = Sequential()
		#model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
		activation_type = 'tanh'
		model_left.add(embedding_layer)
		model_left.add(Conv1D(128, 5, activation=activation_type))
		model_left.add(MaxPooling1D(5))
		model_left.add(Conv1D(128, 5, activation=activation_type))
		model_left.add(MaxPooling1D(5))
		model_left.add(Conv1D(128, 5, activation=activation_type))
		model_left.add(MaxPooling1D(35))
		model_left.add(Flatten())

		#right model
		model_right = Sequential()
		model_right.add(embedding_layer)
		model_right.add(Conv1D(128, 4, activation=activation_type))
		model_right.add(MaxPooling1D(4))
		model_right.add(Conv1D(128, 4, activation=activation_type))
		model_right.add(MaxPooling1D(4))
		model_right.add(Conv1D(128, 4, activation=activation_type))
		model_right.add(MaxPooling1D(28))
		model_right.add(Flatten())

		#third model
		model_3 = Sequential()
		model_3.add(embedding_layer)
		model_3.add(Conv1D(128, 6, activation=activation_type))
		model_3.add(MaxPooling1D(3))
		model_3.add(Conv1D(128, 6, activation=activation_type))
		model_3.add(MaxPooling1D(3))
		model_3.add(Conv1D(128, 6, activation=activation_type))
		model_3.add(MaxPooling1D(30))
		model_3.add(Flatten())

		# merged = concatenate([model_left, model_right, model_3], axis=1)
		merged = Merge([model_left, model_right,model_3], mode='concat') # merge
		self.rcnn_model = Sequential()
		self.rcnn_model.add(merged) # add merge
		self.rcnn_model.add(Dense(128, activation=activation_type))
		self.rcnn_model.add(Dense(self.class_num, activation='softmax'))

		#model = Model(sequence_input, preds)
		self.rcnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# 模型设置
	def fit_model(self, train_x, train_y, vali_x, vali_y):
		self.rcnn_model.fit(
			train_x, 
			train_y, 
			batch_size = config.batch_size, 
			epochs = config.epochs,
			validation_data = (vali_x, vali_y),
			callbacks = [
				# monitor=acc时 mode=max, patience是多少轮性能不好就停止
				EarlyStopping(monitor='val_acc', patience=self.early_stop, verbose=1 ),
				# ModelCheckpoint(filepath=self.model_file, monitor='val_acc', save_best_only=True),
				# LearningRateScheduler(lambda x: x/1.1),
			],
		)
		# self.rcnn_model = load_model(self.model_file)
	
	# 自学习模型
	def selfLearning( self, train_x, train_y, vali_x, vali_y ):
		# 随机取每一类样本一定比例作为无标签数据
		train_idx, semiS_idx = split_each_class(config.unSupervisedRatio, np.argmax(train_y, axis=1), True, self.class_num)
		print "train set size = %d. unlabel set size = %d."%(len(train_idx),len(semiS_idx))
		semiS_x, semiS_y = train_x[semiS_idx], train_y[semiS_idx]
		train_x, train_y = train_x[train_idx], train_y[train_idx]
		un_label_df = pd.DataFrame(range(len(semiS_idx)),columns=["idx"])
		un_label_df["unlabel"] = 0.0
		for k in range(config.semi_batch):
			self.build_model()
			self.fit_model( train_x, train_y, vali_x, vali_y )
			unlabel_idx = list(un_label_df[["idx"]][un_label_df["unlabel"]==0.0].values.reshape((-1,))) # 获取剩余unlabel集合
			print "#"*20+"  vali data set report  "+"#"*20
			pred_y = self.my_evaluate(vali_x,vali_y)
			print "#"*20+"  semiSupervised data set %d report  "%(k)+"#"*20
			pred_y = self.my_evaluate(semiS_x[unlabel_idx],semiS_y[unlabel_idx]) # 评测该轮半监督效果
			need_num = int( 1.0 * len(semiS_x) / config.semi_batch) # 将剩余unlabel集合中 need_num 个样本加入训练集
			best_batch_idx = sort_confidence_and_get_best_idx( pred_y, semiS_y[unlabel_idx], need_num )
			best_batch_idx = list( np.array(unlabel_idx)[best_batch_idx] )
			pred_y = np.argmax(pred_y, axis=1) # one-hot变成class_id
			pred_y = to_categorical(pred_y) # one-hot 处理
			train_x = np.concatenate((train_x,semiS_x[best_batch_idx]),axis=0)
			train_y = np.concatenate((train_y,semiS_y[best_batch_idx]),axis=0)
			un_label_df.ix[best_batch_idx,"unlabel"] = 1.0
			# self.rcnn_model.save(self.model_file)
			print "unlabel samples remains %d. "%(len(un_label_df[un_label_df["unlabel"]==0.0].index))

	# 训练整个模型
	def train_model( self, train_x, train_y, vali_x, vali_y ):
		print "train_model in MedlatModel ......"
		print "train num = %d."%(len(train_y))
		if os.path.exists(self.model_file): os.remove(self.model_file); print "Remove file %s."%(self.model_file)
		if os.path.exists(self.model_file): 
			self.rcnn_model = load_model(self.model_file)
		else:
			if config.use_semi_supervised:
				self.build_model()
				self.fit_model( train_x, train_y, vali_x, vali_y )
				pred_y = self.my_evaluate(vali_x,vali_y)
				self.selfLearning( train_x, train_y, vali_x, vali_y )
			else:
				self.build_model()
				self.fit_model( train_x, train_y, vali_x, vali_y )
				pred_y = self.my_evaluate(vali_x,vali_y)
	
	# 评估模型
	def my_evaluate( self, data_x, data_y ):
		pred_y = self.rcnn_model.predict(data_x)
		pred = np.argmax(pred_y, axis=1) # 求每行最大值索引
		data_y = np.argmax(data_y, axis=1)
		assert pred.shape==data_y.shape, "pred.shape!=data_y.shape"
		# print "confusion matrix: \n",metrics.confusion_matrix(data_y, pred)
		print "accuracy score: ",metrics.accuracy_score(data_y, pred)
		print "classification report:\n",metrics.classification_report(data_y, pred)
		# print "F1 score: ",metrics.f1_score(data_y, pred) # 多分类问题会报错
		return pred_y

if __name__ == '__main__':
	pass

