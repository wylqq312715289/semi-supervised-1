#-*- coding:utf-8 -*-
import os, sys
import numpy as np
CUDA_VISIBLE_DEVICES = "0" # 使用第 CUDA_VISIBLE_DEVICES 块GPU显卡
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from config import config
from modules import RcnnModel1, RcnnModel2
from utils import *

# 获取已知word2vec 长度文件名有写50 { word: vec_50, ... }
def get_embeddings_index():
    print('Indexing word vectors.')
    embeddings_index = {}
    f = open(os.path.join(config.GLOVE_DIR, 'glove.6B.50d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

# 从文本数据中提炼标签与文本数据
def load_data_labels(TEXT_DATA_DIR):
    print "begin to load_data_labels ......"
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                #if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,): f = open(fpath)
                else:                       f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i: t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)
    return texts, labels, labels_index

# 获取[ [vec_50]_wordid, .... }
def get_embedding_weights(word_index,embeddings_index):
    print('Preparing embedding matrix.')
    n_symbols = min(config.MAX_NB_WORDS, len(word_index)) # 训练语料使用的单词个数
    embedding_weights = np.zeros((n_symbols + 1, config.w2vDimension)) # 获取 [vec_50]_wordid, .... }
    for word, i in word_index.items():
        if i >= config.MAX_NB_WORDS: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_weights[i] = embedding_vector
    return n_symbols, embedding_weights

def main():
    ################# step: 1 获取{ word: vec_50, ... } ################# 
    # first, build index mapping words in the embeddings set to their embedding vector
    embeddings_index = get_embeddings_index()

    ################# step: 2 获取文件目录下的文本与label ################# 
    # second, prepare text samples and their labels
    print('Processing text dataset')
    texts, labels, labels_index = load_data_labels(config.TRAIN_DATA_DIR)
    print('Found %s texts.' % len(texts))

    ################# step: 3 ################# 
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(config.MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index #获取{ word:word_idx, ... }字典
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=config.input_length)
    labels_cat = to_categorical(np.asarray(labels)) # 拉成one-hot
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels_cat.shape)

    ################# step: 4 切分训练集与验证集  #################
    X_train, y_train, X_val, y_val = split_vali(data,labels_cat,config.VALIDATION_SPLIT)

    ################# step: 5 获取 [vec_50]_wordid, .... ] #################
    n_symbols, embedding_weights = get_embedding_weights(word_index,embeddings_index)
    class_num = y_train.shape[1] # 实际上是20类分类问题

    ################# step: 6 建立模型并训练 #################
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # set_session(tf.Session(config=tf_config))
    my_rcnn_model = RcnnModel1(n_symbols, embedding_weights, config.input_length, class_num)
    my_rcnn_model.train_model(X_train, y_train, X_val, y_val)

if __name__ == '__main__':
    main()















