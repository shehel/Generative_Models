import math, keras, datetime, pandas as pd, numpy as np, keras.backend as K, threading, json, re, collections
import tarfile, tensorflow as tf, matplotlib.pyplot as plt, xgboost, operator, random, pickle, glob, os, bcolz
import shutil, sklearn, functools, itertools, scipy
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import matplotlib.patheffects as PathEffects
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors, LSHForest
import IPython
from IPython.display import display, Audio
from numpy.random import normal
from gensim.models import word2vec
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import ToktokTokenizer, StanfordTokenizer
from functools import reduce
from itertools import chain

from tensorflow.python.framework import ops
#from tensorflow.contrib import rnn, legacy_seq2seq as seq2seq

from keras_tqdm import TQDMNotebookCallback
from keras import initializers  # Keras 1
from keras.applications.resnet50 import ResNet50, decode_predictions, conv_block, identity_block
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions, preprocess_input


np.set_printoptions(threshold=50, edgeitems=20)
def beep(): return Audio(filename='/home/hearth/Downloads/music/gegen.mp3', autoplay=True)
def dump(obj, fname): pickle.dump(obj, open(fname, 'wb'))
def load(fname): return pickle.load(open(fname, 'rb'))


def limit_mem():
    #K.get_session().close() Not required
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))


def autolabel(plt, fmt='%.2f'):
    rects = plt.patches
    ax = rects[0].axes
    y_bottom, y_top = ax.get_ylim()
    y_height = y_top - y_bottom
    for rect in rects:
        height = rect.get_height()
        if height / y_height > 0.95:
            label_position = height - (y_height * 0.06)
        else:
            label_position = height + (y_height * 0.01)
        txt = ax.text(rect.get_x() + rect.get_width()/2., label_position,
                fmt % height, ha='center', va='bottom')
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])


def column_chart(lbls, vals, val_lbls='%.2f'):
    n = len(lbls)
    p = plt.bar(np.arange(n), vals)
    plt.xticks(np.arange(n), lbls)
    if val_lbls: autolabel(p, val_lbls)


def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname): return bcolz.open(fname)[:]


def load_glove(loc):
    return (load_array(loc+'.dat'),
        pickle.load(open(loc+'_words.pkl','rb'), encoding='latin1'),
        pickle.load(open(loc+'_idx.pkl','rb'), encoding='latin1'))

def plot_multi(im, dim=(4,4), figsize=(6,6), **kwargs ):
    plt.figure(figsize=figsize)
    for i,img in enumerate(im):
        plt.subplot(*dim, i+1)
        plt.imshow(img, **kwargs)
        plt.axis('off')
    plt.tight_layout()


def plot_train(hist):
    h = hist.history
    if 'acc' in h:
        meas='acc'
        loc='lower right'
    else:
        meas='loss'
        loc='upper right'
    plt.plot(hist.history[meas])
    plt.plot(hist.history['val_'+meas])
    plt.title('model '+meas)
    plt.ylabel(meas)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc=loc)


def fit_gen(gen, fn, eval_fn, nb_iter):
    for i in range(nb_iter):
        fn(*next(gen))
        if i % (nb_iter//10) == 0: eval_fn()


def wrap_config(layer):
    return {'class_name': layer.__class__.__name__, 'config': layer.get_config()}


def copy_layer(layer): return layer_from_config(wrap_config(layer))


def copy_layers(layers): return [copy_layer(layer) for layer in layers]


def copy_weights(from_layers, to_layers):
    for from_layer,to_layer in zip(from_layers, to_layers):
        to_layer.set_weights(from_layer.get_weights())


def copy_model(m):
    res = Sequential(copy_layers(m.layers))
    copy_weights(m.layers, res.layers)
    return res


def insert_layer(model, new_layer, index):
    res = Sequential()
    for i,layer in enumerate(model.layers):
        if i==index: res.add(new_layer)
        copied = layer_from_config(wrap_config(layer))
        res.add(copied)
        copied.set_weights(layer.get_weights())
    return res

class BcolzArrayIterator1(object):
    """
    Returns an iterator object into Bcolz carray files
    Original version by Thiago Ramon Gonçalves Montoya
    Docs (and discovery) by @MPJansen
    Refactoring, performance improvements, fixes by Jeremy Howard j@fast.ai
        :Example:
        X = bcolz.open('file_path/feature_file.bc', mode='r')
        y = bcolz.open('file_path/label_file.bc', mode='r')
        trn_batches = BcolzArrayIterator(X, y, batch_size=64, shuffle=True)
        model.fit_generator(generator=trn_batches, samples_per_epoch=trn_batches.N, nb_epoch=1)
        :param X: Input features
        :param y: (optional) Input labels
        :param w: (optional) Input feature weights
        :param batch_size: (optional) Batch size, defaults to 32
        :param shuffle: (optional) Shuffle batches, defaults to false
        :param seed: (optional) Provide a seed to shuffle, defaults to a random seed
        :rtype: BcolzArrayIterator
        >>> A = np.random.random((32*10 + 17, 10, 10))
        >>> c = bcolz.carray(A, rootdir='test.bc', mode='w', expectedlen=A.shape[0], chunklen=16)
        >>> c.flush()
        >>> Bc = bcolz.open('test.bc')
        >>> bc_it = BcolzArrayIterator(Bc, shuffle=True)
        >>> C_list = [next(bc_it) for i in range(11)]
        >>> C = np.concatenate(C_list)
        >>> np.allclose(sorted(A.flatten()), sorted(C.flatten()))
        True
    """

    def __init__(self, X, y=None, w=None, batch_size=32, shuffle=False, seed=None):
        if y is not None and len(X) != len(y):
            raise ValueError('X (features) and y (labels) should have the same length'
                             'Found: X.shape = %s, y.shape = %s' % (X.shape, y.shape))
        if w is not None and len(X) != len(w):
            raise ValueError('X (features) and w (weights) should have the same length'
                             'Found: X.shape = %s, w.shape = %s' % (X.shape, w.shape))
        if batch_size % X.chunklen != 0:
            raise ValueError('batch_size needs to be a multiple of X.chunklen')

        self.chunks_per_batch = batch_size // X.chunklen
        self.X = X
        self.y = y if y is not None else None
        self.w = w if w is not None else None
        self.N = X.shape[0]
        self.batch_size = batch_size
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.shuffle = shuffle
        self.seed = seed


    def reset(self): self.batch_index = 0


    def next(self):
        with self.lock:
            if self.batch_index == 0:
                if self.seed is not None:
                    np.random.seed(self.seed + self.total_batches_seen)
                self.index_array = (np.random.permutation(self.X.nchunks + 1) if self.shuffle
                    else np.arange(self.X.nchunks + 1))

            #batches_x = np.zeros((self.batch_size,)+self.X.shape[1:])
            batches_x, batches_y, batches_w = [],[],[]
            for i in range(self.chunks_per_batch):
                current_index = self.index_array[self.batch_index]
                if current_index == self.X.nchunks:
                    batches_x.append(self.X.leftover_array[:self.X.leftover_elements])
                    current_batch_size = self.X.leftover_elements
                else:
                    batches_x.append(self.X.chunks[current_index][:])
                    current_batch_size = self.X.chunklen
                self.batch_index += 1
                self.total_batches_seen += 1

                idx = current_index * self.X.chunklen
                if not self.y is None: batches_y.append(self.y[idx: idx + current_batch_size])
                if not self.w is None: batches_w.append(self.w[idx: idx + current_batch_size])
                if self.batch_index >= len(self.index_array):
                    self.batch_index = 0
                    break

            batch_x = np.concatenate(batches_x)
            if self.y is None: return batch_x

            batch_y = np.concatenate(batches_y)
            return batch_x, batch_y

            #batch_w = np.concatenate(batches_w)
            #eturn batch_x, batch_y, b


    def __iter__(self): return self

    def __next__(self, *args, **kwargs): return self.next(*args, **kwargs)

    
class BcolzArrayIterator(object):
    """
    Returns an iterator object into Bcolz carray files
    Original version by Thiago Ramon Gonçalves Montoya
    Docs (and discovery) by @MPJansen
    Refactoring, performance improvements, fixes by Jeremy Howard j@fast.ai
        :Example:
        X = bcolz.open('file_path/feature_file.bc', mode='r')
        y = bcolz.open('file_path/label_file.bc', mode='r')
        trn_batches = BcolzArrayIterator(X, y, batch_size=64, shuffle=True)
        model.fit_generator(generator=trn_batches, samples_per_epoch=trn_batches.N, nb_epoch=1)
        :param X: Input features
        :param y: (optional) Input labels
        :param w: (optional) Input feature weights
        :param batch_size: (optional) Batch size, defaults to 32
        :param shuffle: (optional) Shuffle batches, defaults to false
        :param seed: (optional) Provide a seed to shuffle, defaults to a random seed
        :rtype: BcolzArrayIterator
        >>> A = np.random.random((32*10 + 17, 10, 10))
        >>> c = bcolz.carray(A, rootdir='test.bc', mode='w', expectedlen=A.shape[0], chunklen=16)
        >>> c.flush()
        >>> Bc = bcolz.open('test.bc')
        >>> bc_it = BcolzArrayIterator(Bc, shuffle=True)
        >>> C_list = [next(bc_it) for i in range(11)]
        >>> C = np.concatenate(C_list)
        >>> np.allclose(sorted(A.flatten()), sorted(C.flatten()))
        True
    """

    def __init__(self, X, X2=None, y=None, batch_size=32, shuffle=False, seed=None):
        if X2 is not None and len(X) != len(X2):
            raise ValueError('X (features) and X2 (features) should have the same length'
                             'Found: X.shape = %s, X2.shape = %s' % (X.shape, X2.shape))
        if y is not None and len(X) != len(y):
            raise ValueError('X (features) and y (labels) should have the same length'
                             'Found: X.shape = %s, y.shape = %s' % (X.shape, y.shape))
        if batch_size % X.chunklen != 0:
            raise ValueError('batch_size needs to be a multiple of X.chunklen')

        self.chunks_per_batch = batch_size // X.chunklen
        self.X = X
        self.X2 = X2 if X2 is not None else None
        self.y = y if y is not None else None
        self.N = X.shape[0]
        self.batch_size = batch_size
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.shuffle = shuffle
        self.seed = seed


    def reset(self): self.batch_index = 0


    def next(self):
        with self.lock:
            if self.batch_index == 0:
                if self.seed is not None:
                    np.random.seed(self.seed + self.total_batches_seen)
                self.index_array = (np.random.permutation(self.X.nchunks + 1) if self.shuffle
                    else np.arange(self.X.nchunks + 1))

            #batches_x = np.zeros((self.batch_size,)+self.X.shape[1:])
            batches_x, batches_x2, batches_y = [],[],[]
            for i in range(self.chunks_per_batch):
                current_index = self.index_array[self.batch_index]
                if current_index == self.X.nchunks:
                    batches_x.append(self.X.leftover_array[:self.X.leftover_elements])
                    batches_x2.append(self.X2.leftover_array[:self.X2.leftover_elements])
                    current_batch_size = self.X.leftover_elements
                else:
                    batches_x.append(self.X.chunks[current_index][:])
                    batches_x2.append(self.X2.chunks[current_index][:])
                    current_batch_size = self.X.chunklen
                self.batch_index += 1
                self.total_batches_seen += 1

                idx = current_index * self.X.chunklen
                if not self.y is None: batches_y.append(self.y[idx: idx + current_batch_size])
                if self.batch_index >= len(self.index_array):
                    self.batch_index = 0
                    break

            batch_x = np.concatenate(batches_x)
            if self.X2 is None: return batch_x

            batch_x2 = np.concatenate(batches_x2)
            batch_y = np.concatenate(batches_y)
            return ([batch_x, batch_x2],batch_y) 

            #batch_w = np.concatenate(batches_w)
            #return ([batch_x, batch_y], batch_w)


    def __iter__(self): return self

    def __next__(self, *args, **kwargs): return self.next(*args, **kwargs)