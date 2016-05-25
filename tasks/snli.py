#!/usr/bin/python3
"""
KeraSTS interface for the SNLI dataset of the Textual Entailment task.

Training example:
	tools/train.py avg snli data/rte/snli/snli_1.0_train.pickle data/rte/snli/snli_1.0_dev.pickle vocabf="data/rte/snli/v1-vocab.pickle" inp_w_dropout=0.5

Before training, you must however run:

	tools/snli_preprocess.py --revocab data/rte/snli/snli_1.0/snli_1.0_train.jsonl data/rte/snli/snli_1.0/snli_1.0_dev.jsonl data/rte/snli/snli_1.0/snli_1.0_test.jsonl data/rte/snli/snli_1.0_train.pickle data/rte/snli/snli_1.0_dev.pickle data/rte/snli/snli_1.0_test.pickle data/rte/snli/v1-vocab.pickle
"""

from __future__ import print_function
from __future__ import division


from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Activation, Dropout, Lambda, Dense, RepeatVector, TimeDistributedDense, Reshape
from keras.models import Graph

import pickle

from keras.regularizers import l2

import pysts.eval as ev
import numpy as np

from pysts.kerasts import graph_input_anssel
import pysts.kerasts.blocks as B
from pysts.kerasts.callbacks import RTECB
from keras import backend as K


from .rte import RTETask

def get_H_n(X):
    ans=X[:, -1, :]  # get last element from time dim
    return ans


def get_Y(X):
    # TODO remove hardoced number
    return X[:, :60, :]  # get first xmaxlen elem from time dim

def get_R(X):
    Y, alpha = X.values()  # Y should be (None,L,k) and alpha should be (None,L,1) and ans should be (None, k,1)
    tmp=K.permute_dimensions(Y,(0,)+(2,1))  # copied from permute layer, Now Y is (None,k,L) and alpha is always (None,L,1)
    ans=K.T.batched_dot(tmp,alpha)
    return ans


class SnliTask(RTETask):
    def __init__(self):
        self.name = 'snli'
        self.spad = 60
        self.s0pad = self.spad
        self.s1pad = self.spad
        self.emb = None
        self.vocab = None

    def config(self, c):
        c['loss'] = 'categorical_crossentropy'
        c['nb_epoch'] = 32
        c['batch_size'] = 196
        c['epoch_fract'] = 1/4

    def load_vocab(self, vocabf):
        # use plain pickle because unicode
        self.vocab = pickle.load(open(vocabf, "rb"))
        return self.vocab

    def prep_model(self, module_prep_model):
        # Input embedding and encoding
        model = Graph()
        N = B.embedding(model, self.emb, self.vocab, self.s0pad, self.s1pad,
                        self.c['inp_e_dropout'], self.c['inp_w_dropout'], add_flags=self.c['e_add_flags'])

        # Sentence-aggregate embeddings
        final_outputs = module_prep_model(model, N, self.s0pad, self.s1pad, self.c)
        # Measurement
        L=self.s0pad
        k=N
        model.add_node(Lambda(get_H_n, output_shape=(k,)), name='h_n', input=final_outputs[1])

        #model.add_node(Lambda(get_Y, output_shape=(L, k)), name='Y', input=final_outputs[0])
        model.add_node(Dense(k,W_regularizer=l2(0.01)),name='Wh_n', input='h_n')
        model.add_node(RepeatVector(L), name='Wh_n_cross_e', input='Wh_n')
        model.add_node(TimeDistributedDense(k,W_regularizer=l2(0.01)), name='WY', input=final_outputs[0])
        model.add_node(Activation('tanh'), name='M', inputs=['Wh_n_cross_e', 'WY'], merge_mode='sum')
        model.add_node(TimeDistributedDense(1,activation='softmax'), name='alpha', input='M')
        #model.add_node(name='_r', inputs=['Y','alpha'], merge_mode='mul',
         #          layer=Activation('linear'))
        model.add_node(Lambda(get_R, output_shape=(k,1)), name='_r', inputs=['Y','alpha'], merge_mode='join')
        model.add_node(Reshape((k,)),name='r', input='_r')
        model.add_node(Dense(k,W_regularizer=l2(0.01)), name='Wr', input='r')
        model.add_node(Dense(k,W_regularizer=l2(0.01)), name='Wh', input='h_n')
        model.add_node(Activation('tanh'), name='h_star', inputs=['Wr', 'Wh'], merge_mode='sum')

        model.add_node(Dense(3, activation='softmax'), name='out', input='h_star')
        model.add_output(name='score', input='out')
        model.summary()

        return model

    def load_set(self, fname):
        si0, si1, sj0, sj1, f0_, f1_, labels = pickle.load(open(fname, "rb"))
        gr = graph_input_anssel(si0, si1, sj0, sj1, None, None, labels, f0_, f1_)
        return (gr, labels, self.vocab)


def task():
    return SnliTask()
